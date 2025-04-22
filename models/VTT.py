from vit_pytorch.vit import pair, Transformer
import torch
from torch import nn
from einops.layers.torch import Rearrange

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
import gymnasium as gym

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from positional_encodings.torch_encodings import PositionalEncoding2D

from typing import Any, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule

import random

from stable_baselines3.common.policies import ActorCriticPolicy

from vit_pytorch.vit import Transformer

from utils.pretrain_utils import vt_load

from tqdm import tqdm

import torch.optim as optim

import numpy as np
import logging
import math
from functools import partial
from typing import Callable, Literal, Sequence, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from tactile_ssl.utils import apply_masks

from tactile_ssl.model.layers import MemEffAttention, Mlp
from tactile_ssl.model.layers import NestedTensorBlock as Block
from tactile_ssl.model.layers import PatchEmbed, PatchEmbed3D, SinusoidalEmbed, SwiGLUFFNFused

logger = logging.getLogger(__name__)


def named_apply(
    fn: Callable, module: nn.Module, name="", depth_first=True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module




class VTT(nn.Module):

    def __init__(self, *, 
                    image_size, 
                    tactile_size, 
                    image_patch_size, 
                    tactile_patch_size, 
                    dim,           # token的长度，embedding的输出维度
                    depth, 
                    heads, 
                    mlp_dim, 
                    image_channels = 3, 
                    tactile_channels=3, 
                    dim_head = 64, 
                    dropout = 0., 
                    emb_dropout = 0, 
                    num_tactiles=2, 
                    frame_stack=1,
                    pos_embed_fn: Literal["sinusoidal", "learned"] = "sinusoidal",
                    num_register_tokens: int = 0,   
                    # video related
                    num_frames: int = 1,
                 ):
        super().__init__()
        image_height, image_width = pair(image_size)
        tactile_height, tactile_width = pair(tactile_size)
        image_patch_height, image_patch_width = pair(image_patch_size)
        tactile_patch_height, tactile_patch_width = pair(tactile_patch_size)

        self.image_height = image_height
        self.image_width = image_width
        self.tactile_height = tactile_height
        self.tactile_width = tactile_width
        self.image_patch_height = image_patch_height
        self.image_patch_width = image_patch_width
        self.tactile_patch_height = tactile_patch_height
        self.tactile_patch_width = tactile_patch_width

        self.image_channels = image_channels
        self.tactile_channels = tactile_channels

        self.frame_stack = frame_stack

        assert image_height % image_patch_height == 0 and image_width % image_patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert tactile_height % tactile_patch_height == 0 and tactile_width % tactile_patch_width == 0, 'Tactile dimensions must be divisible by the patch size.'

        self.num_patches_image = (image_height // image_patch_height) * (image_width // image_patch_width)
        self.num_patches_tactile = (tactile_height // tactile_patch_height) * (tactile_width // tactile_patch_width) * num_tactiles

        self.num_patches = self.num_patches_image + self.num_patches_tactile

        image_patch_dim = image_channels * image_patch_height * image_patch_width
        tactile_patch_dim = tactile_channels * tactile_patch_height * tactile_patch_width
        
        self.image_to_patch_embedding = nn.Sequential(
            # Rearrange('b (n c) h w -> b c (n h) w', n = self.frame_stack, c = image_channels),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = image_patch_height, p2 = image_patch_width),
            nn.LayerNorm(image_patch_dim),
            nn.Linear(image_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.tactile_to_patch_embedding_1 = nn.Sequential(
            # Rearrange('b (n c) h w -> b c (n h) w', n = self.frame_stack, c = tactile_channels),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = tactile_patch_height, p2 = tactile_patch_width),
            nn.LayerNorm(tactile_patch_dim),
            nn.Linear(tactile_patch_dim, dim),
            nn.LayerNorm(dim)
        )
        self.tactile_to_patch_embedding_2 = nn.Sequential(
            # Rearrange('b (n c) h w -> b c (n h) w', n = self.frame_stack, c = tactile_channels),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = tactile_patch_height, p2 = tactile_patch_width),
            nn.LayerNorm(tactile_patch_dim),
            nn.Linear(tactile_patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()


        "new add"
        """
        添加norm属性
        """
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.num_register_tokens = num_register_tokens

        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, dim))
            if num_register_tokens
            else None
        )
        self.pos_embed_fn = pos_embed_fn

            # Video params
        self.num_frames = num_frames
        # transformer参数
        self.embed_dim = dim

        # 位置编码
        # 是将三张图片  一块编码
        if pos_embed_fn == "sinusoidal":
            # if self.is_video:
            #     self.pos_embed = SinusoidalEmbed(
            #         [num_frames] + list((image_patch_height, image_patch_width)),
            #         [
            #             self.num_frames // self.tubelet_size,
            #             self.patch_size,
            #             self.patch_size,
            #         ],
            #         embed_dim=self.embed_dim,
            #     )
            # else:
            self.pos_embed = SinusoidalEmbed(
                list((image_height*3, image_width)),  #list(self.img_size),  #*3是因为有3张
                [image_patch_size ,image_patch_size],
                embed_dim=dim,
            )
        # elif (
        #     pos_embed_fn == "learned"
        # ):  # NOTE: Different from DINOv2, we don't add learned positional embedding to cls / register tokens
            #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        
        self.norm = norm_layer(self.embed_dim)
        self.head = nn.Identity()        

        "权重初始化"
        self.init_weights()
        # self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.transformer):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    "未调，用来初始化"
    def init_weights(self):
        if self.pos_embed_fn == "learned":
            trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

        named_apply(init_weights_vit_timm, self)

    "这边我未处理乱七八糟的图像大小情况，直接按照一样的大小去处理，可能会报错"
    "改过之后没用到x参数，随便传"
    def interpolate_pos_encoding(self, img_shape, img_dtype, device):
        previous_dtype = img_dtype
        pos_embed = None
        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError("Unknown position embedding function")

        # if self.is_video:
        #     _, _, t, h, w = img_shape
        #     if h == self.img_size[0] and w == self.img_size[1] and t == self.num_frames:
        #         return pos_embed

        #     dim = pos_embed.shape[-1]
        #     t0 = t // self.tubelet_size
        #     w0 = w // self.patch_size
        #     h0 = h // self.patch_size

        #     pos_embed = nn.functional.interpolate(
        #         pos_embed.reshape(1, t0, w0, h0, dim).permute(0, 4, 1, 2, 3),
        #         mode="trilinear",
        #         antialias=self.interpolate_antialias,
        #         size=(t0, w0, h0),
        #     )
        #     assert (t0, w0, h0) == pos_embed.shape[-3:]
        #     pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        #     return pos_embed
        return pos_embed
        # _, _, h, w = img_shape
        # if h == self.img_size[0] and w == self.img_size[1]:
        #     return pos_embed

        # dim = pos_embed.shape[-1]
        # w0 = w // self.patch_size
        # h0 = h // self.patch_size

        # pos_embed = nn.functional.interpolate(
        #     pos_embed.reshape(1, w0, h0, dim).permute(0, 3, 1, 2),
        #     mode="bicubic",
        #     antialias=self.interpolate_antialias,
        #     size=(w0, h0),
        # )
        # assert (w0, h0) == pos_embed.shape[-2:]
        # pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # return pos_embed.to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        "实现pos embedding+modality embedding+mask"
        "*****************这里把x.device改成了device=x[1].device**************************"
        pos_encoding = self.interpolate_pos_encoding(x[1].shape, x[1].dtype, device=x[1].device)
        # print(pos_encoding.shape)   
        "实现embedding"
        embed1 = self.image_to_patch_embedding(x[0])
        embed2 = self.tactile_to_patch_embedding_1(x[1])
        embed3 = self.tactile_to_patch_embedding_2(x[2])
        # print(embed1.shape)
        embed1 = embed1 + pos_encoding[:,:embed1.shape[-2]]
        embed2 = embed2 + pos_encoding[:,embed1.shape[-2]:embed2.shape[-2]*2]
        embed3 = embed3 + pos_encoding[:,embed1.shape[-2]*2:]
        #x = torch.cat([embed1, embed2, embed3], dim=-2)  # dim=1 对应倒数第二个维度
        #x = x + pos_encoding
        #embed1 = x
        "实现mask"
        if masks is not None:
            # x = apply_masks(x, masks)
            embed1 = apply_masks(embed1,masks)
            embed2 = apply_masks(embed2,masks)
            embed3 = apply_masks(embed3,masks)

        x = torch.cat([embed1, embed2, embed3], dim=-2)  # dim=1 对应倒数第二个维度

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_regtokens": x_norm[:, : self.num_register_tokens],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        # if isinstance(x, list):
        #     return self.forward_features_list(x, masks)
        """需要修改forward规则使其满足视触图像要求,
        为了简化代码（其实是懒），我设置使得三者的patch相同，
        这样我只用根据patches大小生成一次mask就行"""
        x_list = []
        x_list.append(x['image'])
        x_list.append(x['tactile1'])
        x_list.append(x['tactile2'])
        x = self.prepare_tokens_with_masks(x_list, masks)
        "前向过程"
        # for blk in self.blocks:
        #     x = blk(x)
        #print(x.shape)
        x =self.transformer(x)

        x_norm = self.norm(x)
        return {
            "x_norm_regtokens": x_norm[:, : self.num_register_tokens],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_norm_patchtokens"]




class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = (224, 224),
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: int = 1,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        pos_embed_fn: Literal["sinusoidal", "learned"] = "learned",
        act_layer=nn.GELU,
        block_fn=Block,
        ffn_layer="mlp",
        block_chunks: int = 0,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            proj_bias (bool): enable bias for proj in attn if True
            ffn_bias (bool): enable bias for ffn if True
            drop_path_rate (float): stochastic depth rate
            drop_path_uniform (bool): apply uniform drop rate across blocks
            weight_init (str): weight init scheme
            init_values (float): layer-scale init values
            embed_layer (nn.Module): patch embedding layer
            act_layer (nn.Module): MLP activation layer
            block_fn (nn.Module): transformer block class
            ffn_layer (str): "mlp", "swiglu", "swiglufused" or "identity"
            block_chunks: (int) split block sequence into block_chunks units for FSDP wrap
            num_register_tokens: (int) number of extra cls tokens (so-called "registers")
            interpolate_antialias: (str) flag to apply anti-aliasing when interpolating positional embeddings
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        img_size = (
            (img_size, img_size) if isinstance(img_size, int) else tuple(img_size)
        )
        assert len(img_size) == 2, "Vision Transformer only works with 2D images"

        self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.n_blocks = depth

        # Video params
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = self.num_frames > 1

        self.num_heads = num_heads
        self.img_size = img_size
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.pos_embed_fn = pos_embed_fn

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                num_frames,
                tubelet_size=self.tubelet_size,
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        num_patches = self.patch_embed.num_patches

        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )
        if pos_embed_fn == "sinusoidal":
            if self.is_video:
                self.pos_embed = SinusoidalEmbed(
                    [num_frames] + list(self.img_size),
                    [
                        self.num_frames // self.tubelet_size,
                        self.patch_size,
                        self.patch_size,
                    ],
                    embed_dim=self.embed_dim,
                )
            else:
                self.pos_embed = SinusoidalEmbed(
                    list(self.img_size),
                    [self.patch_size, self.patch_size],
                    embed_dim=self.embed_dim,
                )
        elif (
            pos_embed_fn == "learned"
        ):  # NOTE: Different from DINOv2, we don't add learned positional embedding to cls / register tokens
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append(
                    [nn.Identity()] * i + blocks_list[i : i + chunksize]
                )
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.init_weights()
        self._rescale_blocks()

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self):
        if self.pos_embed_fn == "learned":
            trunc_normal_(self.pos_embed, std=0.02)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)

        named_apply(init_weights_vit_timm, self)

    def interpolate_pos_encoding(self, img_shape, img_dtype, device):
        previous_dtype = img_dtype
        pos_embed = None
        if self.pos_embed_fn == "sinusoidal":
            pos_embed = self.pos_embed(device).float().unsqueeze(0)
        elif self.pos_embed_fn == "learned":
            pos_embed = self.pos_embed.float()
        else:
            raise NotImplementedError("Unknown position embedding function")

        if self.is_video:
            _, _, t, h, w = img_shape
            if h == self.img_size[0] and w == self.img_size[1] and t == self.num_frames:
                return pos_embed

            dim = pos_embed.shape[-1]
            t0 = t // self.tubelet_size
            w0 = w // self.patch_size
            h0 = h // self.patch_size

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, t0, w0, h0, dim).permute(0, 4, 1, 2, 3),
                mode="trilinear",
                antialias=self.interpolate_antialias,
                size=(t0, w0, h0),
            )
            assert (t0, w0, h0) == pos_embed.shape[-3:]
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        _, _, h, w = img_shape
        if h == self.img_size[0] and w == self.img_size[1]:
            return pos_embed

        dim = pos_embed.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size

        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, w0, h0, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            size=(w0, h0),
        )
        assert (w0, h0) == pos_embed.shape[-2:]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):

        pos_encoding = self.interpolate_pos_encoding(x.shape, x.dtype, device=x.device)
        x = self.patch_embed(x)
        x = x + pos_encoding

        if masks is not None:
            x = apply_masks(x, masks)

        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        return x

    def forward_features_list(self, x_list, masks_list):
        x = [
            self.prepare_tokens_with_masks(x, masks)
            for x, masks in zip(x_list, masks_list)
        ]
        for blk in self.blocks:
            x = blk(x)

        all_x = x
        output = []
        for x, masks in zip(all_x, masks_list):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm_regtokens": x_norm[:, : self.num_register_tokens],
                    "x_norm_patchtokens": x_norm[:, self.num_register_tokens :],
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = blk(x)

        x_norm = self.norm(x)
        return {
            "x_norm_regtokens": x_norm[:, : self.num_register_tokens],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def _get_intermediate_layers_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        # If n is an int, take the n last blocks. If it's a list, take them
        blocks_to_take = (
            range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        )
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # Passing the nn.Identity()
                x = blk(x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        assert len(output) == len(
            blocks_to_take
        ), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, *args, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        return ret["x_norm_patchtokens"]



def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """vit weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)













"测试代码"
# 在文件末尾添加以下代码
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建模型参数
    batch_size = 1
    image_size = (64, 64)
    tactile_size = (32, 32)
    image_patch_size = 8
    tactile_patch_size = 4
    dim = 256
    depth = 4
    heads = 8
    mlp_dim = 512
    num_tactiles = 2
    num_register_tokens = 1  # 添加一个register token用于DINO
    
    # 创建模型实例
    model = VTT(
        image_size=image_size,
        tactile_size=tactile_size,
        image_patch_size=image_patch_size,
        tactile_patch_size=tactile_patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        num_tactiles=num_tactiles,
        num_register_tokens=num_register_tokens
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 创建随机输入数据
    x = {}
    x[1] = torch.randn(batch_size, 3, image_size[0], image_size[1]).to(device)  # 图像
    x[2] = torch.randn(batch_size, 3, tactile_size[0], tactile_size[1]).to(device)  # 触觉1
    x[3] = torch.randn(batch_size, 3, tactile_size[0], tactile_size[1]).to(device)  # 触觉2
    
    # 将输入数据也放入字典格式，便于兼容其他调用方式
    x_dict = {
        'image': x[1],
        'tactile1': x[2],
        'tactile2': x[3],
    }
    
    print("\n===== 测试基本前向传播 =====")
    # 测试prepare_tokens_with_masks方法
    embeddings = model.forward_features(x_dict)
    print(f"Embeddings shape after prepare_tokens_with_masks: {embeddings['x_norm_regtokens'].shape}")
    print(f"Embeddings shape after prepare_tokens_with_masks: {embeddings['x_norm_patchtokens'].shape}")
    print(f"Embeddings shape after prepare_tokens_with_masks: {embeddings['x_prenorm'].shape}")
    
    # 测试transformer处理
    transformer_output = model.transformer(embeddings['x_norm_patchtokens'])
    print(f"Transformer output shape: {transformer_output.shape}")
    
    # 测试完整的forward_features方法
    try:
        features = model.forward_features(x_dict)
        print(f"Features output:")
        for k, v in features.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
            else:
                print(f"  {k}: {type(v)}")
    except Exception as e:
        print(f"Error in forward_features: {e}")
    
    print("\n===== 测试带掩码的前向传播 =====")
    # 创建一些简单的掩码进行测试
    try:
        # 创建一个简单的掩码，屏蔽一些patch
        num_patches = model.num_patches_image + model.num_patches_tactile
        masks = [(torch.randint(0, num_patches, (10,)).to(device),) for _ in range(2)]
        print(masks[0].shape)
        masked_embeddings = model.prepare_tokens_with_masks(x, masks)
        print(f"Masked embeddings shape: {masked_embeddings.shape}")
        
        masked_transformer_output = model.transformer(masked_embeddings)
        print(f"Masked transformer output shape: {masked_transformer_output.shape}")
    except Exception as e:
        print(f"Error in masked forward pass: {e}")
    
    print("\n===== 测试完成 =====")

if __name__ == "__main__":
    main()