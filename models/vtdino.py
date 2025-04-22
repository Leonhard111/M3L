# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from functools import partial
import einops
from omegaconf import ListConfig

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

from tactile_ssl.algorithm import Module
from tactile_ssl.loss.dino_loss import DINOLoss
from tactile_ssl.utils.logging import get_pylogger
from tactile_ssl.utils.logging import img_logger
from tactile_ssl.utils.ema import update_moving_average
from tactile_ssl.utils import patchify_image, patches_to_image


log = get_pylogger(__name__)


class VTDINO(Module, nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        dino_head: partial,
        optim_cfg: partial,
        lr_scheduler_cfg: Optional[partial],
        wd_scheduler_cfg: Optional[partial],
        online_probes: Optional[List[nn.Module]] = None,
        online_probes_lrs: List[float] = [],
        local_mask_scale: Tuple[float, float] = (0.2, 0.8),     #(0.2, 0.8),
        global_mask_scale: Tuple[float, float] = (0.2, 0.8),
        num_global_masks: int = 1,
        num_local_masks: int = 4,
        min_keep_num_sensors: int = 4,
        allow_mask_overlap: bool = False,
        moving_average_decay: Union[float, Tuple[float, ...]] = 0.99,
        teacher_temp: Union[float, Tuple[float, ...]] = (0.04, 0.07),
        teacher_warmup_epochs: int = 10,
        use_momentum=True,
        log_freq_reconstruction: int = 1000,
    ):
        super().__init__()
        self.optim_partial = optim_cfg
        self.lr_scheduler_partial = lr_scheduler_cfg
        self.wd_scheduler_partial = wd_scheduler_cfg
        self.use_momentum = use_momentum
        self.global_mask_scale = global_mask_scale
        self.local_mask_scale = local_mask_scale
        self.num_global_masks = num_global_masks
        self.num_local_masks = num_local_masks
        self.min_keep = min_keep_num_sensors
        self.allow_mask_overlap = allow_mask_overlap
        self.log_freq_img = log_freq_reconstruction

        self.generator = torch.Generator()
        self.step = -1

        # Encoders
        dino_head = partial(dino_head, in_dim=encoder.embed_dim)

        self.student_encoder_dict, self.teacher_encoder_dict = dict(), dict()
        self.student_encoder_dict["backbone"] = encoder
        self.student_encoder_dict["dino_head"] = dino_head()
        self.student_encoder = nn.ModuleDict(self.student_encoder_dict)

        self.teacher_encoder_dict["backbone"] = copy.deepcopy(encoder)
        self.teacher_encoder_dict["dino_head"] = dino_head()
        self.teacher_encoder = nn.ModuleDict(self.teacher_encoder_dict)
        self.teacher_encoder.requires_grad_(False)

        self.dino_loss = DINOLoss(
            out_dim=self.student_encoder_dict["dino_head"].last_layer.out_features
        )

        self.patch_size = encoder.image_patch_height       # 用VTT中视觉图像的高来替代
        self.img_size = encoder.image_height                # 用视觉的高来代替
        self.in_chans = encoder.image_channels          # y用视觉channels来代替，注意检查T,V的通道数相同

        self.online_probes = (
            [] if online_probes is None else nn.ModuleList(online_probes)
        )
        self.online_probes_lrs = online_probes_lrs
        assert len(self.online_probes) == len(
            online_probes_lrs
        ), "Number of online probes should match the number of learning rates"

        # Momentum scheduler if moving average decay is a tuple
        self.momentum_scheduler = None
        if not isinstance(moving_average_decay, float):
            assert isinstance(moving_average_decay, list) or isinstance(
                moving_average_decay, ListConfig
            )
            assert len(moving_average_decay) == 2
            moving_average_decay = tuple(moving_average_decay)
        self.moving_average_decay = moving_average_decay

        self.teacher_temp_scheduler = None
        if not isinstance(teacher_temp, float):
            assert isinstance(teacher_temp, list) or isinstance(
                teacher_temp, ListConfig
            )
            assert len(teacher_temp) == 2
            teacher_temp = tuple(teacher_temp)
        self.teacher_temp = teacher_temp
        self.teacher_warmup_epochs = teacher_warmup_epochs
        self.val_reconstruction_error = []

    def log_on_batch_end(
        self, outputs, stage: Literal["train", "val"] = "train", trainer_instance=None
    ):
        loss = outputs["loss"]
        ssl_loss = outputs["ssl_loss"]
        if trainer_instance is not None and trainer_instance.should_log:
            step = trainer_instance.global_val_step if stage == 'val' else trainer_instance.global_step 
            
            trainer_instance.wandb.log(
                {f"{stage}/loss": loss, f"global_{stage}_step": step}
            )
            trainer_instance.wandb.log(
                {f"{stage}/ssl_loss": ssl_loss, f"global_{stage}_step": step}
            )
            if "ppl_loss" in outputs.keys():
                ppl_loss = outputs["ppl_loss"]
                trainer_instance.wandb.log(
                    {f"{stage}/ppl_loss": ppl_loss, f"global_{stage}_step": step}
                )

            for probe in self.online_probes:
                probe_name = probe.probe_name
                probe_loss = outputs.get(f"{probe_name}_loss", None)
                if probe_loss is not None:
                    trainer_instance.wandb.log(
                        {
                            f"{stage}/{probe_name}_loss": probe_loss,
                            f"global_{stage}_step": step,
                        }
                    )
            trainer_instance.wandb.log(
                {
                    f"{stage}/teacher_temperature": self.current_teacher_temp,
                    f"global_{stage}_step": step,
                }
            )

    def on_train_batch_end(self, outputs, batch, batch_idx, trainer_instance=None):
        assert self.teacher_encoder is not None, "target encoder has not been created"
        self.current_teacher_temp = (
            next(self.teacher_temp_scheduler)
            if self.teacher_temp_scheduler is not None
            else self.teacher_temp
        )
        if self.use_momentum:
            moving_average_decay = (
                next(self.momentum_scheduler)
                if self.momentum_scheduler is not None
                else self.moving_average_decay
            )
            with torch.no_grad():
                update_moving_average(
                    self.teacher_encoder,
                    self.student_encoder,
                    moving_average_decay,
                )
        "没有log"
        self.log_on_batch_end(outputs, stage="train", trainer_instance=trainer_instance)

    def on_validation_batch_end(
        self, outputs: Dict, batch: Dict, batch_idx: int, trainer_instance=None
    ):
        self.log_on_batch_end(outputs, stage="val", trainer_instance=trainer_instance)
        # Plot online probe predictions
        step = trainer_instance.global_val_step
        if trainer_instance is not None and trainer_instance.should_log:
            trainer_instance.wandb.log(
                {
                    f"val/loss": outputs["loss"],
                    f"global_val_step": step,
                }
            )
            if (step % self.log_freq_img == 0) and "pred_img" in outputs.keys():
                Xpred = outputs["pred_img"]
                Xorg = outputs["gt_img"] if "gt_img" in outputs.keys() else None
                if Xorg is not None:
                    self.val_reconstruction_error.append(
                        torch.mean((Xpred - Xorg) ** 2, dim=[1, 2, 3])
                    )
                img_logger(
                    wandb=trainer_instance.wandb,
                    global_step=step,
                    predictions=Xpred,
                    X=Xorg,
                    label="val",
                )

    def on_validation_epoch_end(self, trainer_instance=None):
        if len(self.val_reconstruction_error) > 0:
            reconstruction_error = torch.cat(self.val_reconstruction_error, dim=0)
            root_mean_square_error = torch.sqrt(torch.mean(reconstruction_error, dim=0))
            print(f"RMSE: {root_mean_square_error}")
            trainer_instance.wandb.log({"val/rmse": root_mean_square_error})
            self.val_reconstruction_error = []

    def _sample_block_size(self, height, width, scale):
        _rand = torch.rand(1, generator=self.generator).item()
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(height * width * mask_scale)
        aspect_ratio = 1.0
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h > height:
            h -= 1
        while w > width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, height, width, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """Helper to restrict given mask to a set of acceptable regions"""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, height - h + 1, (1,), generator=self.generator)
            left = torch.randint(0, width - w + 1, (1,), generator=self.generator)
            mask = torch.zeros((height, width), dtype=torch.int32)
            mask_complement = torch.ones_like(mask)
            mask[top : top + h, left : left + w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    log.warning(
                        f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"'
                    )
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((height, width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement

    def sample_masks(self, x):

        batch_size, _, image_height, image_width = x.shape
        height, width = (
            image_height // self.patch_size,
            image_width // self.patch_size,
        )
        #print(height,width)

        local_maskblock_sizes = self._sample_block_size(
            height, width, self.local_mask_scale
        )
        global_maskblock_sizes = self._sample_block_size(
            height, width, self.global_mask_scale
        )
        #print(local_maskblock_sizes,global_maskblock_sizes)
        collated_local_masks, collated_global_masks = [], []
        min_keep_local_patches, min_keep_global_patches = (
            height * width,
            height * width,
        )
        for _ in range(batch_size):
            masks_local, masks_complement = [], []
            for _ in range(self.num_local_masks):
                mask, mask_complement = self._sample_block_mask(
                    height, width, local_maskblock_sizes
                )
                masks_local.append(mask)
                masks_complement.append(mask_complement)
                min_keep_local_patches = min(min_keep_local_patches, len(mask))
            collated_local_masks.append(masks_local)

            acceptable_regions = masks_complement

            if self.allow_mask_overlap:
                acceptable_regions = None

            masks_encoder = []
            for _ in range(self.num_global_masks):
                mask, _ = self._sample_block_mask(
                    height, width, global_maskblock_sizes, acceptable_regions
                )
                masks_encoder.append(mask)
                min_keep_global_patches = min(min_keep_global_patches, len(mask))
            collated_global_masks.append(masks_encoder)

        collated_global_masks = [
            [cm[:min_keep_global_patches] for cm in masks]
            for masks in collated_global_masks
        ]
        collated_local_masks = [
            [cm[:min_keep_local_patches] for cm in masks]
            for masks in collated_local_masks
        ]
        local_masks = data.default_collate(collated_local_masks)
        global_masks = data.default_collate(collated_global_masks)

        for i in range(len(global_masks)):
            global_masks[i] = global_masks[i].to(x.device)
        for i in range(len(local_masks)):
            local_masks[i] = local_masks[i].to(x.device)
        #print(min_keep_local_patches, min_keep_global_patches)
        return global_masks, local_masks

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]], # : torch.Tensor,
        global_masks: List[torch.Tensor],
        local_masks: List[torch.Tensor],
    ):
        assert (
            global_masks is not None and local_masks is not None
        ), "Masks are required for DINOModule during training"

        # TODO: @Akash Sharma - Raise to make sure context encoder implements taking masks as an argument
        student_global_dict = self.student_encoder_dict["backbone"].forward_features(
            x, global_masks
        )
        assert (
            "x_norm_regtokens" in student_global_dict.keys()
        ), "Dino requires backbone to contain 1 register token"
        student_global_cls_tokens = student_global_dict["x_norm_regtokens"]
        student_global_cls_tokens = einops.rearrange(student_global_cls_tokens, "(p b) 1 c -> b p c", p=len(global_masks))
        
        student_local_dict = self.student_encoder_dict["backbone"].forward_features(
            x, local_masks
        )
        student_local_cls_tokens = student_local_dict["x_norm_regtokens"]
        student_local_cls_tokens = einops.rearrange(
            student_local_cls_tokens, "(p b) 1 c -> b p c", p=len(local_masks)
        )
        student_cls_tokens = torch.cat(
            [student_global_cls_tokens, student_local_cls_tokens], dim=-2
        )
        student_cls_tokens_after_head = self.student_encoder_dict["dino_head"](
            student_cls_tokens
        )
        student_cls_tokens_after_head = einops.rearrange(
            student_cls_tokens_after_head, "b p c -> p b 1 c"
        )

        with torch.no_grad():
            teacher_global_dict = self.teacher_encoder_dict[
                "backbone"
            ].forward_features(x, global_masks)
            teacher_global_cls_tokens = teacher_global_dict["x_norm_regtokens"]
            teacher_cls_tokens_after_head = self.teacher_encoder_dict["dino_head"](
                teacher_global_cls_tokens
            )
            teacher_cls_tokens_after_head = teacher_cls_tokens_after_head.detach()

            teacher_dino_softmaxed_centered_list = (
                self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head,
                    teacher_temp=self.current_teacher_temp,
                ).view(
                    self.num_global_masks, -1, *teacher_cls_tokens_after_head.shape[1:]
                )
            )
            self.dino_loss.update_center(teacher_cls_tokens_after_head)

        loss = self.dino_loss(
            list(student_cls_tokens_after_head),
            list(teacher_dino_softmaxed_centered_list),
        )

        return loss

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        self.step = self.step + 1
        self.generator.manual_seed(self.step)

        x = batch["image"]
        global_masks, local_masks = self.sample_masks(x)

        # print(global_masks[0].shape)
        # print(local_masks[0].shape)
        # print(batch["image"].shape)
        loss = self.forward(batch, global_masks, local_masks)   #  loss = self.forward(x, global_masks, local_masks)

        output = {
            "ssl_loss": loss.item(),
        }

        # online probes
        if len(self.online_probes) > 0:
            with torch.no_grad():
                teacher_dict = self.teacher_encoder_dict["backbone"].forward_features(x)
                embedding = teacher_dict["x_norm_patchtokens"]
                embedding = F.layer_norm(embedding, (embedding.size(-1),))
                x_patch_gt = patchify_image(x, self.patch_size)

        online_probes_loss = 0.0
        pred_img = None
        # TODO: Currently only reconstruction probe is supported. Missing targets for other probes
        for probe in self.online_probes:
            probe_name = probe.probe_name
            if probe_name == "reconstruction":
                probe_loss, decoded_x = probe(embedding, target=x_patch_gt, img_shape=x.shape)
                online_probes_loss += probe_loss
                output[f"{probe_name}_loss"] = probe_loss.item()
                # custom for reconstruction probe to get the image
                pred_img = patches_to_image(decoded_x.detach(), self.patch_size, self.img_size).float()
                # We only visualize the first image from the two concatenated images 
                output["pred_img"] = pred_img[:, 0:3, :, :] 
                output["gt_img"] = x[:, 0:3, :, :]
            else:
                raise NotImplementedError(f"Probe {probe_name} missing target")

        loss += online_probes_loss

        output["loss"] = loss  # type: ignore
        output["online_probes_loss"] = online_probes_loss

        return output

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict:
        return self.training_step(batch, batch_idx)

    def configure_optimizers(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, num_iterations_per_epoch, num_epochs
    ) -> Tuple[torch.optim.Optimizer, Optional[Dict], Optional[Dict]]:
        param_dict = {
            pn: p
            for pn, p in self.named_parameters()
            if not pn.startswith("online_probes")
        }
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params},
            {"params": nodecay_params, "WD_exclude": True, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        for probe, lr in zip(self.online_probes, self.online_probes_lrs):
            trainable_probe_params = {
                pn: p for pn, p in probe.named_parameters() if p.requires_grad
            }
            optim_groups.append({"params": trainable_probe_params.values(), "lr": lr})

        log.info(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        log.info(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        optimizer = self.optim_partial(optim_groups)
        if self.lr_scheduler_partial is None:
            return optimizer, None, None

        lr_scheduler = self.lr_scheduler_partial(
            optimizer=optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
            steps_per_epoch=num_iterations_per_epoch,
        )
        if isinstance(self.moving_average_decay, tuple):
            self.momentum_scheduler = (
                self.moving_average_decay[0]
                + i
                * (self.moving_average_decay[1] - self.moving_average_decay[0])
                / (num_epochs * num_iterations_per_epoch)
                for i in range(int(num_epochs * num_iterations_per_epoch) + 1)
            )
        self.current_teacher_temp = self.teacher_temp
        if isinstance(self.teacher_temp, tuple):
            self.teacher_temp_scheduler = self.teacher_temp_schedule(
                num_epochs, num_iterations_per_epoch
            )

            self.current_teacher_temp = self.teacher_temp[0]

        if self.wd_scheduler_partial is None:
            return (
                optimizer,
                {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "monitor": None,
                },
                None,
            )

        wd_scheduler = self.wd_scheduler_partial(
            optimizer,
            T_max=int(num_epochs * num_iterations_per_epoch),
        )
        return (
            optimizer,
            {"scheduler": lr_scheduler, "interval": "step", "monitor": None},
            {"wd_scheduler": wd_scheduler, "interval": "step", "frequency": 1},
        )

    def teacher_temp_schedule(self, num_epochs, num_iterations_per_epoch):
        assert isinstance(
            self.teacher_temp, tuple
        ), "Teacher temp must be a tuple if this function is called"
        for i in range(int(num_epochs * num_iterations_per_epoch) + 1):
            teacher_temp = None
            if i > (self.teacher_warmup_epochs * num_iterations_per_epoch):
                teacher_temp = self.teacher_temp[1]
            else:
                teacher_temp = self.teacher_temp[0] + i * (
                    self.teacher_temp[1] - self.teacher_temp[0]
                ) / (self.teacher_warmup_epochs * num_iterations_per_epoch)
            yield teacher_temp















import torch
import torch.nn as nn
from functools import partial
import math
from typing import Dict, Any, List, Optional, Tuple, Union

from models.VTT import VTT, init_weights_vit_timm
from tactile_ssl.model.layers.dino_head import DINOHead
from models.vtdino import VTDINO
from omegaconf import ListConfig
import copy
import torch.utils.data as data

# 如果tactile_ssl包没有导入，可能需要添加path
import sys
sys.path.append("/home/leonhard/workshop/presentation/M3L_2")

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    batch_size = 4
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




    # 创建VTT实例作为encoder
    encoder = VTT(
        image_size=(64, 64),
        tactile_size=(32, 32),
        image_patch_size=8,
        tactile_patch_size=4,
        dim=256,
        depth=4,
        heads=8,
        mlp_dim=512,
        num_tactiles=2,
        num_register_tokens=1,
        pos_embed_fn="sinusoidal"
    ).to(device)
    
    # 创建DINOHead实例
    dino_head_partial = partial(DINOHead, 
                       out_dim=65536,  # 通常DINO使用的输出维度
                       use_bn=False, 
                       nlayers=3, 
                       hidden_dim=2048, 
                       bottleneck_dim=256)
    
    # 创建优化器配置
    optim_cfg = partial(torch.optim.AdamW, lr=5e-4, weight_decay=0.05)
    
    # 创建学习率调度器配置（可选）
    lr_scheduler_cfg = partial(
        torch.optim.lr_scheduler.CosineAnnealingLR,
        eta_min=1e-6
    )
    
    # 创建VTDINO实例
    vtdino = VTDINO(
        encoder=encoder,
        dino_head=dino_head_partial,
        optim_cfg=optim_cfg,
        lr_scheduler_cfg=lr_scheduler_cfg,
        wd_scheduler_cfg=None,
        local_mask_scale=(0.2, 0.4),
        global_mask_scale=(0.2, 0.8),
        num_global_masks=1, 
        num_local_masks=2,
        min_keep_num_sensors=4,
        allow_mask_overlap=False,
        moving_average_decay=0.99,
        teacher_temp=[0.04, 0.07],
        teacher_warmup_epochs=10,
        use_momentum=True,
    ).to(device)
    
    # 设置当前教师温度，因为测试时没有scheduler
    vtdino.current_teacher_temp = 0.04
    
    # 创建一个模拟的batch数据
    x = {}
    x[1] = torch.randn(batch_size, 3, image_size[0], image_size[1]).to(device)  # 图像
    x[2] = torch.randn(batch_size, 3, tactile_size[0], tactile_size[1]).to(device)  # 触觉1
    x[3] = torch.randn(batch_size, 3, tactile_size[0], tactile_size[1]).to(device)  # 触觉2
    
    # 将输入数据也放入字典格式，便于兼容其他调用方式
    batch = {
        'image': x[1],
        'tactile1': x[2],
        'tactile2': x[3],
    }
    
    # 打印关键参数，确认初始化正确
    print(f"Encoder embedding dimension: {encoder.embed_dim}")
    print(f"Number of register tokens: {encoder.num_register_tokens}")
    print(f"DINO head output dimension: {vtdino.student_encoder_dict['dino_head'].last_layer.out_features}")
    
    # 调用training_step方法
    print("\n===== 测试VTDINO training_step =====")
    try:
        # 设置随机种子确保可重复性
        torch.manual_seed(42)
        # 初始化step以便sample_masks可以正确运行
        vtdino.step = 0
        
        output = vtdino.training_step(batch, batch_idx=0)
        
        # 输出结果
        print("\nTraining step completed successfully!")
        print("\nOutput metrics:")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item()}")
            else:
                print(f"  {key}: {value}")
        
        # 验证是否包含必要的输出键
        assert "ssl_loss" in output, "Output missing ssl_loss"
        assert "loss" in output, "Output missing loss"
        
        # 测试前向传播过程
        print("\n===== 测试VTDINO forward方法 =====")
        global_masks, local_masks = vtdino.sample_masks(batch["image"])
        loss = vtdino.forward(batch, global_masks, local_masks)
        print(f"Forward loss: {loss.item()}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()