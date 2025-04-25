import torch as th
import torch.nn as nn
import gymnasium as gym
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.sac.policies import Actor, SACPolicy

from utils.pretrain_utils import vt_load


class MAESACExtractor(BaseFeaturesExtractor):
    """
    特征提取器，用于从观察中提取MAE编码的特征，专为SAC算法设计。
    
    :param observation_space: 观察空间
    :param mae_model: MAE模型实例
    :param dim_embeddings: 嵌入维度
    :param vision_only_control: 是否仅使用视觉控制
    :param frame_stack: 堆叠的帧数量
    """

    def __init__(
        self, 
        observation_space: gym.Space, 
        mae_model=None,
        dim_embeddings=256,
        vision_only_control=False,
        frame_stack=1
    ) -> None:
        super().__init__(observation_space, dim_embeddings)
        self.flatten = nn.Flatten()
        self.mae_model = mae_model
        self.vision_only_control = vision_only_control
        self.frame_stack = frame_stack

        # 简化的Transformer层，用于进一步处理MAE提取的特征
        self.vit_layer = nn.Sequential(
            nn.Linear(dim_embeddings, dim_embeddings),
            nn.LayerNorm(dim_embeddings),
            nn.ReLU()
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        # 处理图像和触觉数据的形状以适应MAE输入要求
        if 'image' in observations and len(observations['image'].shape) == 5:
            observations['image'] = observations['image'].permute(0, 2, 3, 1, 4)
            observations['image'] = observations['image'].reshape(
                (observations['image'].shape[0], observations['image'].shape[1], 
                 observations['image'].shape[2], -1))
        
        if 'tactile' in observations and len(observations['tactile'].shape) == 5:
            observations['tactile'] = observations['tactile'].reshape(
                (observations['tactile'].shape[0], -1, 
                 observations['tactile'].shape[3], observations['tactile'].shape[4]))
        
        # 转换数据格式并加载到MAE模型中
        vt_torch = vt_load(observations, frame_stack=self.frame_stack)
        if th.cuda.is_available():
            for key in vt_torch:
                vt_torch[key] = vt_torch[key].to('cuda')
        
        # 提取特征向量
        embeddings = self.mae_model.get_embeddings(
            vt_torch, eval=False, use_tactile=not self.vision_only_control)
        
        # 平均池化和特征处理
        embeddings = th.mean(embeddings, dim=1)  # 平均池化
        features = self.vit_layer(embeddings)
        flattened = self.flatten(features)
        
        return flattened


class MAESACPolicy(SACPolicy):
    """
    基于MAE的SAC策略，专为SAC_MAE算法设计。
    
    :param observation_space: 观察空间
    :param action_space: 动作空间
    :param lr_schedule: 学习率调度
    :param net_arch: 网络架构，actor和critic的隐藏层数量和大小
    :param activation_fn: 激活函数
    :param use_sde: 是否使用状态依赖探索
    :param log_std_init: 初始化log_std的值
    :param sde_net_arch: 状态依赖探索的网络架构
    :param use_expln: 是否使用expanded tanh函数
    :param clip_mean: 是否裁剪动作均值
    :param features_extractor_class: 特征提取器类
    :param features_extractor_kwargs: 特征提取器的关键字参数
    :param normalize_images: 是否归一化图像
    :param optimizer_class: 优化器类
    :param optimizer_kwargs: 优化器的关键字参数
    :param n_critics: critic网络的数量
    :param share_features_extractor: 是否在actor和critic之间共享特征提取器
    :param mae_model: MAE模型实例
    :param dim_embeddings: 嵌入维度
    :param frame_stack: 堆叠的帧数量
    :param vision_only_control: 是否仅使用视觉控制
    """
    
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = MAESACExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
        mae_model=None,
        dim_embeddings=256,
        frame_stack=1,
        vision_only_control=False,
    ):
        # 确保使用MAESACExtractor作为特征提取器
        if features_extractor_class != MAESACExtractor:
            features_extractor_class = MAESACExtractor
            
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'mae_model': mae_model, 
                'dim_embeddings': dim_embeddings, 
                'vision_only_control': vision_only_control, 
                'frame_stack': frame_stack
            }
        else:
            features_extractor_kwargs.update({
                'mae_model': mae_model, 
                'dim_embeddings': dim_embeddings, 
                'vision_only_control': vision_only_control, 
                'frame_stack': frame_stack
            })
            
        # 初始化SACPolicy的基础参数
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            sde_net_arch,
            use_expln,
            clip_mean,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )
