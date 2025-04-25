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



class DINOExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """

    def __init__(self, 
                    observation_space: gym.Space, 
                    dino_model,    # 换成encoder
                    dim_embeddings, 
                    vision_only_control, 
                    frame_stack
                    ) -> None:
        super().__init__(observation_space, dim_embeddings)   # 3 =channel            *frame_stack
        self.flatten = nn.Flatten()
        self.dino_model = dino_model
        
        self.running_buffer = {}

        self.vision_only_control = vision_only_control

        self.frame_stack = frame_stack


        self.transformer = Transformer(dim = dim_embeddings, 
                                       depth = 1, 
                                       heads = 4, 
                                       dim_head= 64, 
                                       mlp_dim= dim_embeddings,
                                       dropout = 0)
        # self.vit_layer = VTT(   # 
        #     image_size = (64, 64), # not used
        #     tactile_size = (32, 32), # not used
        #     image_patch_size = 8, # not used
        #     tactile_patch_size = 4, # not used
        #     dim = dim_embeddings, 
        #     depth = 1,
        #     heads = 4,
        #     mlp_dim = dim_embeddings*2,
        #     num_tactiles = 2, # not used
        # )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print("************************")
        # print("用的是DINOExtractor的forward")
        # print("image shape: ", observations['image'].shape)
        # print("tactile shape: ", observations['tactile'].shape)
        if 'image' in observations and len(observations['image'].shape) == 5:
            observations['image'] = observations['image'].permute(0, 2, 3, 1, 4)
            observations['image'] = observations['image'].reshape((observations['image'].shape[0], observations['image'].shape[1], observations['image'].shape[2], -1))
        if 'tactile' in observations and len(observations['tactile'].shape) == 5:
            observations['tactile'] = observations['tactile'].reshape((observations['tactile'].shape[0], -1, observations['tactile'].shape[3], observations['tactile'].shape[4]))
        
        # Get embeddings
        vt_torch = vt_load(observations, frame_stack=self.frame_stack)
        if torch.cuda.is_available():
            for key in vt_torch:
                vt_torch[key] = vt_torch[key].to('cuda',dtype=torch.float32)
        #observations = self.dino_model.get_embeddings(vt_torch, eval=False, use_tactile=not self.vision_only_control)

        #observations = self.vit_layer.transformer(observations)
        #observations = torch.mean(observations, dim=1)
        #print(vt_torch['image'].shape)
        # vt_torch = torch.cat((vt_torch['image'],vt_torch['tactile1'],vt_torch['tactile2']),dim=-1)
        # vt_torch = self.dino_model(vt_torch)
        ob2 = None
        for i in range(self.frame_stack):
            mid = [None] * 3
            mid[0] = vt_torch['image'][:,i*3:(i+1)*3, : ,:]
            mid[0] =self.dino_model(mid[0])
            mid[1] = vt_torch['tactile1'][:,i*3:(i+1)*3, : ,:]
            mid[1] =self.dino_model(mid[1])            
            mid[2] = vt_torch['tactile2'][:,i*3:(i+1)*3, : ,:]
            mid[2] =self.dino_model(mid[2])
            mid[0] = mid[0].unsqueeze(1)
            mid[1] = mid[1].unsqueeze(1)
            mid[2] = mid[2].unsqueeze(1)
            #mid = torch.cat((mid[0] ,mid[1] ,mid[2]),dim = -2)
            if ob2 is None:
                ob2=(torch.cat((mid[0] ,mid[1] ,mid[2]),dim = -2))
            else:
                ob2 = torch.cat((ob2,mid[0] ,mid[1] ,mid[2]),dim = -2)

        ob2 = self.transformer(ob2)
        ob2 = torch.mean(ob2, dim=1)
        flattened = self.flatten(ob2)

        return flattened

class DINOPolicy(ActorCriticPolicy):              # 策略

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        dino_model = None,            #MAE类型实例
        dim_embeddings = 256,
        frame_stack = 1,
        vision_only_control = False,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
       

        features_extractor_class = DINOExtractor
        features_extractor_kwargs = {   'dino_model': dino_model, 
                                        'dim_embeddings': dim_embeddings, 
                                        'vision_only_control': vision_only_control, 
                                        'frame_stack': frame_stack
                                        }
        ortho_init = False

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            use_sde,
            log_std_init,
            full_std,
            use_expln,
            squash_output,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        print(obs['tactile'].shape)
        
        features = self.extract_features(obs)
        print(features.shape)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
















# import argparse
# import torch
# import numpy as np
# import gymnasium as gym

# from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
# from stable_baselines3.common.utils import set_random_seed
# from functools import partial

# import sys
# sys.path.append("/home/leonhard/workshop/presentation/M3L_2")

# import tactile_envs
# import envs
# from models.VTT import VTT
# from models.vtdino import VTDINO
# from models.pretrain_policy import DINOExtractor, DINOPolicy
# from tactile_ssl.model.layers.dino_head import DINOHead
# from utils.pretrain_utils import vt_load

# def main():
#     print("=== 测试DINOExtractor特征提取器 ===")

#     # 设置参数，参照train.py
#     seed = 42
#     set_random_seed(seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"使用设备: {device}")
    
#     # 环境配置
#     env_name = "tactile_envs/Insertion-v0"
#     n_envs = 1  # 测试时使用单个环境
#     state_type = "vision_and_touch"
#     frame_stack = 4
#     use_latch = True
#     camera_idx = 0
#     no_rotation = True
    
#     # 确定触觉传感器数量
#     num_tactiles = 2
    
#     # 创建环境
#     env_config = {"use_latch": use_latch}
#     objects = ["square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus"]
#     holders = ["holder1", "holder2", "holder3"]
    
#     env_list = [
#         envs.make_env(
#             env_name,
#             0,
#             seed,
#             state_type,
#             objects=objects,
#             holders=holders,
#             camera_idx=camera_idx,
#             frame_stack=frame_stack,
#             no_rotation=no_rotation,
#             **env_config,
#         )
#     ]
    
#     env = DummyVecEnv(env_list)  # 单个环境使用DummyVecEnv足够
#     env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
#     # 打印观测空间信息
#     print(f"观测空间: {env.observation_space}")
    
#     # DINO模型配置
#     dim_embedding = 256
    
#     # 创建VTT实例作为encoder
#     encoder = VTT(
#         image_size=(64, 64),
#         tactile_size=(32, 32),
#         image_patch_size=8,
#         tactile_patch_size=4,
#         dim=dim_embedding,
#         depth=4,
#         heads=8,
#         mlp_dim=dim_embedding * 2,
#         num_tactiles=num_tactiles,
#         image_channels=3*frame_stack,
#         tactile_channels=3*frame_stack,
#         frame_stack=frame_stack,
#         num_register_tokens=1,
#         pos_embed_fn="sinusoidal"
#     ).to(device)
    
#     # 创建DINOHead实例
#     dino_head_partial = partial(
#         DINOHead, 
#         out_dim=65536,  # 通常DINO使用的输出维度
#         use_bn=False, 
#         nlayers=3, 
#         hidden_dim=2048, 
#         bottleneck_dim=256
#     )
    
#     # 创建优化器配置
#     optim_cfg = partial(torch.optim.AdamW, lr=5e-4, weight_decay=0.05)
#     lr_scheduler_cfg = partial(torch.optim.lr_scheduler.CosineAnnealingLR, eta_min=1e-6)
    
#     # 创建VTDINO实例
#     dino_model = VTDINO(
#         encoder=encoder,
#         dino_head=dino_head_partial,
#         optim_cfg=optim_cfg,
#         lr_scheduler_cfg=lr_scheduler_cfg,
#         wd_scheduler_cfg=None,
#         local_mask_scale=(0.2, 0.8),
#         global_mask_scale=(0.2, 0.8),
#         num_global_masks=1, 
#         num_local_masks=4,
#         min_keep_num_sensors=4,
#         allow_mask_overlap=False,
#         moving_average_decay=0.99,
#         teacher_temp=[0.04, 0.07],
#         teacher_warmup_epochs=10,
#         use_momentum=True,
#     ).to(device)
    
#     # 初始化教师温度
#     dino_model.current_teacher_temp = 0.04
    
#     # 创建DINOExtractor特征提取器
#     vision_only_control = False  # 使用视觉和触觉
#     extractor = DINOExtractor(
#         observation_space=env.observation_space, 
#         dino_model=dino_model, 
#         dim_embeddings=dim_embedding, 
#         vision_only_control=vision_only_control, 
#         frame_stack=frame_stack
#     ).to(device)
    
#     print(f"提取器初始化完成，现在测试前向传播...")
    
#     # 获取一个观察样本
#     obs = env.reset()
#     print(f"观察样本形状:")
#     for key, value in obs.items():
#         print(key)
#         print(f"  {key}: {value.shape}")
    
#     # 将观测转换为PyTorch张量
#     obs_tensor = {}
#     for key, value in obs.items():
#         print(key)
#         obs_tensor[key] = torch.tensor(value).to(device)
    
#     # 打印关键提示信息
#     print("\n执行特征提取...")
    
#     # 前向传播
#     with torch.no_grad():  # 不需要梯度
#         features = extractor(obs_tensor)
    
#     # 打印特征形状和统计信息
#     print(f"提取的特征形状: {features.shape}")
#     print(f"特征均值: {features.mean().item()}")
#     print(f"特征标准差: {features.std().item()}")
#     print(f"特征最小值: {features.min().item()}")
#     print(f"特征最大值: {features.max().item()}")
    
#     # 测试是否可以与PPO策略集成
#     print("\n测试与DINOPolicy的集成...")
#     lr_schedule = lambda _: 1e-4
#     policy = DINOPolicy(
#         observation_space=env.observation_space,
#         action_space=env.action_space,
#         lr_schedule=lr_schedule,
#         net_arch=dict(pi=[256, 256], vf=[256, 256]),
#         dino_model=dino_model,
#         dim_embeddings=dim_embedding,
#         vision_only_control=vision_only_control,
#         frame_stack=frame_stack
#     ).to(device)


#     obs_tensor = {}
#     for key, value in obs.items():
#         print(key)
#         obs_tensor[key] = torch.tensor(value).to(device)

#     #print(obs_tensor['tactile1'].shape)
#     # 测试策略前向传播
#     with torch.no_grad():
#         actions, values, log_probs = policy(obs_tensor)
    
#     print(f"动作形状: {actions.shape}")
#     print(f"动作值: {actions.cpu().numpy()}")
#     print(f"值函数形状: {values.shape}")
#     print(f"值函数: {values.item()}")
    
#     print("\n=== 测试完成 ===")
#     env.close()

# if __name__ == "__main__":
#     main()




import argparse
import torch
import numpy as np
import gymnasium as gym
import os
import logging
from functools import partial

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.append("/home/leonhard/workshop/test/M3L")

import tactile_envs
import envs
from models.pretrain_policy import DINOExtractor, DINOPolicy
from utils.pretrain_utils import vt_load

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 设置参数
    parser = argparse.ArgumentParser("测试DINOExtractor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--state_type", type=str, default="vision_and_touch", 
                        choices=["vision", "touch", "vision_and_touch"])
    parser.add_argument("--frame_stack", type=int, default=1)
    parser.add_argument("--vision_only_control", type=bool, default=False)
    parser.add_argument("--dim_embedding", type=int, default=384)
    parser.add_argument("--env", type=str, default="tactile_envs/Insertion-v0")
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--no_rotation", type=bool, default=True)
    config = parser.parse_args()
    
    set_random_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 环境配置
    env_config = {"use_latch": True}
    objects = ["square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus"]
    holders = ["holder1", "holder2", "holder3"]
    
    # 确定触觉传感器数量
    num_tactiles = 0
    if config.state_type == "vision_and_touch" or config.state_type == "touch":
        num_tactiles = 2
    
    logger.info("创建环境...")
    env_list = [
        envs.make_env(
            config.env,
            i,
            config.seed,
            config.state_type,
            objects=objects,
            holders=holders,
            camera_idx=config.camera_idx,
            frame_stack=config.frame_stack,
            no_rotation=config.no_rotation,
            **env_config,
        )
        for i in range(config.n_envs)
    ]
    
    env = DummyVecEnv(env_list)  # 单个环境使用DummyVecEnv
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    logger.info(f"观测空间: {env.observation_space}")
    logger.info(f"动作空间: {env.action_space}")
    
    # 加载与trainDINO.py中相同的预训练DINOv2模型
    logger.info("加载预训练DINOv2模型...")
    try:
        dino_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        # 冻结DINO模型参数
        for param in dino_model.parameters():
            param.requires_grad = False
        logger.info("成功加载DINOv2模型并冻结参数")
    except Exception as e:
        logger.error(f"加载DINOv2模型时出错: {e}")
        return False
    
    # 将模型移动到正确设备
    dino_model = dino_model.to(device)
    
    # 创建DINOExtractor特征提取器
    logger.info("创建DINOExtractor特征提取器...")
    extractor = DINOExtractor(
        observation_space=env.observation_space, 
        dino_model=dino_model, 
        dim_embeddings=config.dim_embedding, 
        vision_only_control=config.vision_only_control, 
        frame_stack=config.frame_stack
    ).to(device)
    
    logger.info("测试DINOExtractor前向传播...")
    # 获取观察样本
    obs = env.reset()
    
    # 打印观察样本的形状
    logger.info("观察样本形状:")
    for key, value in obs.items():
        logger.info(f"  {key}: {value.shape}")
    
    # 将观测转换为PyTorch张量
    obs_tensor = {}
    for key, value in obs.items():
        obs_tensor[key] = torch.tensor(value).to(device)
    
    # 使用特征提取器
    try:
        with torch.no_grad():  # 不计算梯度
            features = extractor(obs_tensor)
            
        # 打印特征形状和统计信息
        logger.info(f"提取的特征形状: {features.shape}")
        logger.info(f"特征均值: {features.mean().item()}")
        logger.info(f"特征标准差: {features.std().item()}")
        logger.info(f"特征最小值: {features.min().item()}")
        logger.info(f"特征最大值: {features.max().item()}")
        logger.info("特征提取成功!")
    except Exception as e:
        logger.error(f"特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试与策略的集成
    logger.info("测试与DINOPolicy的集成...")
    lr_schedule = lambda _: 1e-4
#***************

    obs = env.reset()
    
    # 打印观察样本的形状
    logger.info("观察样本形状:")
    for key, value in obs.items():
        logger.info(f"  {key}: {value.shape}")
    
    # 将观测转换为PyTorch张量
    obs_tensor = {}
    for key, value in obs.items():
        obs_tensor[key] = torch.tensor(value).to(device)
#*****************


    try:
        policy = DINOPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lr_schedule,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            dino_model=dino_model,
            dim_embeddings=config.dim_embedding,
            vision_only_control=config.vision_only_control,
            frame_stack=config.frame_stack
        ).to(device)
        
        # 测试策略前向传播
        with torch.no_grad():
            actions, values, log_probs = policy(obs_tensor)
        
        logger.info(f"动作形状: {actions.shape}")
        logger.info(f"动作值: {actions.cpu().numpy()}")
        logger.info(f"值函数形状: {values.shape}")
        logger.info(f"值函数: {values.item() if values.numel() == 1 else values.mean().item()}")
        logger.info("策略集成测试成功!")
        
        # 执行一些动作，看看环境响应
        logger.info("测试环境交互...")
        for i in range(5):
            action, _ = policy.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            logger.info(f"Step {i}: Reward = {rewards[0]}")
            
        logger.info("环境交互测试成功!")
        return True
    except Exception as e:
        logger.error(f"策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        env.close()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ DINOExtractor测试通过!")
    else:
        print("\n❌ DINOExtractor测试失败!")

"python pretrain_policy.py  --frame_stack 4 --dim_embedding 384"