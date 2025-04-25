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
from pretrain_models import MAEExtractor
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


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
        features_extractor_class: Type[BaseFeaturesExtractor] = MAEExtractor,
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
        # 准备特征提取器参数
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
            
        # 确保使用正确的特征提取器
        features_extractor_class = MAEExtractor
        features_extractor_kwargs = {   'mae_model': mae_model, 
                                        'dim_embeddings': dim_embeddings, 
                                        'vision_only_control': vision_only_control, 
                                        'frame_stack': frame_stack
                                     }
            
        # 处理lr_schedule参数 - 如果是浮点数，转换为常量函数
        if isinstance(lr_schedule, (float, int)):
            lr_value = float(lr_schedule)
            lr_schedule = lambda _: lr_value
            
        # 初始化SACPolicy的基础参数 - 只使用SACPolicy接受的参数
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            use_sde,
            log_std_init,
            use_expln,
            clip_mean,
            features_extractor_class,  # 强制使用MAESACExtractor作为特征提取器
            features_extractor_kwargs,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
            n_critics,
            share_features_extractor
        )
        if self.features_extractor is None:
            self.features_extractor = MAEExtractor(
            observation_space,
            **features_extractor_kwargs
        )


    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        # print("*******************")
        # features = self.extract_features(obs,self.features_extractor)
        # print("*******************")
        # print(features.shape)
        return self._predict(obs, deterministic=deterministic)


def main():
    """
    测试MAESACPolicy的前向功能
    """
    import argparse
    import torch
    import os
    import gymnasium as gym
    import numpy as np
    import tactile_envs
    import envs
    from pretrain_models import VTT, VTMAE, MAEPolicy, MAEExtractor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.utils import set_random_seed
    from utils.pretrain_utils import vt_load

    # 解析命令行参数
    parser = argparse.ArgumentParser("测试MAESACPolicy的前向功能")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--env", type=str, default="tactile_envs/Insertion-v0", help="环境名称")
    parser.add_argument("--state_type", type=str, default="vision_and_touch", 
                        choices=["vision", "touch", "vision_and_touch"], help="状态类型")
    parser.add_argument("--frame_stack", type=int, default=2, help="帧堆叠数")
    parser.add_argument("--dim_embedding", type=int, default=128, help="嵌入维度")
    args = parser.parse_args()
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 环境配置
    num_tactiles = 0
    if args.state_type == "vision_and_touch" or args.state_type == "touch":
        num_tactiles = 2
        if "HandManipulate" in args.env:
            num_tactiles = 1
    
    env_config = {"use_latch": True}
    objects = ["square"]  # 只使用一个对象节省内存
    holders = ["holder1"]  # 只使用一个支架节省内存
    
    # 创建环境
    print(f"创建环境: {args.env}, 状态类型: {args.state_type}")
    env_fn = envs.make_env(
        args.env,
        0,
        args.seed,
        args.state_type,
        objects=objects,
        holders=holders,
        camera_idx=0,
        frame_stack=args.frame_stack,
        no_rotation=True,
        **env_config
    )
    env = DummyVecEnv([env_fn])
    
    # 确定设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用CUDA设备")
    else:
        device = torch.device("cpu")
        print("使用CPU设备")
    
    # 创建MAE模型
    print("创建MAE模型...")
    v = VTT(
        image_size=(64, 64),
        tactile_size=(32, 32),
        image_patch_size=8,
        tactile_patch_size=4,
        dim=args.dim_embedding,
        depth=2,  # 减少深度以加快测试
        heads=3,
        mlp_dim=args.dim_embedding,
        num_tactiles=num_tactiles,
        image_channels=3*args.frame_stack,
        tactile_channels=3*args.frame_stack,
        frame_stack=args.frame_stack,
    )

    mae = VTMAE(
        encoder=v,
        masking_ratio=0.75, 
        decoder_dim=args.dim_embedding // 2,
        decoder_depth=1,  # 减少深度以加快测试
        decoder_heads=2,
        num_tactiles=num_tactiles,
        early_conv_masking=True,
        use_sincosmod_encodings=True,
        frame_stack=args.frame_stack
    )
    
    mae = mae.to(device)
    mae.eval()
    
    # 特征提取器配置
    features_extractor_kwargs = {
        'mae_model': mae,
        'dim_embeddings': args.dim_embedding,
        'vision_only_control': args.state_type == "vision",
        'frame_stack': args.frame_stack
    }
    
    # 创建MAESACPolicy
    print("创建MAESACPolicy...")
    observation_space = env.observation_space
    action_space = env.action_space
    
    policy = MAESACPolicy(
        observation_space=observation_space,
        action_space=action_space,
        lr_schedule=3e-4,
        net_arch=dict(pi=[64, 64], qf=[64, 64]),  # 使用小型网络以加快测试
        mae_model=mae,
        dim_embeddings=args.dim_embedding,
        frame_stack=args.frame_stack,
        vision_only_control=args.state_type == "vision",
    )
    policy = policy.to(device)
    policy.eval()
    
    # 获取一个观测样本
    print("获取环境观测样本...")
    obs = env.reset()
    
    # 打印观测形状
    print("观测样本结构:")
    if isinstance(obs, dict):
        for key, value in obs.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    # 将观测转换为PyTorch张量并移至指定设备
    device_obs = {}
    for key, value in obs.items():
        device_obs[key] = torch.tensor(value, device=device)
    
    # 测试前向函数
    print("\n开始测试MAESACPolicy的前向功能...")
    
    try:
        with torch.no_grad():
            # 不使用确定性策略
            action_non_det = policy.forward(device_obs, deterministic=False)
            print(f"非确定性动作: shape={action_non_det.shape}, values={action_non_det.cpu().numpy()}")
            
            # 使用确定性策略
            action_det = policy.forward(device_obs, deterministic=True)
            print(f"确定性动作: shape={action_det.shape}, values={action_det.cpu().numpy()}")
            
        print("\n测试extract_features功能...")
        with torch.no_grad():
            features = policy.extract_features(device_obs,policy.features_extractor)
            print(f"提取的特征: shape={features.shape}")
            
        print("\n测试action_log_prob功能...")
        with torch.no_grad():
            actions_pi, log_prob = policy.actor.action_log_prob(device_obs)
            print(f"动作: shape={actions_pi.shape}, 对数概率: shape={log_prob.shape}")
            
        print("\n测试MAE嵌入提取功能...")
        with torch.no_grad():
            # 准备适用于MAE的输入格式
            if 'image' in device_obs and len(device_obs['image'].shape) == 5:
                image = device_obs['image']
                device_obs['image'] = image.permute(0, 2, 3, 1, 4).reshape(
                    (image.shape[0], image.shape[2], image.shape[3], -1))
                
            if 'tactile' in device_obs and len(device_obs['tactile'].shape) == 5:
                tactile = device_obs['tactile']
                device_obs['tactile'] = tactile.reshape(
                    (tactile.shape[0], -1, tactile.shape[3], tactile.shape[4]))
                
            vt_torch = vt_load(device_obs, frame_stack=args.frame_stack)
            for key in vt_torch:
                #vt_torch[key] = vt_torch[key].to(device)
                vt_torch[key] = torch.tensor(vt_torch[key],device=device, dtype=torch.float32) 
                
            embeddings = mae.get_embeddings(
                vt_torch, eval=True, use_tactile=args.state_type != "vision")
            print(f"MAE嵌入: shape={embeddings.shape}")
            
        print("\n所有测试均已成功完成!")
        
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n清理资源...")
    env.close()
    
if __name__ == "__main__":
    main()