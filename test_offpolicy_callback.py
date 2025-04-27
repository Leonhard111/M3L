"""
测试离线策略评估回调
此脚本验证新开发的针对OffPolicyAlgorithm的回调是否正常工作
"""
import argparse
import os
import torch
import numpy as np
import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed
import tactile_envs
import envs

from models.pretrain_models import VTT, VTMAE, MAEExtractor
from models.sac_mae import SAC_MAE
from models.sac_mae_policy import MAESACPolicy

# 导入新的离线策略回调
from utils.offpolicy_callbacks import OffPolicyEvalCallback, create_offpolicy_callbacks

def parse_args():
    parser = argparse.ArgumentParser("测试离线策略评估回调")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--env", type=str, default="tactile_envs/Insertion-v0", help="环境名称")
    parser.add_argument("--state_type", type=str, default="vision_and_touch", 
                        choices=["vision", "touch", "vision_and_touch"], help="状态类型")
    parser.add_argument("--total_timesteps", type=int, default=5000, help="总训练步数")
    parser.add_argument("--frame_stack", type=int, default=4, help="帧堆叠数")
    parser.add_argument("--buffer_size", type=int, default=10000, help="经验回放缓冲区大小")
    parser.add_argument("--learning_starts", type=int, default=500, help="开始学习前的收集步数")
    parser.add_argument("--eval_freq", type=int, default=1000, help="评估频率(时间步)")
    parser.add_argument("--save_path", type=str, default="./test_results", help="结果保存路径")
    parser.add_argument("--dim_embedding", type=int, default=128, help="嵌入维度")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设置环境
    num_tactiles = 0
    if args.state_type == "vision_and_touch" or args.state_type == "touch":
        num_tactiles = 2
    
    env_config = {"use_latch": True}
    objects = ["square", "triangle"]  # 简化测试
    holders = ["holder1", "holder2"]
    
    env_fn = lambda: envs.make_env(
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
    )()
    
    env = DummyVecEnv([env_fn])
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    # 创建MAE模型
    print("创建MAE模型...")
    v = VTT(
        image_size=(64, 64),
        tactile_size=(32, 32),
        image_patch_size=8,
        tactile_patch_size=4,
        dim=args.dim_embedding,
        depth=3,
        heads=3,
        mlp_dim=args.dim_embedding,
        num_tactiles=num_tactiles,
        image_channels=3*args.frame_stack,
        tactile_channels=3*args.frame_stack,
        frame_stack=args.frame_stack,
    )

    mae = VTMAE(
        encoder=v,
        masking_ratio=0.95,
        decoder_dim=args.dim_embedding // 2,
        decoder_depth=2,
        decoder_heads=3,
        num_tactiles=num_tactiles,
        early_conv_masking=True,
        use_sincosmod_encodings=True,
        frame_stack=args.frame_stack
    )
    
    if torch.cuda.is_available():
        mae = mae.cuda()
    mae.eval()
    
    # 初始化MAE训练
    mae.initialize_training({"lr": 1e-4, "batch_size": 64})
    
    # 特征提取器和策略配置
    features_extractor_kwargs = {
        'mae_model': mae,
        'dim_embeddings': args.dim_embedding,
        'vision_only_control': False,
        'frame_stack': args.frame_stack
    }
    
    policy_kwargs = {
        "features_extractor_class": MAEExtractor,
        "features_extractor_kwargs": features_extractor_kwargs,
        "net_arch": dict(pi=[64, 64], qf=[64, 64]),  # 小型网络用于快速测试
    }
    
    # 创建SAC_MAE模型
    print("创建SAC_MAE模型...")
    model = SAC_MAE(
        MAESACPolicy,
        env,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,  # 设置一个较小值以便快速启动学习
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        tensorboard_log=f"{args.save_path}/tensorboard/",
        mae_batch_size=32,
        separate_optimizer=False,
        policy_kwargs=policy_kwargs,
        mae=mae,
        verbose=1,
    )
    
    # 创建配置对象，供回调使用
    class SimpleConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    config = SimpleConfig(
        use_latch=True,
        no_rotation=True,
        representation=True,
        frame_stack=args.frame_stack,
        state_type=args.state_type,
        env=args.env,
        camera_idx=0,
        save_freq=args.eval_freq,
        n_envs=1,
        eval_every=args.eval_freq,
        wandb_dir=args.save_path,
        wandb_id=None,
        wandb_entity=None,
        rollout_length=1
    )
    
    # 创建回调
    print("创建离线策略回调...")
    
    # 验证方法1：使用单独的评估回调
    eval_callback = OffPolicyEvalCallback(
        args.env,
        args.state_type,
        no_tactile=(num_tactiles == 0),
        representation=True,
        eval_freq=args.eval_freq,
        config=config,
        objects=objects,
        holders=holders,
        camera_idx=0,
        frame_stack=args.frame_stack,
    )
    
    # 验证方法2：使用回调创建函数
    print("测试create_offpolicy_callbacks函数...")
    callbacks_from_create = create_offpolicy_callbacks(
        config, model, num_tactiles, objects, holders
    )
    
    # 使用单独的评估回调进行测试训练
    print(f"开始测试训练，使用离线策略评估回调，总步数: {args.total_timesteps}...")
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    
    # 保存结果
    model.save(f"{args.save_path}/test_model")
    print(f"测试完成，模型已保存到 {args.save_path}/test_model")
    
    # 清理资源
    env.close()
    
    print("测试脚本执行完毕")
    print("检查以下几点来确认回调是否正常工作:")
    print("1. 是否仅在learning_starts后开始评估（应该是在步数达到500后）")
    print("2. 评估是否按照eval_freq的频率进行（每1000步）")
    print("3. 查看tensorboard日志中是否记录了评估回报")

if __name__ == "__main__":
    main()
