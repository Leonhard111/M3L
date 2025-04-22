import argparse
import torch
from functools import partial

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CallbackList

import tactile_envs
import envs
from utils.callbacks import create_callbacks
from models.ppo_dino import PPO_DINO
from models.pretrain_policy import DINOPolicy
from models.VTT import VTT
from models.vtdino import VTDINO
from tactile_ssl.model.layers.dino_head import DINOHead

def str2bool(v):
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    raise ValueError(f"boolean argument should be either True or False (got {v})")

def main():
    parser = argparse.ArgumentParser("VTDINO")

    # 基础参数
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=int(1e5))
    parser.add_argument("--eval_every", type=int, default=int(2e5))
    parser.add_argument("--total_timesteps", type=int, default=int(3e6))
    
    parser.add_argument("--wandb_dir", type=str, default="./wandb/")
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
 
    # 环境参数
    parser.add_argument(
        "--env",
        type=str,
        default="tactile_envs/Insertion-v0",
        choices=[
            "tactile_envs/Insertion-v0",
            "Door",
            "HandManipulateBlockRotateZFixed-v1",
            "HandManipulateEggRotateFixed-v1",
            "HandManipulatePenRotateFixed-v1"
        ],
    )
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument(
        "--state_type",
        type=str,
        default="vision_and_touch",
        choices=["vision", "touch", "vision_and_touch"]
    )
    parser.add_argument("--norm_reward", type=str2bool, default=True)
    parser.add_argument("--use_latch", type=str2bool, default=True)
    
    parser.add_argument("--camera_idx", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--no_rotation", type=str2bool, default=True)

    # DINO参数
    parser.add_argument("--representation", type=str2bool, default=True)
    parser.add_argument("--dim_embedding", type=int, default=256)
    parser.add_argument("--use_sincosmod_encodings", type=str2bool, default=True)
    parser.add_argument("--num_global_masks", type=int, default=2)
    parser.add_argument("--num_local_masks", type=int, default=8)
    parser.add_argument("--global_mask_scale_min", type=float, default=0.48)
    parser.add_argument("--global_mask_scale_max", type=float, default=1.0)
    parser.add_argument("--local_mask_scale_min", type=float, default=0.2)
    parser.add_argument("--local_mask_scale_max", type=float, default=0.48)
    parser.add_argument("--allow_mask_overlap", type=str2bool, default=True)
    parser.add_argument("--moving_average_decay", type=float, default=0.998)
    parser.add_argument("--teacher_temp_min", type=float, default=0.04)
    parser.add_argument("--teacher_temp_max", type=float, default=0.07)
    parser.add_argument("--teacher_warmup_epochs", type=int, default=10)
    
    parser.add_argument("--dino_batch_size", type=int, default=512)
    parser.add_argument("--train_dino_every", type=int, default=1)

    # PPO参数
    parser.add_argument("--rollout_length", type=int, default=32768)
    parser.add_argument("--ppo_epochs", type=int, default=10)
    parser.add_argument("--lr_ppo", type=float, default=1e-4)
    parser.add_argument("--vision_only_control", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=512)

    # PPO-DINO参数
    parser.add_argument("--separate_optimizer", type=str2bool, default=True)

    config = parser.parse_args()

    set_random_seed(config.seed)

    num_tactiles = 0
    if config.state_type == "vision_and_touch" or config.state_type == "touch":
        num_tactiles = 2
        if config.env == "HandManipulateBlockRotateZFixed-v1" or config.env == "HandManipulateEggRotateFixed-v1" or config.env == "HandManipulatePenRotateFixed-v1":
            num_tactiles = 1
            
    env_config = {
        "use_latch": config.use_latch,
    }
    
    objects = [
        "square",
        "triangle",
        "horizontal",
        "vertical",
        "trapezoidal",
        "rhombus",
    ]
    holders = ["holder1", "holder2", "holder3"]

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

    if config.n_envs < 100:
        env = SubprocVecEnv(env_list)
    else:
        env = DummyVecEnv(env_list)
    env = VecNormalize(env, norm_obs=False, norm_reward=config.norm_reward)
    # encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    # 创建VTT实例作为encoder
    encoder = VTT(
        image_size=(64, 64),
        tactile_size=(32, 32),
        image_patch_size=8,
        tactile_patch_size=4,
        dim=config.dim_embedding,
        depth=4,
        heads=8,
        mlp_dim=config.dim_embedding * 2,
        num_tactiles=num_tactiles,
        image_channels=3*config.frame_stack,
        tactile_channels=3*config.frame_stack,
        frame_stack=config.frame_stack,
        num_register_tokens=1,
        pos_embed_fn="sinusoidal" if config.use_sincosmod_encodings else "learned"
    )


    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')

    # 创建DINOHead实例 - 调整输出维度
    dino_head_partial = partial(
        DINOHead, 
        out_dim=8192,  # 修改为8192，更适合256维度的token  8192 = 256 * 32
        use_bn=False, 
        nlayers=3, 
        hidden_dim=2048, 
        bottleneck_dim=256
    )
    
    # 创建优化器配置
    optim_cfg = partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
    lr_scheduler_cfg = partial(torch.optim.lr_scheduler.CosineAnnealingLR, eta_min=1e-6)
    
    # 创建VTDINO实例
    dino = VTDINO(
        encoder=encoder,
        dino_head=dino_head_partial,
        optim_cfg=optim_cfg,
        lr_scheduler_cfg=lr_scheduler_cfg,
        wd_scheduler_cfg=None,
        local_mask_scale=(config.local_mask_scale_min, config.local_mask_scale_max),
        global_mask_scale=(config.global_mask_scale_min, config.global_mask_scale_max),
        num_global_masks=config.num_global_masks, 
        num_local_masks=config.num_local_masks,
        min_keep_num_sensors=4,
        allow_mask_overlap=config.allow_mask_overlap,
        moving_average_decay=config.moving_average_decay,
        teacher_temp=[config.teacher_temp_min, config.teacher_temp_max],
        teacher_warmup_epochs=config.teacher_warmup_epochs,
        use_momentum=True,
    )
    
    if torch.cuda.is_available():
        dino.cuda()
    dino.current_teacher_temp = config.teacher_temp_min
    dino.eval()

    if config.representation:
        policy = DINOPolicy
        policy_kwargs={
            "dino_model": dino,
            "dim_embeddings": config.dim_embedding,
            "vision_only_control": config.vision_only_control,
            "net_arch": dict(pi=[256, 256], vf=[256, 256]),
            "frame_stack": config.frame_stack,
        }

        # 创建PPO_DINO模型时，会自动调用dino.configure_optimizers
        model = PPO_DINO(
            policy,
            env,
            verbose=1,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            dino_batch_size=config.dino_batch_size,
            separate_optimizer=config.separate_optimizer,
            policy_kwargs=policy_kwargs,
            dino=dino,  # 直接传递dino实例，PPO_DINO会正确配置其优化器
        )
        
        callbacks = create_callbacks(
            config, model, num_tactiles, objects, holders
        )
        model.learn(
            total_timesteps=config.total_timesteps, callback=CallbackList(callbacks)
        )
    else:
        # 如果不使用表征学习，直接用PPO训练（仅用于基线比较）
        model = PPO_DINO(
            DINOPolicy,
            env,
            verbose=1,
            learning_rate=config.lr_ppo,
            tensorboard_log=config.wandb_dir+"dino_privileged_tensorboard/",
            batch_size=config.batch_size,
            n_steps=config.rollout_length // config.n_envs,
            n_epochs=config.ppo_epochs,
            policy_kwargs={
                "dino_model": dino,
                "dim_embeddings": config.dim_embedding,
                "vision_only_control": config.vision_only_control,
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "frame_stack": config.frame_stack,
            },
            dino=dino,
            dino_batch_size=config.dino_batch_size,
        )
        callbacks = create_callbacks(
            config, model, num_tactiles, objects, holders
        )
        model.learn(
            total_timesteps=config.total_timesteps, callback=CallbackList(callbacks)
        )


if __name__ == "__main__":
    main()
