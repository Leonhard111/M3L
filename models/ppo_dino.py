import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

import sys

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from utils.pretrain_utils import vt_load

import copy


SelfPPO = TypeVar("SelfPPO", bound="PPO")


class PPO_DINO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        dino = None,
        dino_batch_size = 32,
        separate_optimizer = False,
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        
        
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.dino_batch_size = dino_batch_size
        self.separate_optimizer = separate_optimizer

        if _init_setup_model:
            self._setup_model()
            self.dino = dino
            
            # 配置DINO优化器 - 使用VTDINO的configure_optimizers
            # 估算每个epoch的迭代次数
            num_iterations_per_epoch = max(1, self.rollout_buffer.buffer_size // self.batch_size)
            # 估算总epoch数(允许用户训练更长时间)
            num_epochs = 1000  # 使用一个足够大的值，让VTDINO的scheduler能够正常工作
            
            # 调用VTDINO的configure_optimizers方法获取优化器和调度器
            if hasattr(self.dino, 'configure_optimizers'):
                try:
                    dino_optimizer, lr_scheduler_dict, wd_scheduler_dict = self.dino.configure_optimizers(
                        num_iterations_per_epoch=num_iterations_per_epoch,
                        num_epochs=num_epochs
                    )
                    self.dino_optimizer = dino_optimizer
                    
                    # 保存调度器以便后续使用
                    if lr_scheduler_dict is not None:
                        self.dino_lr_scheduler = lr_scheduler_dict.get('scheduler')
                    else:
                        self.dino_lr_scheduler = None
                        
                    if wd_scheduler_dict is not None:
                        self.dino_wd_scheduler = wd_scheduler_dict.get('wd_scheduler')
                    else:
                        self.dino_wd_scheduler = None
                        
                    if self.verbose > 0:
                        print(f"Successfully configured DINO optimizer and scheduler from VTDINO class")
                except Exception as e:
                    # 如果配置失败，回退到简单的优化器
                    print(f"Error configuring DINO optimizer from class: {e}. Using default optimizer.")
                    self.dino_optimizer = th.optim.Adam(self.dino.parameters(), lr=1e-4)
                    self.dino_lr_scheduler = None
                    self.dino_wd_scheduler = None
            else:
                # 如果DINO没有configure_optimizers方法，使用默认优化器
                self.dino_optimizer = th.optim.Adam(self.dino.parameters(), lr=1e-4)
                self.dino_lr_scheduler = None
                self.dino_wd_scheduler = None

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def load_dino(self, dino):
        """
        加载一个新的DINO模型并配置其优化器
        """
        self.dino = dino
        
        # 配置DINO优化器 - 使用VTDINO的configure_optimizers
        # 估算每个epoch的迭代次数
        num_iterations_per_epoch = max(1, self.rollout_buffer.buffer_size // self.batch_size)
        # 估算总epoch数(允许用户训练更长时间)
        num_epochs = 1000  # 使用一个足够大的值，让VTDINO的scheduler能够正常工作
        
        # 调用VTDINO的configure_optimizers方法获取优化器和调度器
        if hasattr(self.dino, 'configure_optimizers'):
            try:
                dino_optimizer, lr_scheduler_dict, wd_scheduler_dict = self.dino.configure_optimizers(
                    num_iterations_per_epoch=num_iterations_per_epoch,
                    num_epochs=num_epochs
                )
                self.dino_optimizer = dino_optimizer
                
                # 保存调度器以便后续使用
                if lr_scheduler_dict is not None:
                    self.dino_lr_scheduler = lr_scheduler_dict.get('scheduler')
                else:
                    self.dino_lr_scheduler = None
                    
                if wd_scheduler_dict is not None:
                    self.dino_wd_scheduler = wd_scheduler_dict.get('wd_scheduler')
                else:
                    self.dino_wd_scheduler = None
            except Exception as e:
                # 如果配置失败，回退到简单的优化器
                print(f"Error configuring DINO optimizer from class: {e}. Using default optimizer.")
                self.dino_optimizer = th.optim.Adam(self.dino.parameters(), lr=1e-4)
                self.dino_lr_scheduler = None
                self.dino_wd_scheduler = None
        else:
            # 如果DINO没有configure_optimizers方法，使用默认优化器
            self.dino_optimizer = th.optim.Adam(self.dino.parameters(), lr=1e-4)
            self.dino_lr_scheduler = None
            self.dino_wd_scheduler = None

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        
        # 更新DINO优化器的学习率（如果有调度器）
        if hasattr(self, 'dino_lr_scheduler') and self.dino_lr_scheduler is not None:
            self.dino_lr_scheduler.step()
        
        # 更新DINO权重衰减（如果有调度器）
        if hasattr(self, 'dino_wd_scheduler') and self.dino_wd_scheduler is not None:
            self.dino_wd_scheduler.step()
            
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                
                observations = rollout_data.observations
                frame_stack = 1
                if 'image' in observations and len(observations['image'].shape) == 5:
                    frame_stack = observations['image'].shape[1]
                    observations['image'] = observations['image'].permute(0, 2, 3, 1, 4) 
                    observations['image'] = observations['image'].reshape((observations['image'].shape[0], observations['image'].shape[1], observations['image'].shape[2], -1))
                if 'tactile' in observations and len(observations['tactile'].shape) == 5:
                    frame_stack = observations['tactile'].shape[1]
                    observations['tactile'] = observations['tactile'].reshape((observations['tactile'].shape[0], -1, observations['tactile'].shape[3], observations['tactile'].shape[4]))
                
                # 清零DINO的优化器梯度
                self.dino_optimizer.zero_grad()
                # 清零策略的优化器梯度
                self.policy.optimizer.zero_grad()
                
                # 加载数据
                x = vt_load(copy.deepcopy(observations), frame_stack=frame_stack)
                x = torch.act((x['image'],x['tactile1'],x['tactile2']),dim=-1)
                x
                # 执行DINO的训练步骤
                #dino_loss = self.dino.training_step(x, 0)
                # 反向传播DINO的损失
                #dino_loss['loss'].backward()
                
                # 如果设置为separate_optimizer为True，只有DINO优化器会在这里更新
                #if self.separate_optimizer:
                #    self.dino_optimizer.step()

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # 反向传播PPO的损失
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                # 更新策略优化器
                self.policy.optimizer.step()
            
            
            self._n_updates += 1
            if not continue_training:
                break
            
            # 在每个epoch结束时调用DINO的回调方法
            self.dino.on_train_epoch_end()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)

        self.logger.record("train/dino_loss", dino_loss['loss'].item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )











import argparse
import torch
import numpy as np
import gymnasium as gym
import os
import logging
from functools import partial
import copy
import einops
import traceback

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import sys
sys.path.append("/home/leonhard/workshop/presentation/M3L_2")

import tactile_envs
import envs
from models.VTT import VTT
from models.vtdino import VTDINO
from models.ppo_dino import PPO_DINO
from models.pretrain_policy import DINOPolicy, DINOExtractor
from tactile_ssl.model.layers.dino_head import DINOHead
from utils.pretrain_utils import vt_load

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 设置参数
    seed = 42
    set_random_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 环境配置
    env_name = "tactile_envs/Insertion-v0"
    n_envs = 2  # 测试用小环境数
    state_type = "vision_and_touch"
    frame_stack = 4
    use_latch = True
    camera_idx = 0
    no_rotation = True
    total_timesteps = 10000  # 只进行少量训练步骤进行测试
    
    # DINO和PPO参数 (参照dino_vit.yaml)
    dim_embedding = 256
    num_global_masks = 2
    num_local_masks = 8
    global_mask_scale = (0.48, 1.0)
    local_mask_scale = (0.1, 0.48)
    moving_average_decay = 0.998
    allow_mask_overlap = True
    teacher_temp = [0.04, 0.07]
    teacher_warmup_epochs = 10
    
    # 确定触觉传感器数量
    num_tactiles = 2
    
    # 创建环境
    logger.info("创建训练环境...")
    env_config = {"use_latch": use_latch}
    objects = ["square", "triangle", "horizontal", "vertical", "trapezoidal", "rhombus"]
    holders = ["holder1", "holder2", "holder3"]
    
    env_list = [
        envs.make_env(
            env_name,
            i,
            seed,
            state_type,
            objects=objects,
            holders=holders,
            camera_idx=camera_idx,
            frame_stack=frame_stack,
            no_rotation=no_rotation,
            **env_config,
        )
        for i in range(n_envs)
    ]
    
    env = SubprocVecEnv(env_list)
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    # 打印观测空间信息
    logger.info(f"观测空间: {env.observation_space}")
    logger.info(f"动作空间: {env.action_space}")
    
    # 创建VTT实例作为encoder
    logger.info("创建VTT编码器...")
    encoder = VTT(
        image_size=(64, 64),
        tactile_size=(32, 32),
        image_patch_size=8,
        tactile_patch_size=4,
        dim=dim_embedding,
        depth=4,
        heads=8,
        mlp_dim=dim_embedding * 2,
        num_tactiles=num_tactiles,
        image_channels=3*frame_stack,
        tactile_channels=3*frame_stack,
        frame_stack=frame_stack,
        num_register_tokens=1,
        pos_embed_fn="sinusoidal"
    ).to(device)
    
    # 创建DINOHead实例
    dino_head_partial = partial(
        DINOHead, 
        out_dim=65536,  # 通常DINO使用的输出维度
        use_bn=False, 
        nlayers=3, 
        hidden_dim=2048, 
        bottleneck_dim=256
    )
    
    # 创建优化器配置
    logger.info("配置优化器...")
    optim_cfg = partial(torch.optim.AdamW, lr=1e-4, weight_decay=0.05)
    lr_scheduler_cfg = partial(torch.optim.lr_scheduler.CosineAnnealingLR, eta_min=1e-6)
    
    # 创建VTDINO实例
    logger.info("创建VTDINO模型...")
    dino_model = VTDINO(
        encoder=encoder,
        dino_head=dino_head_partial,
        optim_cfg=optim_cfg,
        lr_scheduler_cfg=lr_scheduler_cfg,
        wd_scheduler_cfg=None,
        local_mask_scale=local_mask_scale,
        global_mask_scale=global_mask_scale,
        num_global_masks=num_global_masks, 
        num_local_masks=num_local_masks,
        min_keep_num_sensors=4,
        allow_mask_overlap=allow_mask_overlap,
        moving_average_decay=moving_average_decay,
        teacher_temp=teacher_temp,
        teacher_warmup_epochs=teacher_warmup_epochs,
        use_momentum=True,
    ).to(device)
    
    # 初始化教师温度
    dino_model.current_teacher_temp = teacher_temp[0]
    
    # 创建策略
    logger.info("创建DINOPolicy策略...")
    lr_schedule = lambda _: 1e-4
    
    # 创建并配置PPO_DINO训练器
    logger.info("创建PPO_DINO训练器...")
    model = PPO_DINO(
        policy=DINOPolicy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=128,  # 小批量用于快速测试
        batch_size=64,
        n_epochs=5,  # 减少训练轮次加快测试
        gamma=0.99,
        verbose=1,
        device=device,
        dino=dino_model,
        dino_batch_size=32,
        policy_kwargs={
            "dino_model": dino_model,
            "dim_embeddings": dim_embedding,
            "vision_only_control": False,
            "frame_stack": frame_stack,
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),  # 简化网络架构加快测试
        }
    )
    
    # 执行训练
    logger.info(f"开始训练 {total_timesteps} 步...")
    try:
        model.learn(total_timesteps=total_timesteps)
        logger.info("训练完成!")
        
        # 测试模型推理
        logger.info("测试模型推理...")
        obs = env.reset()
        for i in range(10):
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if dones.any():
                logger.info(f"步骤 {i}: 一些环境完成")
        
        logger.info("测试成功!")
        return True
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        traceback.print_exc()
        return False
    finally:
        env.close()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ VTDINO+DINOPolicy训练测试通过!")
    else:
        print("\n❌ VTDINO+DINOPolicy训练测试失败!")









