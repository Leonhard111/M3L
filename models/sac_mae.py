from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.sac.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, SACPolicy

from models.sac_mae_policy import MAESACPolicy

from utils.pretrain_utils import vt_load
import copy

SelfSAC = TypeVar("SelfSAC", bound="SAC")


class SAC_MAE(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
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

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: SACPolicy
    actor: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        # new add
        mae = None,
        mae_batch_size = 32,
        separate_optimizer = False,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None


        "删了"
        # if _init_setup_model:
        #     self._setup_model()

        "new add"

        self.mae_batch_size = mae_batch_size
        self.separate_optimizer = separate_optimizer
        if _init_setup_model:
            self._setup_model()
            self.mae = mae
            self.mae_optimizer = th.optim.Adam(self.mae.parameters(), lr=1e-4)
    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def load_mae(self, mae):
        self.mae = mae
        self.mae_optimizer = th.optim.Adam(self.mae.parameters(), lr=1e-4)


    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            # replay_data
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]


            try:
                n_iter = replay_data.observations['image'].shape[0] // self.mae_batch_size
            except:
                n_iter = replay_data.observations['tactile'].shape[0] // self.mae_batch_size

            # self.policy.optimizer.zero_grad() # NEW

            observations = replay_data.observations
            # print("image shape: ", observations['image'].shape)
            # print("tactile shape: ", observations['tactile'].shape)
            frame_stack = 1
            if 'image' in observations and len(observations['image'].shape) == 5:
                frame_stack = observations['image'].shape[1]
                observations['image'] = observations['image'].permute(0, 2, 3, 1, 4) 
                observations['image'] = observations['image'].reshape((observations['image'].shape[0], observations['image'].shape[1], observations['image'].shape[2], -1))
            if 'tactile' in observations and len(observations['tactile'].shape) == 5:
                frame_stack = observations['tactile'].shape[1]
                observations['tactile'] = observations['tactile'].reshape((observations['tactile'].shape[0], -1, observations['tactile'].shape[3], observations['tactile'].shape[4]))
            
            # x = vt_load(copy.deepcopy(observations), frame_stack=frame_stack)
            # mae_loss = self.mae(x)
            # mae_loss.backward()
            
            if not self.separate_optimizer:
                self.policy.optimizer.zero_grad()
                n_iter = 1

            for i in range(n_iter):
                # Optimization step

                if self.separate_optimizer:
                    self.mae_optimizer.zero_grad()

                    batch_obs = copy.deepcopy({k: v[i*self.mae_batch_size:(i+1)*self.mae_batch_size] for k, v in observations.items()})
                    x = vt_load(batch_obs, frame_stack=frame_stack)

                    # Ensure all data is float32
                    for key in x:
                        if isinstance(x[key], th.Tensor):
                            x[key] = x[key].to(dtype=th.float32)

                    mae_loss = self.mae(x)
                    mae_loss.backward()

                    self.mae_optimizer.step()
                else:
                    x = vt_load(copy.deepcopy(observations), frame_stack=frame_stack)
                    mae_loss = self.mae(x)
                    mae_loss.backward()

            # if self.separate_optimizer:
            #     self.policy.optimizer.zero_grad()



            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

        "new add"
        self.logger.record("train/mae_loss", mae_loss.item())

        
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSAC:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["actor", "critic", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

def main():
    """
    测试SAC_MAE算法的示例函数，使用正确的policy和extractor
    优化版本：适用于低内存(16G)和低显存(6G)环境
    """
    import argparse
    import torch
    import os
    import gymnasium
    import tactile_envs
    import envs
    import gc
    from models.pretrain_models import VTT, VTMAE, MAEPolicy,MAEExtractor
    from models.sac_mae_policy import MAESACPolicy
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.callbacks import CheckpointCallback
    
    # 导入新的离线策略评估回调
    from utils.offpolicy_callbacks import OffPolicyEvalCallback, create_offpolicy_callbacks

    # 解析命令行参数
    parser = argparse.ArgumentParser("测试SAC_MAE算法")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--env", type=str, default="tactile_envs/Insertion-v0", help="环境名称")
    parser.add_argument("--state_type", type=str, default="vision_and_touch", 
                        choices=["vision", "touch", "vision_and_touch"], help="状态类型")
    parser.add_argument("--total_timesteps", type=int, default=20000, help="总训练步数")
    parser.add_argument("--frame_stack", type=int, default=2, help="帧堆叠数")
    parser.add_argument("--buffer_size", type=int, default=20000, help="经验回放缓冲区大小")
    parser.add_argument("--learning_starts", type=int, default=500, help="开始学习前的收集步数")
    parser.add_argument("--batch_size", type=int, default=128, help="批次大小")
    parser.add_argument("--mae_batch_size", type=int, default=16, help="MAE训练批次大小")
    parser.add_argument("--dim_embedding", type=int, default=128, help="嵌入维度")
    parser.add_argument("--save_path", type=str, default="./models/rubbish_test/saved", help="模型保存路径")
    parser.add_argument("--log_path", type=str, default="./rubbish_test/logs", help="日志保存路径")
    parser.add_argument("--n_envs", type=int, default=1, help="环境数量")
    parser.add_argument("--eval_freq", type=int, default=5000, help="评估频率")
    parser.add_argument("--save_freq", type=int, default=10000, help="模型保存频率")
    parser.add_argument("--memory_efficient", action="store_true", help="使用内存高效模式")
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 环境配置
    num_tactiles = 0
    if args.state_type == "vision_and_touch" or args.state_type == "touch":
        num_tactiles = 2
        if "HandManipulate" in args.env:
            num_tactiles = 1
    
    env_config = {"use_latch": True}
    objects = ["square", "triangle"]  # 减少对象数量以节省内存
    holders = ["holder1", "holder2"]  # 减少支架数量以节省内存
    
    # 创建单环境（不使用SubprocVecEnv以节省内存）
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
    env = VecNormalize(env, norm_obs=False, norm_reward=True)
    
    print(f"创建环境: {args.env}, 状态类型: {args.state_type}, 触觉传感器数量: {num_tactiles}")
    
    # 确定设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # 限制CUDA内存使用
        torch.cuda.set_per_process_memory_fraction(0.8)  # 使用最多80%的GPU内存
        print("使用CUDA，限制显存使用")
    else:
        device = torch.device("cpu")
        print("使用CPU")
    
    # 创建MAE模型（降低模型复杂度）
    print("创建MAE模型...")
    v = VTT(
        image_size=(64, 64),
        tactile_size=(32, 32),
        image_patch_size=8,
        tactile_patch_size=4,
        dim=args.dim_embedding,
        depth=3,  # 减少深度
        heads=3,  # 减少头数
        mlp_dim=args.dim_embedding,  # 减少MLP维度
        num_tactiles=num_tactiles,
        image_channels=3*args.frame_stack,
        tactile_channels=3*args.frame_stack,
        frame_stack=args.frame_stack,
    )

    mae = VTMAE(
        encoder=v,
        masking_ratio=0.95, 
        decoder_dim=args.dim_embedding // 2,  # 减少解码器维度
        decoder_depth=2,  # 减少解码器深度
        decoder_heads=3,  # 减少解码器头数
        num_tactiles=num_tactiles,
        early_conv_masking=True,
        use_sincosmod_encodings=True,
        frame_stack=args.frame_stack
    )
    
    mae = mae.to(device)
    mae.eval()
    
    # 初始化MAE训练
    print("初始化MAE训练...")
    mae.initialize_training({"lr": 1e-4, "batch_size": args.mae_batch_size})
    
    # 特征提取器和策略配置
    features_extractor_kwargs = {
        'mae_model': mae,
        'dim_embeddings': args.dim_embedding,
        'vision_only_control': False,
        'frame_stack': args.frame_stack
    }
    # MAEExtractor1 = MAEExtractor(env.action_space,**features_extractor_kwargs)
    # print(type(MAEExtractor1))
    # 创建SAC_MAE模型
    policy_kwargs = {
        "mae_model": mae,
        "features_extractor_class": MAEExtractor,
        "features_extractor_kwargs": features_extractor_kwargs,
        "net_arch": dict(pi=[128, 128], qf=[128, 128]),  # 减少网络层大小
    }
    
    print("创建SAC_MAE模型...")
    model = SAC_MAE(
        MAESACPolicy,
        env,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        tensorboard_log=f"{args.log_path}/tensorboard_sac_mae/",
        mae_batch_size=args.mae_batch_size,
        separate_optimizer=True,
        policy_kwargs=policy_kwargs,
        mae=mae,
        verbose=1,
    )
    print(model.policy.features_extractor.frame_stack)
    # 如果需要测试特征提取器和策略，使用更小的测试样本
    if args.memory_efficient:
        print("跳过特征提取器和策略测试以节省内存")
    else:
        print("\n======== 开始特征提取器和策略测试 ========")
        
        # 从环境获取真实观测样本
        print("获取环境观测样本...")
        obs = env.reset()
        
        # 检查并打印观测格式
        print(f"环境观测类型: {type(obs)}")
        print(f"环境观测结构: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
        
        if isinstance(obs, dict):
            for key, value in obs.items():
                print(f"观测 '{key}' 形状: {value.shape}")
        
        # 创建与环境观测格式相匹配的测试数据
        print("\n创建匹配环境观测格式的测试数据...")
        
        test_obs = {}
        if isinstance(obs, dict):
            for key, value in obs.items():
                if key == "image":
                    # 保持与环境完全相同的形状，并明确使用float32类型
                    test_obs["image"] = torch.rand(value.shape, device=device, dtype=torch.float32)
                    print(f"创建测试图像数据，形状: {test_obs['image'].shape}")
                elif key == "tactile" and num_tactiles > 0:
                    # 保持与环境完全相同的形状，并明确使用float32类型
                    test_obs["tactile"] = torch.rand(value.shape, device=device, dtype=torch.float32)
                    print(f"创建测试触觉数据，形状: {test_obs['tactile'].shape}")
                else:
                    # 处理其他类型的观测数据，保持float32类型
                    test_obs[key] = torch.tensor(value, device=device, dtype=torch.float32)
                    print(f"创建其他测试数据 '{key}'，形状: {test_obs[key].shape}")
        else:
            # 如果观测不是字典类型，使用默认形状，明确float32类型
            print("警告：环境观测不是字典类型，使用默认形状")
            test_obs = {"image": torch.rand((1, 64, 64, 3*args.frame_stack), device=device, dtype=torch.float32)}
            if num_tactiles > 0:
                test_obs["tactile"] = torch.rand((1, num_tactiles, 32, 32, 3*args.frame_stack), device=device, dtype=torch.float32)
        
        # 在使用前检查维度是否符合vt_load的要求
        print("\n检查测试数据维度是否符合vt_load要求...")
        if "image" in test_obs:
            if test_obs["image"].ndim == 5:  # (batch, frame_stack, h, w, c)
                # 此时需要转换形状
                print(f"转换图像数据形状: {test_obs['image'].shape} → ", end="")
                image = test_obs["image"]
                batch, frames, h, w, c = image.shape
                image = image.permute(0, 2, 3, 1, 4)
                image = image.reshape(batch, h, w, frames * c)
                test_obs["image"] = image
                print(f"{test_obs['image'].shape}")
        
        if "tactile" in test_obs:
            if test_obs["tactile"].ndim == 6:  # (batch, num_tactiles, frame_stack, h, w, c)
                print(f"转换触觉数据形状: {test_obs['tactile'].shape} → ", end="")
                tactile = test_obs["tactile"]
                batch, n_tactiles, frames, h, w, c = tactile.shape
                tactile = tactile.reshape(batch, n_tactiles, h, w, frames * c)
                test_obs["tactile"] = tactile
                print(f"{test_obs['tactile'].shape}")
        
        # 测试特征提取器
        print("\n测试MAEExtractor...")
        features_extractor = model.policy.features_extractor
        model.policy.set_training_mode(False)
        print(f"特征提取器类型: {type(features_extractor)}")
        
        with torch.no_grad():
            try:
                features = features_extractor(test_obs)
                print(f"特征提取成功 - 形状: {features.shape}")
            except Exception as e:
                print(f"特征提取失败: {e}")
                # 尝试使用vt_load直接处理
                print("\n尝试使用vt_load直接处理测试数据...")
                try:
                    # 确保所有输入都转换为float32
                    float32_obs = {}
                    for k, v in test_obs.items():
                        if isinstance(v, torch.Tensor):
                            float32_obs[k] = v.to(dtype=torch.float32)
                        else:
                            float32_obs[k] = v
                    
                    vt_data = vt_load(float32_obs, frame_stack=args.frame_stack)
                    print(f"vt_load处理成功，结果形状: {', '.join([f'{k}: {v.shape}' for k, v in vt_data.items()])}")
                    # 打印张量数据类型，帮助调试
                    for k, v in vt_data.items():
                        print(f"'{k}' 数据类型: {v.dtype}")
                except Exception as e2:
                    print(f"vt_load处理失败: {e2}")
        
        print("======== 特征提取器和策略测试完成 ========\n")
        
        # 释放测试数据占用的内存
        del test_obs
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    # 创建回调函数，使用新的离线策略评估回调
    class SimpleConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    # 简化配置，用于测试
    simple_config = SimpleConfig(
        use_latch=True,
        no_rotation=True,
        representation=True,
        frame_stack=args.frame_stack,
        state_type=args.state_type,
        env=args.env,
        camera_idx=0,
        save_freq=args.save_freq,
        n_envs=args.n_envs,
        eval_every=args.eval_freq,
        wandb_dir=args.log_path,
        wandb_id=None,
        wandb_entity=None,
        rollout_length=1  # 为兼容原有回调
    )
    
    print("创建离线策略回调...")
    try:
        # 尝试使用专用回调
        from utils.offpolicy_callbacks import create_offpolicy_callbacks
        callbacks = create_offpolicy_callbacks(
            simple_config, model, num_tactiles, objects, holders
        )
        print("成功创建离线策略专用回调")
    except ImportError as e:
        print(f"无法导入离线策略回调，使用默认回调: {e}")
        # 回退到普通回调
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=f"{args.save_path}/checkpoints/",
            name_prefix="sac_mae_model",
            save_vecnormalize=True
        )
        
        eval_callback = OffPolicyEvalCallback(
            args.env,
            args.state_type,
            no_tactile=(num_tactiles == 0),
            representation=True,
            eval_freq=args.eval_freq,
            config=simple_config,
            objects=objects,
            holders=holders,
            camera_idx=0,
            frame_stack=args.frame_stack,
        )
        
        callbacks = [checkpoint_callback, eval_callback]
    
    # 训练模型
    print(f"开始训练SAC_MAE模型，总步数: {args.total_timesteps}...")

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    print("训练完成")
    
    # 保存最终模型
    final_model_path = f"{args.save_path}/final_model"
    model.save(final_model_path)
    print(f"最终模型已保存到 {final_model_path}")
    
    # 保存环境正则化统计信息
    env_stats_path = f"{args.save_path}/vec_normalize_stats.pkl"
    env.save(env_stats_path)
    print(f"环境统计信息已保存到 {env_stats_path}")
    # except Exception as e:
        # print(f"训练过程中发生错误: {e}")
    # finally:
    #     # 确保资源被释放
    #     print("清理资源...")
        # env.close()
        # del model
        # del mae
        # del env
        # torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        # print("测试完成")

if __name__ == "__main__":
    main()
