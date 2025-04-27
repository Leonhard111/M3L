import wandb
import numpy as np
import os
import logging

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.vec_env import DummyVecEnv

import envs
from utils.wandb_logger import WandbLogger
from utils.pretrain_utils import log_videos

# 设置基本日志记录
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("offpolicy_callbacks")

class OffPolicyTensorboardCallback(BaseCallback):
    """
    离线策略算法的自定义Tensorboard回调
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if hasattr(self.model, "ep_success_buffer"):
            self.logger.record("rollout/avg_success", np.mean(self.model.ep_success_buffer))
        return True


class OffPolicyEvalCallback(BaseCallback):
    """
    专门为离线策略算法(如SAC)设计的评估回调
    确保只在模型开始学习后才进行评估
    """
    def __init__(
        self,
        env,
        state_type,
        no_tactile=False,
        representation=True,
        eval_freq=10000,  # 基于时间步而非rollout
        verbose=0,
        config=None,
        objects=["square"],
        holders=["holder2"],
        camera_idx=0,
        frame_stack=1,
    ):
        super(OffPolicyEvalCallback, self).__init__(verbose)
        self.n_samples = 4
        self.eval_seed = 100
        self.no_tactile = no_tactile
        self.representation = representation
        self.eval_freq = eval_freq
        self.last_eval_step = 0
        
        # 使用外部logger而不是self.logger (在_on_training_start中设置)
        logger.info(f"离线策略评估回调初始化，评估频率: {eval_freq}步")

        env_config = {"use_latch": config.use_latch}

        self.test_env = DummyVecEnv(
            [
                envs.make_env(
                    env,
                    0,
                    self.eval_seed,
                    state_type=state_type,
                    objects=objects,
                    holders=holders,
                    camera_idx=camera_idx,
                    frame_stack=frame_stack,
                    no_rotation=config.no_rotation,
                    **env_config
                )
            ]
        )
        
    def _on_training_start(self) -> None:
        """当训练开始时调用，此时model已经可用"""
        logger.info(f"离线策略评估回调：训练开始，模型学习起始步数为 {self.model.learning_starts}")

    def _on_step(self) -> bool:
        """
        在每个时间步检查是否应该进行评估
        仅在模型已经开始学习且达到评估频率时进行
        """
        # 检查是否已经开始学习
        if not hasattr(self.model, "learning_starts"):
            return True
            
        is_training = self.model.num_timesteps >= self.model.learning_starts
        is_eval_time = (self.model.num_timesteps - self.last_eval_step) >= self.eval_freq
        
        if is_training and is_eval_time:
            logger.info(f"时间步 {self.model.num_timesteps}: 开始离线策略评估")
            self.last_eval_step = self.model.num_timesteps
            self._run_evaluation()
            
        return True
    
    def _run_evaluation(self):
        """执行评估并记录结果"""
        ret, obses, rewards_per_step = self.eval_model()
        frame_stack = 1
        try:
            frame_stack = obses[0]["image"].shape[1]
        except:
            # 尝试从不同维度获取frame_stack信息
            try:
                if len(obses[0]["image"].shape) > 3:
                    frame_stack = obses[0]["image"].shape[-1] // 3
            except:
                logger.warning("无法确定帧堆叠数量，使用默认值1")
            
        self.logger.record("eval/return", ret)
        
        log_videos(
            obses,
            rewards_per_step,
            self.logger,
            self.model.num_timesteps,
            frame_stack=frame_stack,
        )

    def eval_model(self):
        """评估模型性能"""
        logger.info("开始收集评估rollout...")
        obs = self.test_env.reset()
        dones = [False]
        reward = 0
        obses = []
        rewards_per_step = []
        while not dones[0]:
            action, _ = self.model.predict(obs, deterministic=False)
            obs, rewards, dones, info = self.test_env.step(action)
            reward += rewards[0]
            rewards_per_step.append(rewards[0])
            obses.append(obs)
        
        logger.info(f"评估完成，总回报: {reward}")
        return reward, obses, rewards_per_step


def create_offpolicy_callbacks(config, model, num_tactiles, objects, holders):
    """
    为离线策略算法创建回调函数集合
    """
    no_tactile = num_tactiles == 0
    project_name = "MultimodalLearning"
    if config.env in ["Door"]:
        project_name += "_robosuite"

    callbacks = []

    # 使用专门为离线策略设计的评估回调
    eval_callback = OffPolicyEvalCallback(
        config.env,
        config.state_type,
        no_tactile=no_tactile,
        representation=config.representation,
        eval_freq=config.eval_every,  # 使用基于时间步的频率
        config=config,
        objects=objects,
        holders=holders,
        camera_idx=config.camera_idx,
        frame_stack=config.frame_stack,
    )
    callbacks.append(eval_callback)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(config.save_freq // config.n_envs, 1),
        save_path="./logs/",
        name_prefix="rl_model",
        save_replay_buffer=True,  # 为离线策略保存回放缓冲区
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(OffPolicyTensorboardCallback())

    default_logger = configure_logger(
        verbose=1, tensorboard_log=model.tensorboard_log, tb_log_name="SAC"
    )
    wandb.init(
        project=project_name,
        config=config,
        save_code=True,
        name=default_logger.dir.split("/")[-1],
        dir=config.wandb_dir,
        id=config.wandb_id,
        entity=config.wandb_entity,
    )
    logger = WandbLogger(
        default_logger.dir, default_logger.output_formats, log_interval=1000
    )
    model.set_logger(logger)
    checkpoint_callback.save_path = wandb.run.dir

    return callbacks
