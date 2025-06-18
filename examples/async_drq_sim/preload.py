#!/usr/bin/env python3

import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import gym
import time
import glob
from absl import app, flags
from flax.training import checkpoints
import cv2

from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.utils.launcher import make_drq_agent

import franka_sim

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to the checkpoint", required=True)
flags.DEFINE_integer("checkpoint_step", None, "Specific checkpoint step to load")
flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_integer("num_episodes", 10, "Number of episodes to run.")
flags.DEFINE_boolean("render", True, "Render the environment.")
flags.DEFINE_boolean("record_video", False, "Record video of the episodes.")
flags.DEFINE_string("video_path", "/media/midea/data/serl/videos", "Path to save videos.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("deterministic", True, "Whether to use deterministic policy.")
flags.DEFINE_float("sleep_time", 0.0, "Time to sleep between steps for visualization.")

def list_checkpoint_dirs(base_path):
    """列出并分析checkpoint目录"""
    checkpoint_dirs = []
    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)
        if os.path.isdir(full_path) and item.startswith("checkpoint_"):
            try:
                step = int(item.split("_")[1])
                checkpoint_dirs.append((step, full_path))
            except (IndexError, ValueError):
                continue
    
    # 按步骤排序
    checkpoint_dirs.sort(key=lambda x: x[0])
    return checkpoint_dirs

def main(_):
    # 设置设备
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    
    # 创建环境
    if FLAGS.render:
        env = gym.make(FLAGS.env, render_mode="human")
    else:
        env = gym.make(FLAGS.env)
        
    if FLAGS.env == "PandaPickCube-v0":
        env = gym.wrappers.FlattenObservation(env)
    if FLAGS.env == "PandaPickCubeVision-v0":
        env = SERLObsWrapper(env)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)

    # 设置视频录制
    video_recorder = None
    if FLAGS.record_video:
        os.makedirs(FLAGS.video_path, exist_ok=True)
        video_file = os.path.join(FLAGS.video_path, f"{FLAGS.env}_{int(time.time())}.mp4")
        if hasattr(env, "render_mode") and env.render_mode == "human":
            print("警告: 在render_mode为'human'时无法录制视频。使用rgb_array替代。")
            record_env = gym.make(FLAGS.env, render_mode="rgb_array")
            if FLAGS.env == "PandaPickCubeVision-v0":
                record_env = SERLObsWrapper(record_env)
                record_env = ChunkingWrapper(record_env, obs_horizon=1, act_exec_horizon=None)
            fps = 30
            width, height = 640, 480
            video_recorder = cv2.VideoWriter(
                video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
            )
            print(f"正在录制视频到 {video_file}")

    # 识别观察空间中的图像键
    image_keys = [key for key in env.observation_space.keys() if key != "state"]
    
    # 初始化智能体
    rng = jax.random.PRNGKey(FLAGS.seed)
    sample_obs = env.observation_space.sample()
    sample_action = env.action_space.sample()
    
    # 创建具有相同架构的智能体
    agent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=sample_obs,
        sample_action=sample_action,
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )
    
    # 检查checkpoint并加载模型
    print(f"正在查找checkpoints: {FLAGS.checkpoint_path}")
    
    # 分析checkpoint目录结构
    checkpoint_dirs = list_checkpoint_dirs(FLAGS.checkpoint_path)
    
    if not checkpoint_dirs:
        raise ValueError(f"在{FLAGS.checkpoint_path}中未找到checkpoint目录")
    
    # 确定要加载的checkpoint
    if FLAGS.checkpoint_step is not None:
        # 查找指定步骤的checkpoint
        target_dir = None
        for step, dir_path in checkpoint_dirs:
            if step == FLAGS.checkpoint_step:
                target_dir = dir_path
                break
        if target_dir is None:
            raise ValueError(f"未找到步骤为{FLAGS.checkpoint_step}的checkpoint")
    else:
        # 使用最新的checkpoint
        _, target_dir = checkpoint_dirs[-1]
        
    checkpoint_file = os.path.join(target_dir, "checkpoint")
    print(f"正在加载checkpoint: {target_dir}")
    
    # 尝试加载checkpoint
    try:
        restored_state = checkpoints.restore_checkpoint(
            ckpt_dir=target_dir,
            target=agent.state
        )
        
        if restored_state is None:
            raise ValueError(f"无法从{target_dir}加载checkpoint")
            
        agent = agent.replace(state=restored_state)
        print(f"成功加载checkpoint: {target_dir}")
        
    except Exception as e:
        print(f"加载checkpoint时出错: {e}")
        print("正在显示checkpoint目录内容:")
        
        # 显示目录内容以帮助诊断
        for root, dirs, files in os.walk(target_dir):
            level = root.replace(target_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                print(f"{sub_indent}{f}")
                
        raise ValueError(f"无法加载checkpoint: {target_dir}") from e
    
    # 在设备间复制agent
    agent = jax.device_put(
        jax.tree_map(jnp.array, agent), sharding.replicate()
    )
    
    # 定义策略函数
    def policy_fn(observations):
        return agent.sample_actions(
            observations=jax.device_put(observations), 
            deterministic=FLAGS.deterministic
        )
    
    # 运行评估
    print(f"使用{'确定性' if FLAGS.deterministic else '随机性'}策略运行{FLAGS.num_episodes}个回合...")
    
    total_returns = 0.0
    for episode in range(FLAGS.num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        steps = 0
        
        while not done:
            # 获取策略动作
            action = policy_fn(obs)
            action = np.asarray(jax.device_get(action))
            
            # 执行环境步骤
            next_obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            steps += 1
            
            # 录制视频
            if FLAGS.record_video and video_recorder is not None:
                frame = record_env.render()
                video_recorder.write(frame)
            
            # 可视化间隔
            if FLAGS.sleep_time > 0:
                time.sleep(FLAGS.sleep_time)
            
            # 更新观察
            obs = next_obs
            if truncated:
                break
                
        total_returns += episode_return
        print(f"回合 {episode + 1}: 回报 = {episode_return:.2f}, 步数 = {steps}")
    
    # 关闭视频录制器
    if FLAGS.record_video and video_recorder is not None:
        video_recorder.release()
    
    print(f"{FLAGS.num_episodes}个回合的平均回报: {total_returns / FLAGS.num_episodes:.2f}")
    
    # 使用evaluate函数进行额外评估
    print("\n正在运行正式评估...")
    eval_info = evaluate(
        policy_fn=partial(agent.sample_actions, argmax=FLAGS.deterministic),
        env=env,
        num_episodes=FLAGS.num_episodes,
    )
    
    print(f"评估结果: {eval_info}")

if __name__ == "__main__":
    app.run(main)