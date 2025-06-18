import os
import re
import pickle as pkl
import numpy as np
from pathlib import Path
import jax
import jax.numpy as jnp
from absl import flags, app
import msgpack
from functools import partial

import gym
import franka_sim  # 确保环境注册
from serl_launcher.agents.continuous.drq import DrQAgent
from serl_launcher.utils.launcher import make_drq_agent
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.common.evaluation import evaluate
from gym.wrappers.record_episode_statistics import RecordEpisodeStatistics

FLAGS = flags.FLAGS

flags.DEFINE_string("env", "PandaPickCubeVision-v0", "Name of environment.")
flags.DEFINE_string("checkpoint_path", "/media/midea/data/serl/franka/1.0", "Path to load checkpoints.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_string("rlds_path", None, "Path to RLDS logs (optional).")
flags.DEFINE_boolean("render", False, "Render the environment.")

def find_latest_checkpoint(checkpoint_path):
    """查找最新的checkpoint目录"""
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"checkpoint路径不存在: {checkpoint_path}")
    
    # 查找checkpoint_数字目录
    all_dirs = os.listdir(checkpoint_path)
    step_dir_pairs = []
    for d in all_dirs:
        m = re.match(r"checkpoint_(\d+)", d)
        if m:
            step_dir_pairs.append((int(m.group(1)), os.path.join(checkpoint_path, d)))
    
    if not step_dir_pairs:
        raise RuntimeError(
            f"未找到任何checkpoint于{checkpoint_path}，目录内容为: {all_dirs}。\n"
            "请确认训练脚本已正确保存checkpoint（应为'checkpoint_数字'命名的子目录）"
        )
    
    # 找到最新的checkpoint
    latest_step, latest_dir = max(step_dir_pairs, key=lambda x: x[0])
    print(f"找到最新checkpoint: {latest_dir}，步数: {latest_step}")
    
    return latest_dir, latest_step

def load_checkpoint_params(checkpoint_dir):
    """手动加载checkpoint文件中的参数"""
    try:
        # 尝试加载flax checkpoint
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint")
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, "rb") as f:
                state_dict = pkl.load(f)
                print(f"成功加载checkpoint: {checkpoint_file}")
                return state_dict
        
        # 如果找不到flax checkpoint，尝试加载msgpack文件
        msgpack_file = os.path.join(checkpoint_dir, "checkpoint")
        if os.path.exists(msgpack_file):
            with open(msgpack_file, "rb") as f:
                state_dict = msgpack.unpack(f, raw=False)
                print(f"成功加载msgpack checkpoint: {msgpack_file}")
                return state_dict
    except Exception as e:
        print(f"加载checkpoint失败: {e}")
    
    return None

def debug_print_shape_dict(d, name="数据"):
    """打印字典中每个键的形状，用于调试"""
    print(f"\n==== {name}形状信息 ====")
    for k, v in d.items():
        if isinstance(v, dict):
            debug_print_shape_dict(v, name=f"{name}.{k}")
        elif hasattr(v, 'shape'):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {type(v)}")
    print("=====================")

def manual_evaluate(agent, env, num_episodes=5, seed=42):
    """手动实现评估函数，确保正确收集结果"""
    
    returns = []
    lengths = []
    successes = []
    
    eval_rng = jax.random.PRNGKey(seed)
    
    for episode in range(num_episodes):
        print(f"评估回合 {episode+1}/{num_episodes}")
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0
        episode_length = 0
        
        while not (done or truncated):
            # 获取动作
            eval_rng, key = jax.random.split(eval_rng)
            action = agent.sample_actions(
                observations=jax.device_put(obs), 
                seed=key,
                deterministic=True
            )
            action = np.asarray(jax.device_get(action))
            
            # 执行动作
            next_obs, reward, done, truncated, info = env.step(action)
            
            # 累积回报和长度
            episode_return += reward
            episode_length += 1
            
            # 更新观察
            obs = next_obs
        
        # 记录结果
        returns.append(float(episode_return))
        lengths.append(episode_length)
        successes.append(info.get('success', False))
        
        print(f"  回合{episode+1} - 回报: {episode_return}, 长度: {episode_length}")
    
    # 整合结果
    eval_info = {
        'return': np.array(returns),
        'episode_length': np.array(lengths),
        'success': np.array(successes)
    }
    
    return eval_info

def main(_):
    rng = jax.random.PRNGKey(FLAGS.seed)
    
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

    image_keys = [key for key in env.observation_space.keys() if key != "state"]

    # 创建agent
    agent = make_drq_agent(
        seed=FLAGS.seed,
        sample_obs=env.observation_space.sample(),
        sample_action=env.action_space.sample(),
        image_keys=image_keys,
        encoder_type=FLAGS.encoder_type,
    )

    try:
        # 尝试加载最新checkpoint
        latest_checkpoint_dir, step = find_latest_checkpoint(FLAGS.checkpoint_path)
        state_dict = load_checkpoint_params(latest_checkpoint_dir)
        
        if state_dict and 'params' in state_dict:
            print(f"更新agent参数，checkpoint步数: {step}")
            # 更新agent参数
            agent = agent.replace(state=agent.state.replace(params=state_dict['params']))
        else:
            print("警告：无法加载有效的checkpoint参数，将使用随机初始化的参数")
    except Exception as e:
        print(f"加载checkpoint时出错: {e}")
        print("将使用随机初始化的参数继续评估")

    # 创建评估环境
    eval_env = gym.make(FLAGS.env)
    if FLAGS.env == "PandaPickCubeVision-v0":
        eval_env = SERLObsWrapper(eval_env)
        eval_env = ChunkingWrapper(eval_env, obs_horizon=1, act_exec_horizon=None)
    eval_env = RecordEpisodeStatistics(eval_env)

    # 复制agent到设备
    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)
    # 使用最新的JAX API，避免弃用警告
    agent = jax.device_put(jax.tree_util.tree_map(jnp.array, agent), sharding.replicate())

    print("开始评估...")
    
    # 明确为评估创建随机密钥
    eval_rng = jax.random.PRNGKey(FLAGS.seed + 100)  # 使用不同的种子以避免潜在的冲突
    
    # 创建自定义策略函数，确保传递正确的随机密钥
    def safe_policy_fn(observations):
        nonlocal eval_rng
        eval_rng, key = jax.random.split(eval_rng)
        return np.asarray(jax.device_get(
            agent.sample_actions(
                observations=jax.device_put(observations),
                seed=key,  # 显式传递有效的随机密钥
                deterministic=True
            )
        ))
    
    try:
        # 尝试使用evaluate函数进行评估
        eval_info = evaluate(
            policy_fn=safe_policy_fn,
            env=eval_env,
            num_episodes=FLAGS.eval_n_trajs,
        )
        
        if not eval_info or len(eval_info) == 0:
            print("原始评估函数返回空结果，切换到手动评估...")
            eval_info = manual_evaluate(agent, eval_env, FLAGS.eval_n_trajs, FLAGS.seed + 100)
    except Exception as e:
        print(f"使用原始评估函数时出错：{e}")
        print("切换到手动评估...")
        eval_info = manual_evaluate(agent, eval_env, FLAGS.eval_n_trajs, FLAGS.seed + 100)
    
    # 打印评估结果
    print("\n===== 评估结果 =====")
    if 'return' in eval_info:
        print(f"平均回报: {np.mean(eval_info['return']):.2f} ± {np.std(eval_info['return']):.2f}")
        print(f"所有回合回报: {eval_info['return']}")
    
    if 'episode_length' in eval_info:
        print(f"平均回合长度: {np.mean(eval_info['episode_length']):.2f}")
    
    if 'success' in eval_info:
        success_rate = np.mean(eval_info['success'])
        print(f"成功率: {success_rate:.2f}")
    
    print("===================\n")

if __name__ == "__main__":
    app.run(main)