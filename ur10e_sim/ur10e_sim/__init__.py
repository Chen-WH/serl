from ur10e_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register

register(
    id="UR10ePickCube-v0",
    entry_point="ur10e_sim.envs:UR10ePickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="UR10ePickCubeVision-v0",
    entry_point="ur10e_sim.envs:UR10ePickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
