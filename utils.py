# utils.py

import numpy as np
import gym
from gym import Wrapper
from tqdm import tqdm
import os
import tensorflow as tf

class ProcgenEnvCompatibilityWrapper(Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        try:
            return self.env.render(mode)
        except Exception as e:
            print(f"Warning: render() failed with mode '{mode}': {e}")
            return None

    def close(self):
        try:
            self.env.close()
        except Exception as e:
            print(f"Warning: close() encountered an error: {e}")

def make_env(seed=15, start_level=None, game='coinrun-v0', num_levels=200, difficulty="easy", background=True, render_mode='rgb_array'):
    def create_single_env(start_level=start_level):
        if start_level is None:
            start_level = np.random.randint(num_levels)
        env = gym.make(
            "procgen:procgen-" + game,
            num_levels=num_levels,
            start_level=start_level,
            rand_seed=seed,
            render_mode=render_mode,
            distribution_mode=difficulty,
            use_backgrounds=background
        )
        env = ProcgenEnvCompatibilityWrapper(env)

        return env

    return create_single_env

def test_agent(agent=None, envs=None, total_timesteps=1000, agent_type="ppo"):
    """Test the PPO agent using the separated policy and value networks."""
    obs, _ = envs.reset()

    num_envs = envs.num_envs
    episode_rewards = [[] for _ in range(num_envs)]  
    episode_lengths = [[] for _ in range(num_envs)]  

    current_rewards = np.zeros(num_envs)  
    current_lengths = np.zeros(num_envs, dtype=int)  

    timesteps = 0

    with tqdm(total=total_timesteps, desc="Testing Agent", dynamic_ncols=True) as pbar:
        while timesteps < total_timesteps:
            if agent_type == "ppo" and agent is not None:
                actions, _, _ = agent.select_action(obs, training=False)
                actions = actions.squeeze() 
            elif agent_type == "random":
                actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
            else:
                raise ValueError("Invalid agent_type. Must be 'ppo' or 'random'.")

            next_obs, rewards, dones, _, infos = envs.step(actions)

            current_rewards += rewards
            current_lengths += 1

            for i in range(num_envs):
                if dones[i]:
                    episode_rewards[i].append(current_rewards[i])
                    episode_lengths[i].append(current_lengths[i])

                    current_rewards[i] = 0
                    current_lengths[i] = 0

            obs = next_obs
            timesteps += num_envs
            pbar.update(num_envs)

    all_rewards = [reward for env_rewards in episode_rewards for reward in env_rewards]
    all_lengths = [length for env_lengths in episode_lengths for length in env_lengths]

    avg_reward = np.mean(all_rewards) if all_rewards else 0
    std_reward = np.std(all_rewards) if all_rewards else 0
    avg_length = np.mean(all_lengths) if all_lengths else 0
    std_length = np.std(all_lengths) if all_lengths else 0

    print(f"{agent_type.upper()} Agent Test Results: Avg Reward: {avg_reward:.2f}, Std Reward: {std_reward:.2f}, Avg Length: {avg_length:.2f}, in {len(all_rewards)} episodes")

    return {
        "episodes_counter": len(all_rewards),
        "average_reward": avg_reward,
        "std_reward": std_reward,
        "average_length": avg_length,
        "std_length": std_length,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }

def save_models(policy_model, value_model, name, directory, log=False):
    """Save both the policy and value models."""
    policy_path = os.path.join(directory, name + "_policy.keras")
    value_path = os.path.join(directory, name + "_value.keras")
    
    policy_model.save(policy_path)
    value_model.save(value_path)
    
    if log:
        print(f"Models saved to {directory}:")
        print(f"  - Policy Model: {policy_path}")
        print(f"  - Value Model: {value_path}")

def load_models(agent, name, directory, log=False):
    """Load both the policy and value models into the PPO agent."""
    policy_path = os.path.join(directory, name + "_policy.keras")
    value_path = os.path.join(directory, name + "_value.keras")

    agent.policy_network = tf.keras.models.load_model(policy_path)
    agent.value_network = tf.keras.models.load_model(value_path)

    if log:
        print(f"Models loaded successfully from {directory}:")
        print(f"  - Policy Model: {policy_path}")
        print(f"  - Value Model: {value_path}")
