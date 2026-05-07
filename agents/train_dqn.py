"""
train_dqn.py - DQN Agent Training for Micro-loan Dynamic Pricing
================================================================
Alternative agent using Deep Q-Network (DQN) for comparison with PPO.
DQN is well-suited for discrete action spaces like our 57-rate action space.

Usage:
    python agents/train_dqn.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from environment.loan_env import MicroLoanEnv


class DQNRewardCallback(BaseCallback):
    """Track episode rewards during DQN training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list = []
        self.timestep_log: list = []
        self._ep_reward = 0.0
        self._n_ep = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])
        reward = reward[0] if hasattr(reward, "__len__") else reward
        self._ep_reward += reward

        done = self.locals.get("dones", [False])
        done = done[0] if hasattr(done, "__len__") else done

        if done:
            self.episode_rewards.append(self._ep_reward)
            self.timestep_log.append(self.num_timesteps)
            self._ep_reward = 0.0
            self._n_ep += 1

            if self._n_ep % 100 == 0 and self.verbose > 0:
                recent = self.episode_rewards[-50:]
                print(
                    f"  DQN Episode {self._n_ep:5d} | "
                    f"Timestep {self.num_timesteps:8d} | "
                    f"Mean (50ep): {np.mean(recent):+.2f}"
                )
        return True


def train_dqn_agent(
    total_timesteps: int = 100_000,
    save_path: str = "results/dqn_microloan_model",
    verbose: int = 1,
) -> tuple:
    """
    Train a DQN agent on the MicroLoanEnv.

    DQN uses experience replay and a target network to stabilize training
    on discrete action spaces. Useful as a comparison to PPO.

    Args:
        total_timesteps: Training budget
        save_path: Model save location
        verbose: Verbosity level

    Returns:
        model: Trained DQN model
        callback: Reward tracking callback
    """
    print("=" * 60)
    print("  Micro-loan Dynamic Pricing - DQN Training")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)

    # Single environment (DQN doesn't support vectorized envs in SB3 easily)
    train_env = Monitor(MicroLoanEnv())
    eval_env = Monitor(MicroLoanEnv())

    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space:      {train_env.action_space}")
    print(f"Total timesteps:   {total_timesteps:,}\n")

    # DQN model with double DQN and dueling enabled
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        policy_kwargs={
            "net_arch": [128, 128],
        },
        learning_rate=1e-4,
        buffer_size=50_000,          # Replay buffer size
        learning_starts=1_000,       # Steps before first update
        batch_size=64,
        tau=0.005,                   # Soft update coefficient
        gamma=0.99,
        train_freq=4,                # Update every 4 steps
        gradient_steps=1,
        target_update_interval=1000, # Hard update target network
        exploration_fraction=0.3,    # Explore for 30% of training
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log="results/dqn_logs",
        verbose=verbose,
        seed=42,
    )

    reward_cb = DQNRewardCallback(verbose=verbose)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="results/dqn_best",
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
        verbose=0,
    )

    print(f"Starting DQN training for {total_timesteps:,} timesteps...")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[reward_cb, eval_cb],
        progress_bar=True,
    )

    print("-" * 60)
    model.save(save_path)
    print(f"DQN model saved to: {save_path}.zip")

    train_env.close()
    eval_env.close()

    return model, reward_cb


if __name__ == "__main__":
    model, callback = train_dqn_agent(
        total_timesteps=100_000,
        save_path="results/dqn_microloan_model",
        verbose=1,
    )
    print("DQN training complete.")