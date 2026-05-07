"""
train_ppo.py - PPO Agent Training for Micro-loan Dynamic Pricing
================================================================
Trains a Proximal Policy Optimization (PPO) agent using Stable-Baselines3
to learn optimal interest rate policies for microfinance borrowers.

Usage:
    python agents/train_ppo.py
"""

import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from environment.loan_env import MicroLoanEnv


# ────────────────────────────────────────────────────────────────────────────
# Custom Callback for Logging
# ────────────────────────────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    Custom callback that logs mean episode rewards during training.
    Used to generate the learning curve graph later.
    """

    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards: list = []
        self.timestep_log: list = []
        self._episode_reward = 0.0
        self._n_episodes = 0

    def _on_step(self) -> bool:
        """Called at each environment step."""
        # Accumulate reward
        reward = self.locals.get("rewards", [0])[0]
        self._episode_reward += reward

        # Check if episode ended
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self._episode_reward)
            self.timestep_log.append(self.num_timesteps)
            self._episode_reward = 0.0
            self._n_episodes += 1

            if self._n_episodes % 50 == 0 and self.verbose > 0:
                recent = self.episode_rewards[-50:]
                print(
                    f"  Episode {self._n_episodes:5d} | "
                    f"Timestep {self.num_timesteps:8d} | "
                    f"Mean reward (last 50): {np.mean(recent):+.2f}"
                )

        return True  # Continue training

    def get_learning_curve(self):
        """Return timesteps and smoothed mean rewards for plotting."""
        if not self.episode_rewards:
            return [], []
        # Smooth with rolling window
        window = 20
        smoothed = []
        for i in range(len(self.episode_rewards)):
            start = max(0, i - window)
            smoothed.append(np.mean(self.episode_rewards[start : i + 1]))
        return self.timestep_log, smoothed


# ────────────────────────────────────────────────────────────────────────────
# Training Function
# ────────────────────────────────────────────────────────────────────────────

def train_ppo_agent(
    total_timesteps: int = 100_000,
    save_path: str = "results/ppo_microloan_model",
    log_path: str = "results/ppo_logs",
    verbose: int = 1,
) -> tuple:
    """
    Train a PPO agent on the MicroLoanEnv.

    Architecture:
        - Policy: MlpPolicy
        - Network: [128, 128] hidden layers
        - Algorithm: PPO (Proximal Policy Optimization)

    Args:
        total_timesteps: Number of environment steps to train for
        save_path: Where to save the trained model
        log_path: Directory for TensorBoard logs
        verbose: Logging verbosity (0=silent, 1=info, 2=debug)

    Returns:
        model: Trained PPO model
        callback: RewardLoggerCallback containing learning curve data
    """
    print("=" * 60)
    print("  Micro-loan Dynamic Pricing - PPO Training")
    print("  Mifos Initiative DMP 2026")
    print("=" * 60)

    # Create output directories
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else "results", exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # ── Environment Setup ────────────────────────────────────────────────────
    # Wrap in Monitor for episode stats, then vectorize for SB3
    def make_env():
        env = MicroLoanEnv()
        env = Monitor(env)
        return env

    # Use 4 parallel environments for faster training
    n_envs = 4
    vec_env = make_vec_env(make_env, n_envs=n_envs)

    # Separate eval environment (single env, no parallelism)
    eval_env = Monitor(MicroLoanEnv())

    print(f"\nEnvironment: MicroLoanEnv")
    print(f"  Observation space: {vec_env.observation_space}")
    print(f"  Action space:      {vec_env.action_space}")
    print(f"  Parallel envs:     {n_envs}")
    print(f"  Total timesteps:   {total_timesteps:,}")
    print()

    # ── PPO Model ────────────────────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        # Network architecture: two hidden layers of 128 neurons
        policy_kwargs={"net_arch": [128, 128]},
        # PPO hyperparameters
        learning_rate=3e-4,       # Adam optimizer LR
        n_steps=2048,             # Steps per rollout per env
        batch_size=64,            # Mini-batch size
        n_epochs=10,              # Epochs per update
        gamma=0.99,               # Discount factor
        gae_lambda=0.95,          # GAE lambda
        clip_range=0.2,           # PPO clip range
        ent_coef=0.01,            # Entropy bonus (encourages exploration)
        vf_coef=0.5,              # Value function loss coefficient
        max_grad_norm=0.5,        # Gradient clipping
        tensorboard_log=log_path,
        verbose=verbose,
        seed=42,
    )

    print(f"PPO Policy Architecture: MlpPolicy [128 → 128]")
    print(f"Learning rate: 3e-4 | Clip range: 0.2 | Entropy coef: 0.01")
    print()

    # ── Callbacks ────────────────────────────────────────────────────────────
    # 1. Log rewards every step
    reward_callback = RewardLoggerCallback(log_interval=1000, verbose=verbose)

    # 2. Periodic checkpoints every 25k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=25_000 // n_envs,
        save_path=os.path.join(log_path, "checkpoints"),
        name_prefix="ppo_microloan",
    )

    # 3. Evaluate on held-out env every 10k steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(log_path, "best_model"),
        log_path=log_path,
        eval_freq=10_000 // n_envs,
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=0,
    )

    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([reward_callback, checkpoint_callback, eval_callback])

    # ── Training ─────────────────────────────────────────────────────────────
    print(f"Starting training for {total_timesteps:,} timesteps...")
    print("-" * 60)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    print("-" * 60)
    print(f"Training complete!")

    # ── Save Model ───────────────────────────────────────────────────────────
    model.save(save_path)
    print(f"Model saved to: {save_path}.zip")

    # Save learning curve data as numpy arrays
    timesteps, rewards = reward_callback.get_learning_curve()
    np.save(os.path.join("results", "learning_curve_timesteps.npy"), timesteps)
    np.save(os.path.join("results", "learning_curve_rewards.npy"), rewards)
    print(f"Learning curve data saved to results/")

    vec_env.close()
    eval_env.close()

    return model, reward_callback


# ────────────────────────────────────────────────────────────────────────────
# Quick Evaluation After Training
# ────────────────────────────────────────────────────────────────────────────

def evaluate_trained_agent(
    model_path: str = "results/ppo_microloan_model",
    n_episodes: int = 100,
) -> dict:
    """
    Run the trained PPO agent and collect metrics.

    Args:
        model_path: Path to saved model (without .zip)
        n_episodes: Number of evaluation episodes

    Returns:
        dict: Evaluation metrics (reward, default rate, acceptance rate, etc.)
    """
    print(f"\nEvaluating trained agent over {n_episodes} episodes...")

    model = PPO.load(model_path)
    env = MicroLoanEnv()

    total_rewards = []
    all_rates = []
    n_defaults = 0
    n_accepted = 0
    n_total = 0
    fairness_violations = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))

            ep_reward += reward
            n_total += 1
            rate = info["interest_rate"]
            all_rates.append(rate)

            if info["accepted"]:
                n_accepted += 1
            if info["defaulted"]:
                n_defaults += 1

            # Count fairness violations
            rb = info.get("reward_breakdown", {})
            if rb.get("fairness", 0) < 0:
                fairness_violations += 1

            done = terminated or truncated

        total_rewards.append(ep_reward)

    metrics = {
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "default_rate": n_defaults / max(n_accepted, 1),
        "acceptance_rate": n_accepted / n_total,
        "avg_interest_rate": float(np.mean(all_rates)),
        "fairness_violation_rate": fairness_violations / n_total,
        "total_episodes": n_episodes,
    }

    print(f"  Mean reward:          {metrics['mean_reward']:+.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Default rate:         {metrics['default_rate']:.2%}")
    print(f"  Acceptance rate:      {metrics['acceptance_rate']:.2%}")
    print(f"  Avg interest rate:    {metrics['avg_interest_rate']:.2f}%")
    print(f"  Fairness violations:  {metrics['fairness_violation_rate']:.2%}")

    env.close()
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Entry Point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Train the agent
    model, callback = train_ppo_agent(
        total_timesteps=100_000,
        save_path="results/ppo_microloan_model",
        verbose=1,
    )

    # Quick post-training evaluation
    metrics = evaluate_trained_agent(
        model_path="results/ppo_microloan_model",
        n_episodes=100,
    )

    print("\nDone! Run evaluation/baseline.py for full comparison.")