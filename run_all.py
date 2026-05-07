"""
run_all.py - Master Script for Micro-loan RL Pricing
=====================================================
Runs the complete pipeline in order:
  1. Train PPO agent
  2. Run baseline comparison  (Graphs 1 & 2)
  3. Run stress tests         (Graphs 3, 4 & 5)

Usage:
    python run_all.py              # Full pipeline
    python run_all.py --skip-train # Skip training, just generate graphs
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description="Micro-loan RL Pricing Pipeline")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training and use existing model"
    )
    parser.add_argument(
        "--timesteps", type=int, default=100_000,
        help="Training timesteps (default: 100000)"
    )
    args = parser.parse_args()

    os.makedirs("results/graphs", exist_ok=True)

    print("\n" + "█" * 60)
    print("  MICRO-LOAN DYNAMIC PRICING - MIFOS DMP 2026")
    print("  Full Pipeline Runner")
    print("█" * 60)

    # ── Step 1: Train ────────────────────────────────────────────────────────
    if not args.skip_train:
        print("\n[STEP 1/3] Training PPO Agent...")
        from agents.train_ppo import train_ppo_agent
        model, callback = train_ppo_agent(
            total_timesteps=args.timesteps,
            save_path="results/ppo_microloan_model",
            verbose=1,
        )
        print("Training complete!")
    else:
        print("\n[STEP 1/3] Skipping training (--skip-train flag set)")

    # ── Step 2: Baseline Comparison ──────────────────────────────────────────
    print("\n[STEP 2/3] Running Baseline Comparison...")
    from evaluation.baseline import run_baseline_comparison
    results = run_baseline_comparison(
        model_path="results/ppo_microloan_model",
        n_episodes=200,
        output_dir="results/graphs",
    )

    # ── Step 3: Stress Tests + Fairness ─────────────────────────────────────
    print("\n[STEP 3/3] Running Stress Tests & Fairness Analysis...")
    from evaluation.stress_test import run_stress_tests
    run_stress_tests(
        model_path="results/ppo_microloan_model",
        n_episodes=100,
        output_dir="results/graphs",
    )

    # ── Done ─────────────────────────────────────────────────────────────────
    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE!")
    print("  All 5 graphs saved to: results/graphs/")
    print()
    print("  graph1_reward_comparison.png")
    print("  graph2_default_rate_comparison.png")
    print("  graph3_learning_curve.png")
    print("  graph4_stress_test.png")
    print("  graph5_fairness_analysis.png")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()