#!/usr/bin/env python3
"""
Test script to sync your most recent training run to Neptune
This demonstrates how neptune_sync.py works
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import neptune
import os
import json

# Configuration
results_dir = 'results'
run_name = 'run_9'  # Your most recent test run

print("=" * 70)
print("TESTING NEPTUNE SYNC WITH YOUR TEST RUN")
print("=" * 70)

# Read tags
run_path = os.path.join(results_dir, run_name)
with open(os.path.join(run_path, 'tags'), 'r') as f:
    tags = f.read().splitlines()

print(f"\nRun: {run_name}")
print(f"Tags: {tags}")

# Initialize Neptune run
print(f"\nConnecting to Neptune project: {os.getenv('NEPTUNE_PROJECT')}...")
neptune_run = neptune.init_run(
    project=os.getenv('NEPTUNE_PROJECT'),
    api_token=os.getenv('NEPTUNE_TOKEN'),
    tags=tags + ['test-sync'],
    name=f"Test sync - {run_name}",
)

print(f"✓ Created Neptune run: {neptune_run._sys_id}")
print(f"  View at: {neptune_run.get_url()}")

# Upload config
print("\nUploading configuration...")
with open('config/config.json') as f:
    config = json.load(f)
neptune_run['config'] = config

# Upload evaluation metrics
print("\nUploading evaluation metrics...")
logs_path = os.path.join(run_path, 'logs', 'eval')

# Average speedup (single value)
with open(os.path.join(logs_path, 'average_speedup'), 'r') as f:
    avg_speedup = float(f.read().strip())
    neptune_run['eval/average_speedup'] = avg_speedup
    print(f"  • Average speedup: {avg_speedup:.3f}x")

# Final speedups (per benchmark)
with open(os.path.join(logs_path, 'final_speedup'), 'r') as f:
    speedups = [float(line.strip()) for line in f.readlines() if line.strip()]
    for i, speedup in enumerate(speedups):
        neptune_run[f'eval/final_speedup'].append(speedup)
    print(f"  • Final speedups: {len(speedups)} values")

# Rewards
with open(os.path.join(logs_path, 'reward'), 'r') as f:
    rewards = [float(line.strip()) for line in f.readlines() if line.strip()]
    for reward in rewards:
        neptune_run['eval/reward'].append(reward)
    print(f"  • Rewards: {len(rewards)} values")

# Cumulative reward
with open(os.path.join(logs_path, 'cumulative_reward'), 'r') as f:
    cum_rewards = [float(line.strip()) for line in f.readlines() if line.strip()]
    for cum_reward in cum_rewards:
        neptune_run['eval/cumulative_reward'].append(cum_reward)
    print(f"  • Cumulative rewards: {len(cum_rewards)} values")

# Upload training metrics
print("\nUploading training metrics...")
logs_path = os.path.join(run_path, 'logs', 'train')

with open(os.path.join(logs_path, 'reward'), 'r') as f:
    train_rewards = [float(line.strip()) for line in f.readlines() if line.strip()]
    for reward in train_rewards:
        neptune_run['train/reward'].append(reward)
    print(f"  • Training rewards: {len(train_rewards)} values")

with open(os.path.join(logs_path, 'entropy'), 'r') as f:
    entropies = [float(line.strip()) for line in f.readlines() if line.strip()]
    for entropy in entropies:
        neptune_run['train/entropy'].append(entropy)
    print(f"  • Entropy: {len(entropies)} values")

print("\n" + "=" * 70)
print("✓ SYNC COMPLETE!")
print("=" * 70)
print(f"\nView your results at:")
print(f"  {neptune_run.get_url()}")
print(f"\nYou should see:")
print(f"  • Config parameters")
print(f"  • Training plots (reward, entropy)")
print(f"  • Evaluation plots (speedup, reward)")
print(f"  • Tags: {', '.join(tags + ['test-sync'])}")
print()

# Stop the run
neptune_run.stop()
print("Neptune run stopped.")
