"""
Main script for Dynamic Airspace Sectorisation (DAS) using PPO.

Usage
-----
Training (uncomment the block below):
    Adjust total_timesteps and hyperparameters, then run this script.
    The trained model is saved as 'ppo_policy.zip'.

Testing (default):
    Loads a pre-trained PPO policy and evaluates it over `test_time` episodes,
    reporting average reward and per-interval wall-clock runtime.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from gym_foo.envs.foo_env import FooEnv  # noqa: F401 — registers the custom env
import time
import numpy as np

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env = gym.make('foo-v1')

# ---------------------------------------------------------------------------
# Training  (uncomment to retrain from scratch)
# ---------------------------------------------------------------------------
# model = PPO(
#     'MlpPolicy',
#     env,
#     learning_rate=0.0003,
#     n_steps=3,
# ).learn(total_timesteps=30000)
# model.save('ppo_policy')
# print('---------- TRAIN ENDS ----------')

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
print('---------- TEST STARTS ----------')

model = PPO.load('ppo_policy')

test_time = 1        # number of tests
num_intervals = 6 * 7    # must match env.max_step

all_rewards = 0
interval_runtimes = [0.0] * num_intervals

# Per-interval outcome accumulators (summed over test runs for averaging)
interval_conflicts     = [0.0] * num_intervals
interval_crossings     = [0.0] * num_intervals
interval_dissimilarity = [0.0] * num_intervals
interval_rewards       = [0.0] * num_intervals

for i in range(test_time):
    obs, _ = env.reset()
    total_reward = 0

    for interval in range(num_intervals):
        t0 = time.time()

        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        conflict     = info['conflict']/10000 * 0.4
        crossing     = info['crossing']/10000 * 0.3
        dissimilarity = info['dis-similarity'] * 0.3

        print(f"interval: {interval}")
        print(f"  conflict:      {conflict}")
        print(f"  crossing:      {crossing}")
        print(f"  dissimilarity: {dissimilarity}")

        interval_conflicts[interval]     += conflict
        interval_crossings[interval]     += crossing
        interval_dissimilarity[interval] += dissimilarity
        interval_rewards[interval]       += reward

        total_reward += reward
        interval_runtimes[interval] += time.time() - t0

    all_rewards += total_reward


avg_rewards            = round(all_rewards / test_time, 2)
avg_interval_runtimes  = [rt / test_time for rt in interval_runtimes]
avg_conflicts          = [v / test_time for v in interval_conflicts]
avg_crossings          = [v / test_time for v in interval_crossings]
avg_dissimilarity      = [v / test_time for v in interval_dissimilarity]
avg_interval_rewards   = [v / test_time for v in interval_rewards]

# ── Per-interval report ──────────────────────────────────────────────────────
print('\n========== PER-INTERVAL OUTCOMES (averaged over test runs) ==========')
print(f"{'Interval':>8}  {'Conflict':>10}  {'Crossing':>10}  {'Dissimilarity':>13}  {'Reward':>8}")
for idx in range(num_intervals):
    print(f"{idx:>8}  {avg_conflicts[idx]:>10.4f}  {avg_crossings[idx]:>10.4f}"
          f"  {avg_dissimilarity[idx]:>13.4f}  {avg_interval_rewards[idx]:>8.4f}")

# ── Daily (every-6-interval) totals ─────────────────────────────────────────
print('\n========== DAILY TOTALS (sum over 6 intervals, averaged over test runs) ==========')
print(f"{'Day':>4}  {'Intervals':>12}  {'Conflict':>10}  {'Crossing':>10}  {'Dissimilarity':>13}  {'Reward':>8}")
days = num_intervals // 6
for day in range(days):
    s, e = day * 6, (day + 1) * 6
    print(f"{day+1:>4}  {s:>5}–{e-1:<5}  "
          f"{sum(avg_conflicts[s:e]):>10.4f}  {sum(avg_crossings[s:e]):>10.4f}"
          f"  {sum(avg_dissimilarity[s:e]):>13.4f}  {sum(avg_interval_rewards[s:e]):>8.4f}")

# ── Runtime report ───────────────────────────────────────────────────────────
print(f'\nAverage of total rewards: {avg_rewards}')
for idx, rt in enumerate(avg_interval_runtimes, 1):
    print(f'Average runtime for interval {idx}: {rt:.3f} seconds')
print(f'Total average runtime across all intervals: {sum(avg_interval_runtimes):.3f} seconds')

env.close()
