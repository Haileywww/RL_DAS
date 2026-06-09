"""
Main script for Dynamic Airspace Sectorisation (DAS) using PPO.

Usage
-----
Training (uncomment the block below):
    Adjust total_timesteps and hyperparameters, then run this script.
    The trained model is saved as 'PPO_policy.zip'.

Testing (default):
    Loads a pre-trained PPO policy and evaluates it over `test_time` episodes,
    reporting average reward and per-interval wall-clock runtime.
"""

import os
import sys
import subprocess

# Auto install dependencies 
requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
pip_command = [sys.executable, '-m', 'pip', 'install', '-r', requirements_path]
# Linux specific flag
if sys.platform.startswith('linux'):
    pip_command.insert(4, '--break-system-packages')
print("Starting automated dependency installation...", flush=True)
try:
    subprocess.check_call(pip_command, stdout=sys.stdout, stderr=sys.stderr)
    print("Dependencies installed successfully!", flush=True)
except subprocess.CalledProcessError as e:
    print(f"Automated installation failed: {e}, please install manually refering to README", file=sys.stderr, flush=True)
    sys.exit(1)


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
#     verbose=1,
#     learning_rate=0.0003,
#     n_steps=3,
# ).learn(total_timesteps=30000)
# model.save('PPO_policy')
# print('Training complete — model saved as ppo_policy.zip')

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
MODEL_NAME    = 'PPO_policy'
test_time     = 1   # number of customized episodes
num_intervals = 6 * 7    # must match env.max_step

model = PPO.load(MODEL_NAME)

W = 60  # total width of the banner box
title   = 'Dynamic Airspace Sectorisation — DAS/PPO'
details = f'Model: {MODEL_NAME}  │  Episodes: {test_time}  │  Intervals: {num_intervals}'
print('╔' + '═' * W + '╗')
print('║' + title.center(W)   + '║')
print('║' + details.center(W) + '║')
print('╚' + '═' * W + '╝')
print()

all_rewards = 0
interval_runtimes = [0.0] * num_intervals

interval_conflicts     = [0.0] * num_intervals
interval_crossings     = [0.0] * num_intervals
interval_dissimilarity = [0.0] * num_intervals
interval_rewards       = [0.0] * num_intervals

# ── Per-interval report (printed live) ──────────────────────────────────────
_ROW  = '─' * 6 + '┼' + '─' * 12 + '┼' + '─' * 12 + '┼' + '─' * 15 + '┼' + '─' * 11
_FOOT = '─' * 6 + '┴' + '─' * 12 + '┴' + '─' * 12 + '┴' + '─' * 15 + '┴' + '─' * 11

section = ' PER-INTERVAL OUTCOMES '
print('═' * W)
print(section.center(W, '═'))
print('═' * W)
print(f'  (values shown per interval; daily/episode averages follow)\n')
print(f" {'Intv':>4} │ {'Conflict':>10} │ {'Crossing':>10} │ {'Dissimilarity':>13} │ {'Reward':>9}")
print(_ROW, flush=True)

for i in range(test_time):
    if test_time > 1:
        ep_label = f'  ── Episode {i + 1} / {test_time} ──'
        print(ep_label)
    obs, _ = env.reset()
    total_reward = 0

    for interval in range(num_intervals):
        t0 = time.time()

        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        conflict      = info['conflict']      / 10000 * 0.4
        crossing      = info['crossing']      / 10000 * 0.3
        dissimilarity = info['dis-similarity']         * 0.3

        interval_conflicts[interval]     += conflict
        interval_crossings[interval]     += crossing
        interval_dissimilarity[interval] += dissimilarity
        interval_rewards[interval]       += reward

        total_reward += reward
        interval_runtimes[interval] += time.time() - t0

        print(f" {interval:>4} │ {conflict:>10.4f} │ {crossing:>10.4f}"
              f" │ {dissimilarity:>13.4f} │ {reward:>9.4f}", flush=True)

    all_rewards += total_reward

print(_FOOT)

avg_rewards            = round(all_rewards / test_time, 2)
avg_interval_runtimes  = [rt / test_time for rt in interval_runtimes]
avg_conflicts          = [v  / test_time for v  in interval_conflicts]
avg_crossings          = [v  / test_time for v  in interval_crossings]
avg_dissimilarity      = [v  / test_time for v  in interval_dissimilarity]
avg_interval_rewards   = [v  / test_time for v  in interval_rewards]

# ── Per-interval averages (only shown when test_time > 1) ───────────────────
if test_time > 1:
    print()
    section = ' PER-INTERVAL AVERAGES (over all episodes) '
    print('═' * W)
    print(section.center(W, '═'))
    print('═' * W)
    print(f"\n {'Intv':>4} │ {'Conflict':>10} │ {'Crossing':>10} │ {'Dissimilarity':>13} │ {'Reward':>9}")
    print(_ROW)
    for idx in range(num_intervals):
        print(f" {idx:>4} │ {avg_conflicts[idx]:>10.4f} │ {avg_crossings[idx]:>10.4f}"
              f" │ {avg_dissimilarity[idx]:>13.4f} │ {avg_interval_rewards[idx]:>9.4f}")
    print(_FOOT)

# ── Daily (every-6-interval) totals ─────────────────────────────────────────
print()
section = ' DAILY TOTALS (sum of 6 intervals / day) '
print('═' * W)
print(section.center(W, '═'))
print('═' * W)
print(f"\n {'Day':>4} │ {'Intervals':>11} │ {'Conflict':>10} │ {'Crossing':>10}"
      f" │ {'Dissimilarity':>13} │ {'Reward':>9}")
print('─' * 6 + '┼' + '─' * 13 + '┼' + '─' * 12 + '┼' + '─' * 12 + '┼' + '─' * 15 + '┼' + '─' * 11)
days = num_intervals // 6
for day in range(days):
    s, e = day * 6, (day + 1) * 6
    interval_range = f'{s} – {e - 1}'
    print(f" {day+1:>4} │ {interval_range:>11} │ {sum(avg_conflicts[s:e]):>10.4f}"
          f" │ {sum(avg_crossings[s:e]):>10.4f} │ {sum(avg_dissimilarity[s:e]):>13.4f}"
          f" │ {sum(avg_interval_rewards[s:e]):>9.4f}")
print('─' * 6 + '┴' + '─' * 13 + '┴' + '─' * 12 + '┴' + '─' * 12 + '┴' + '─' * 15 + '┴' + '─' * 11)

# ── Runtime report ───────────────────────────────────────────────────────────
print()
section = ' RUNTIME SUMMARY '
print('═' * W)
print(section.center(W, '═'))
print('═' * W)
min_rt  = min(avg_interval_runtimes)
max_rt  = max(avg_interval_runtimes)
mean_rt = sum(avg_interval_runtimes) / len(avg_interval_runtimes)
total_rt = sum(avg_interval_runtimes)
print(f'  Per-interval avg : {mean_rt:.3f} s'
      f'  (fastest: {min_rt:.3f} s  slowest: {max_rt:.3f} s)')
print(f'  Total (all {num_intervals} intervals): {total_rt:.3f} s')

# ── Episode summary ──────────────────────────────────────────────────────────
print()
section = ' EPISODE SUMMARY '
print('═' * W)
print(section.center(W, '═'))
print('═' * W)
print(f'  Average total reward (over {test_time} episode{"s" if test_time > 1 else ""}): {avg_rewards}')
print('═' * W)

env.close()
