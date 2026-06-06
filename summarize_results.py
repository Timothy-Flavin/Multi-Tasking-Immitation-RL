import json
import numpy as np

with open("dependence_results/vectorized_results.json", "r") as f:
    results = json.load(f)

for name, runs in results.items():
    final_rewards = []
    for run in runs:
        if run["reward_curve"]:
            # Last 10 episodes average
            last_rewards = [r[1] for r in run["reward_curve"][-160:]] # 16 envs * 10
            final_rewards.append(np.mean(last_rewards))
    if final_rewards:
        print(f"{name:<20} | Final Avg Reward: {np.mean(final_rewards):>8.2f} +/- {np.std(final_rewards):>8.2f}")
