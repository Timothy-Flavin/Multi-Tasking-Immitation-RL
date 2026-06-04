#!/bin/bash
set -e

echo "Starting Baseline Benchmarks (6 runs)..."
python -u profile_rl.py --steps 200000 --device auto > final_baseline_results.txt 2>&1
echo "Baseline Complete."

echo "Starting Vectorized Benchmarks (12 runs: 6 Mixed, 6 Full GPU)..."
python -u profile_vec_rl.py --steps 200000 --envs 64 --device auto > final_vectorized_results.txt 2>&1
echo "Vectorized Complete."
