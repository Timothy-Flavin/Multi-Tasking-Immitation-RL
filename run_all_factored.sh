#!/bin/bash
set -e
mkdir -p fac_res
echo "Running baseline factored credit assignment benchmarks..."
python -u runner.py > fac_res/baseline_run.log 2>&1
echo "Baseline complete."

echo "Running vectorized factored credit assignment benchmarks..."
python -u runner_vectorized.py > fac_res/vectorized_run.log 2>&1
echo "Vectorized complete."

echo "Running LunarLander hybrid action benchmarks..."
python -u lander_vectorized.py > fac_res/lander_run.log 2>&1
echo "LunarLander complete."
