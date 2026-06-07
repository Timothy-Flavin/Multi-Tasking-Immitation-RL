source venv/bin/activate
mkdir -p profile_results
echo "Running vectorized factored credit assignment benchmarks..."
python -u runner_vectorized.py > dependence_results/vectorized_run.log 2>&1
echo "Vectorized complete."
echo "Cartpole benchmarks"
python -u profile_vec_rl.py --steps 200000 --envs 64 --device auto > profile_results/final_vectorized_results.txt 2>&1
echo "Vectorized Complete."
echo "Running lander benchmarks"
python lander_vectorized.py > lander_results/lander_vectorized_run.log 2>&1
echo "Lander benchmarks complete."
