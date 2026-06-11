# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Ensure ALL output directories exist before running benchmarks
New-Item -ItemType Directory -Force -Path "profile_results" | Out-Null
New-Item -ItemType Directory -Force -Path "dependence_results" | Out-Null
New-Item -ItemType Directory -Force -Path "lander_results" | Out-Null

Write-Host "Running vectorized factored credit assignment benchmarks..."
python -u runner_vectorized.py *> dependence_results\vectorized_run.log

Write-Host "Vectorized complete."
Write-Host "Cartpole benchmarks"

python -u profile_vec_rl.py --steps 200000 --envs 64 --device auto *> profile_results\final_vectorized_results.txt

Write-Host "Vectorized Complete."
Write-Host "Running lander benchmarks"

python lander_vectorized.py *> lander_results\lander_vectorized_run.log

Write-Host "Lander benchmarks complete."