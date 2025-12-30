#!/bin/bash

#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=23:59:59
#SBATCH --job-name=crl_test
#SBATCH --output=sbatch_logs/crl_test.out
#SBATCH --error=sbatch_logs/crl_test.err

source $HOME/.bash_profile

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.7
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_DETERMINISTIC_OPS=0

module load stack/2024-06
module load cuda/12.9

cd $HOME/contrastive-gcrl-pomdp/impls

uv run main.py \
	--agent "agents/crl.py" \
	--env_name "antmaze-medium-navigate-v0"

