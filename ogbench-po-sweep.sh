
uv run sweep.py \
	--config ogbench-po-sweep.py \
	--template slurm-template.txt \
	--command sbatch \
	--base_dir exp/ogbench-po-v1/
