
uv run sweep.py \
	--config sweep_seeds_dev.txt \
	--template slurm-template.txt \
	--command sbatch \
	--base_dir exp/seeds_dev/
