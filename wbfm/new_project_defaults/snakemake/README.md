# Contains main pipeline commands

Please run "DRYRUN" before "RUNME"!

# Explanation of files

These are helper scripts:
- RUNME_*.sh 
- DRYRUN.sh

The core scripts are these:
- pipeline.smk - this defines the 'rules' that tell snakemake how to produce outputs from scripts, and what inputs those scripts need
- config.yaml - this defines a lot of helper variables and the code script names for the pipeline.smk file
- cluster_config.yaml - this defines SLURM variables, some of which are unique to each rule
- 