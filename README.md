# Whole Brain Freely Moving Tracking and Trace extraction

This repository contains python code for analyzing raw volumetric images in two channels: red (tracking) and green (activity).

The segmentation portion of the algorithm is in a [sibling repository](https://github.com/Zimmer-lab/segmentation), but all main pipeline steps and GUIs are in this one.


# Installation for running

This project is designed to be installed with Anaconda, and requires two external local packages to be installed.
The following are step-by-step instructions.

If you just want to run the code, then you can use the pre-installed environments installed on the cluster, which can be activated using:
```
conda activate /scratch/neurobiology/zimmer/.conda/envs/wbfm/
```

Or:
```
conda activate /scratch/neurobiology/zimmer/.conda/envs/segmentation/
```


# Running a project: start from nothing

## Preparation:

1. Red channel (tracking) in a single bigtiff file
2. Green channel (signal) in a single bigtiff file
3. 2 conda environments (see above for installation instructions, or use pre-installed versions on the cluster)

## Full workflow

All examples assume that you are in the main folder:
```commandline
cd path/to/your/github/code/DLC_for_WBFM/
```

For reference, each command is in proper number order in DLC_for_WBFM/scripts

### Creating a project

Working examples to create a project are available for 
[linux](DLC_for_WBFM/scripts/examples/0-create_new_project-linux-EXAMPLE.sh)
and [windows](DLC_for_WBFM/scripts/examples/0-create_new_project-windows-EXAMPLE.sh)

Currently, everything is designed to run on the command line.
However, if your data is visible locally (mounted is okay), for the initial project creation you can use a gui:

```commandline
python gui/create_project_gui.py
```

### *IMPORTANT*

The method of creating the project will determine the style of filepaths that are saved in the project.
Thus, if you create and run the project on different operating systems, you will probably need to manually update the path to the raw data.

Specifically be aware of these variables:
```yaml
red_bigtiff_fname
green_bigtiff_fname
```

in the main project file: config.yaml

## Running full workflow using snakemake

This code is designed in several different scripts, which can be running using two commands.
The organization between these steps uses the workflow manager [snakemake](https://snakemake.readthedocs.io/en/stable/).
Snakemake works by keeping track of output files with special names, and only reliably works for the first run.
If you are rerunning an old project, see the next section.

Once a project is made, the analysis can be run in the following way:
1. Activate the segmentation environment
2. cd to the /snakemake folder within the project:
```commandline
cd /path/to/your/project/snakemake
```
3. Do a dry run to catch any errors in initialization:
```bash
bash DRYRUN_segmentation.sh
```
4. If there are errors, there are two options:
   1. If this is not a new project, you might have to run steps one by one (see the next subsection).
   2. If this is a new project, then it is probably a bug and you should file a GitHub issue and possibly talk to Charlie
5. Run the relevant RUNME script, either cluster or local. Probably, you want the cluster version:
```bash
bash RUNME_cluster_segmentation.sh
```
6. Wait for this to finish. Depending on scheduling, it could take 4-12 hours.
7. Check the log files (they will be in the /snakemake folder) to make sure there were no errors
8. Repeat steps 3-7 for the *_post_segmentation.sh DRYRUN and RUNME scripts

# Running a project: start from a previous project

Snakemake works by keeping track of output files with special names, and only reliably works for the first run.
Thus, you should run each step one by one.
Note that you can check the current status of the project by moving to the project and running a script. Example:
```bash
cd /path/to/your/project
python log/print_project_status.py
```

You can directly run the python scripts, or, most likely, run them using sbatch using the following syntax.

### Running single steps on the cluster (sbatch)

Once the project is created, each step can be run via sbatch using:

```commandline
sbatch single_step_dispatcher.sbatch -s 1 -t /scratch/neurobiology/zimmer/Charles/dlc_stacks/worm10-gui_test/project_config.yaml
```

where '-s' is a shortcut for the step to run (1, 2a, 2b, 2c, 3a, 3b, 4) and '-t' is a path to the project config file.


# Summary of GUIs

Initial creation of project:
```bash
python gui/create_project_gui.py --project_path PATH-TO-YOUR-PROJECT
```

Visualization of most steps in the analysis is also possible, and they can be accessed via the progress gui:

```bash
python gui/progress_gui.py --project_path PATH-TO-YOUR-PROJECT
```

Manual annotation and more detailed visualization:

```bash
python gui/trace_explorer.py --project_path PATH-TO-YOUR-PROJECT
```

# More details

See [detailed pipeline steps](docs/detailed_pipeline_steps.md)

See [detailed installation instructions](docs/installation_instructions.md)
