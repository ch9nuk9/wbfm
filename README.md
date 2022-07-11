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

# BASIC: start a new project from nothing

## Preparation:

1. Red channel (tracking) in a single bigtiff file
2. Green channel (signal) in a single bigtiff file
3. 1 conda environment (see above for installation instructions, or use pre-installed versions on the cluster)

## Full workflow

### Creating a project

Working examples to create a project are available for 
[linux](wbfm/scripts/examples/0-create_new_project-linux-EXAMPLE.sh)
and [windows](wbfm/scripts/examples/0-create_new_project-windows-EXAMPLE.sh).
Everything can be run on the command line.

If your data is visible locally (mounted is okay), for the initial project creation you can use a gui:

```commandline
python gui/create_project_gui.py
```

#### *IMPORTANT*

The method of creating the project (gui or command line) will determine the style of filepaths that are saved in the project.
Thus, if you create and run the project on different operating systems, you will probably need to manually update the path to the raw data.

Example: Create the project on Windows, but run it on the cluster (Linux).

Specifically be aware of these variables:
```yaml
red_bigtiff_fname
green_bigtiff_fname
```

in the main project file: config.yaml

In addition, if creating from a windows computer, you may need to use dos2unix to fix any files that you want to execute, specifically those referenced below:
1. RUNME_*.sh
2. DRYRUN.sh

### Checklist of most important parameters to change

1. project_config.yaml
   1. start_volume_bigtiff
   2. num_frames
   3. num_slices (after flyback removal)
2. preprocessing_config.yaml
   1. raw_number_of_planes (before any removal)
   2. starting_plane (set 1 to remove a single plane)

For all other settings, the defaults should work well.

#### *IMPORTANT*
If you changed the name of your project, you must update it in the snakemake/config.yaml file under 'project_dir'


### Running the rest of the workflow

This code is designed in several different scripts, which can be running using a single command.
The organization between these steps uses the workflow manager [snakemake](https://snakemake.readthedocs.io/en/stable/).
Snakemake works by keeping track of output files with special names, and only reliably works for the first run.
If you are rerunning an old project, see the next section.

Once a project is made, the analysis can be run in the following way:
1. Activate the wbfm environment
2. cd to the /snakemake folder within the project:
```commandline
cd /path/to/your/project/snakemake
```
3. Do a dry run to catch any errors in initialization:
```bash
bash DRYRUN.sh
```
4. If there are errors, there are three possibilities:
   1. If this is not a new project, you might have to run steps one by one (see the next subsection).
   2. If you changed the name of the project, read the *IMPORTANT* tip above 
   3. If this is a new project, then it is probably a bug and you should file a GitHub issue and possibly talk to Charlie
5. Run the relevant RUNME script, either cluster or local. Probably, you want the cluster version:
```bash
bash RUNME_cluster.sh
```
6. Wait for this to finish. Depending on scheduling, it could take 12-48 hours.
7. Check the log files (they will be in the /snakemake folder) to make sure there were no errors.
Almost all errors will crash the program, but if you find one that doesn't, please file an issue!

# ADVANCED: start from an incomplete project

Snakemake works by keeping track of output files with special names, and only reliably works for the first run.
If you have simply not run all of the steps or there was a crash, then continue with the above section.
However, if you are re-running analysis steps, then see below.

In this case, you must run each step one by one.
Note that you can check the current status of the project by moving to the project and running a script. Example:
```bash
cd /path/to/your/project
python log/print_project_status.py
```

You can directly run the python scripts, or, most likely, run them using sbatch using the following syntax.

### Running single steps on the cluster (sbatch)

Once the project is created, each step can be run via sbatch using this command in the scripts/cluster folder:

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


# Summary of common problems

### Changing folder names
If you changed the name of your project, you must update it in the snakemake/config.yaml file under 'project_dir'

### Raw data paths on Windows vs. cluster
The method of creating the project will determine the style of filepaths that are saved in the project.
Thus, if you create and run the project on different operating systems, you will probably need to manually update the path to the raw data.

Specifically be aware of these variables:
```yaml
red_bigtiff_fname
green_bigtiff_fname
```

in the main project file: config.yaml

### Creating project from Windows

In addition, if creating from a windows computer, you may need to use dos2unix to fix any files that you want to execute, specifically those referenced below:
1. RUNME_*.sh
2. DRYRUN.sh


# More details

See [detailed pipeline steps](docs/detailed_pipeline_steps.md)

See [detailed installation instructions](docs/installation_instructions.md)
