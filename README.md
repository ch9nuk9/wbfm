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

For more detail, see:
[detailed installation instructions](docs/installation_instructions.md)


# Recommended: start a new project from nothing

## Preparation:

1. Red channel (tracking) in a single bigtiff file
2. Green channel (signal) in a single bigtiff file
3. 1 conda environment (see above for installation instructions, or use pre-installed versions on the cluster)

## Full workflow

### Creating a single project (single recording)

Working examples to create a project are available for 
[linux](wbfm/scripts/examples/0-create_new_project-linux-EXAMPLE.sh)
and [windows](wbfm/scripts/examples/0-create_new_project-windows-EXAMPLE.sh).
Recommended: run these commands on the command line.

If your data is visible locally (mounted is okay), for the initial project creation you can use a gui:

```commandline
cd /path/to/this/code/wbfm/gui
python create_project_gui.py
```

### Creating multiple projects

Note that each recording will generate one project.

If you have many projects to create, you can use the create_projects_from_folder.sh script.
This script will create a project for each data folder in a given directory.

```commandline
cd /path/to/this/code/wbfm/scripts/cluster
bash create_multiple_projects_from_data_parent_folder.sh -t /path/to/data/folder -p /path/to/projects/folder
```

For running projects, you will most likely want to run them all simultaneously instead of one-by-one.
See this [section](#running-the-rest-of-the-workflow-for-multiple-projects).

#### *IMPORTANT*

The method of creating the project (gui or command line) will determine the style of filepaths that are saved in the project.
Thus, if you create and run the project on different operating systems, you will probably need to manually update the path to the raw data.

Example: Create the project on Windows, but run it on the cluster (Linux).

Specifically be aware of these variables in the main project file (config.yaml):
```yaml
red_bigtiff_fname
green_bigtiff_fname
```

AND the subfolder file: snakemake/config.yaml
```yaml
project_dir
```

In addition, if creating from a windows computer, you may need to use the command line script dos2unix to fix any files that you want to execute, specifically those referenced below:
1. RUNME_*.sh
2. DRYRUN.sh

Finally, due to changes in the cluster you may get a permission error when importing skimage.
This should be fixable with one line; check the file RUNME_cluster.sh for more information.

### Checklist of most important parameters to change or validate

You should check two things: the correct files were found, and the correct metadata (video length) was detected.

1. project_config.yaml
   1. bigtiff_start_volume
   2. num_frames
   3. num_slices (after flyback removal)
2. preprocessing_config.yaml
   1. raw_number_of_planes (before any removal)
   2. starting_plane (set 1 to remove a single plane)

For all other settings, the defaults should work well.

#### *IMPORTANT*
If you changed the name of your project or you changed operating systems, you must
update the 'project_dir' variable in the snakemake/config.yaml file.
This variable should be matched to the operating system you are running on, for example starting with 'S:' for windows or '/' for linux (cluster).

### Running the rest of the workflow for single project

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
4. If there are errors, there are three easy possibilities:
   1. If this is not a new project, you might have to run steps one by one (see the next subsection).
   2. If you changed the name of the project, read the *IMPORTANT* tip above
   3. If you get a permission issue, read the *IMPORTANT* tip above
   4. If you still have a problem, then it is probably a bug and you should file a GitHub issue and possibly talk to Charlie
5. Run the relevant RUNME script, either cluster or local. Probably, you want the cluster version:
```bash
bash RUNME_cluster.sh
```
6. This will run ALL steps in sequence.
   1. Note: you can't close the terminal! You may need to use a long-term terminal program like tmux or screen.
   2. If you get errors, see step 4.
   3. It will print a lot of green-colored text, and the current analysis step will be written near the top, for example: 'rule: preprocessing'. Depending on how busy the cluster is, it could take 6-24 hours / 1000 volumes.
7. Check the log files (they will be in the /snakemake folder) to make sure there were no errors.
Almost all errors will crash the program (display a lot of red text), but if you find one that doesn't, please file an issue!
8. If the program crashes and you fix the problem, then you should be able to start again from step 3 (DRYRUN). This should rerun only the steps that failed and after, not the steps that succeeded. 

### Running the rest of the workflow for multiple projects

If you have many projects to run, you can use the run_all_projects_in_parent_folder.sh script.
This is especially useful if you created the projects using the create_multiple_projects_from_data_parent_folder.sh script.

```commandline
cd /path/to/this/code/wbfm/scripts/cluster
bash run_all_projects_in_parent_folder.sh -p /path/to/projects/folder
```

This will perform all of the steps in the above section automatically.

#### Check ongoing progress

There are three ways to check progress:
1. Check the currently running jobs
2. Check the log files in the snakemake/ subfolder
3. Check the produced analysis files using a gui

Method 1:
Run this command on the cluster:
```commandline
squeue -u <your_username>
```

Method 2:
Use the tail command to check the log files:
```commandline
tail -f /path/to/your/project/snakemake/log/[MOST_RECENT_LOG].log
```

Note that the -f flag will keep the terminal open and update the log file as it changes.

Method 3:
Use the progress_gui.py gui on a local machine with mounted data to check the actual images, segmentation, and tracking produced.
```commandline
cd /path/to/this/code/wbfm/gui
python progress_gui.py -p /path/to/your/project
```

### Manual annotation and rerunning

Tracking can be checked and corrected using the main trace_explorer [GUI](#Summary of GUIs).
However, this does not automatically regenerate the final trace dataframes.
For this, some steps must be rerun.
The steps are the same as running steps within an incomplete project, but in short:

1. Run step 3b
   - Note that setting only_use_previous_matches=True in tracking_config.yaml is suggested, and will speed the process dramatically
2. Possible: run step 1-alt (rebuild the segmentation metadata)
   - This is only necessary if segmentation was changed in the manual annotation step 
2. Run step 4

All steps can be run using multi_step_dispatcher.sh
Example usage is given within that file.

Note that it is possible to use snakemake to "know" which files need to be updated, if any files were copied to/from a local machine, snakemake will become confused and attempt to rerun the entire pipeline.

### Advanced: running steps within an incomplete project (including if it crashed)

See [detailed pipeline steps](docs/detailed_pipeline_steps.md)

# Summary of GUIs

All guis are in the folder: /folder_of_this_README/wbfm/gui/example.py

1. Initial creation of project. 
See sections above on carefully checking paths:
```bash
python wbfm/gui/create_project_gui.py
```

2. Visualization of most steps in the analysis is also possible, and they can be accessed via the progress gui. This also tells you which steps are completed:
```bash
python wbfm/gui/progress_gui.py
```
Or, if you know the project already:
```bash
python wbfm/gui/progress_gui.py --project_path PATH-TO-YOUR-PROJECT
```

3. Manual annotation and more detailed visualization. 
Note, this can take minutes to load:

```bash
python wbfm/gui/trace_explorer.py --project_path PATH-TO-YOUR-PROJECT
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

### Errors when running GUI

Some versions of napari on some versions of python on windows give this error:
UnicodeDecodeError

This can be fixed by running everything with an extra flag. Instead of:

```python /path/to/code.py```

Do this:

```python -X utf8 /path/to/code.py```


# More details

See [detailed pipeline steps](docs/detailed_pipeline_steps.md)

See [detailed installation instructions](docs/installation_instructions.md)

See [known issues](docs/known_issues.md)

If you would like to contribute, see [how to contribute](docs/how_to_contribute.md)
