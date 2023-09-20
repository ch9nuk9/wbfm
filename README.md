# Whole Brain Freely Moving Tracking and Trace extraction

This repository contains python code for analyzing raw volumetric images in two channels: red (tracking) and green (activity).

The segmentation portion of the algorithm is in a [sibling repository](https://github.com/Zimmer-lab/segmentation), but all main pipeline steps and GUIs are in this one.


# Installation for running

This project is designed to be installed with Anaconda, and requires two external local packages to be installed.

## If you just want to run the GUI

See [GUI README](wbfm/gui/README.md)


## On the cluster

If you just want to run the code (most people), then you can use the pre-installed environments installed on the cluster, which can be activated using:
```
conda activate /scratch/neurobiology/zimmer/.conda/envs/wbfm/
```

For more detail, see:
[detailed installation instructions](docs/installation_instructions.md)

# Running the pipeline

## Preparation

See expected folder structure [here](docs/data_folder_organization.md).

1. Red channel (tracking) in a single bigtiff file
2. Green channel (signal) in a single bigtiff file
3. 1 conda environment (see above for installation instructions, or use pre-installed versions on the cluster)

## Full pipeline

### Creating a single project (single recording)

Most people will create multiple projects at once (next section), but for a single recording, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

### Creating multiple projects

Each recording will generate one project, and you can use the create_projects_from_folder.sh script.

Important: the data folders must be organized in a specific way, see [here](docs/data_folder_organization.md).

```commandline
cd /path/to/this/code/wbfm/scripts/cluster
bash create_multiple_projects_from_data_parent_folder.sh -t /path/to/data/folder -p /path/to/projects/folder
```

As of September 2023, this is the proper path to this code on the cluster:
```commandline
cd /scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/cluster
```

For running projects, you will most likely want to run them all simultaneously instead of one-by-one.
See this [section](#running-the-rest-of-the-workflow-for-multiple-projects).

#### *IMPORTANT*

Creating and running the project on different operating systems will cause problems.
See [Summary of common problems](#summary-of-common-problems) for more details.

### Checklist of most important parameters to change or validate

You should check two things: the correct files were found, and the correct metadata (video length) was detected.

1. project_config.yaml
   1. bigtiff_start_volume
   2. num_frames
   3. num_slices (after flyback removal)
      1. This must be the same as 'raw_number_of_planes' minus 'starting_plane' below
      2. Note: This will be removed in a future release
2. preprocessing_config.yaml
   1. raw_number_of_planes (before any removal)
   2. starting_plane (set 1 to remove a single plane)
3. segment_config.yaml
   1. max_number_of_objects (increase for immobilized projects)

For all other settings, the defaults should work well.

#### *IMPORTANT*
If you changed the name of your project after creation or you changed operating systems, you must
update the 'project_dir' variable in the snakemake/config.yaml file.
This variable should be matched to the operating system you are running on, for example starting with 'S:' for windows or '/' for linux (cluster).

### Running the rest of the workflow for single project

Most people will run multiple projects at once (next section), but for a single recording, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

### Running the rest of the workflow for multiple projects

If you have many projects to run, you can use the run_all_projects_in_parent_folder.sh script.
This is especially useful if you created the projects using the create_multiple_projects_from_data_parent_folder.sh script.

```commandline
cd /path/to/this/code/wbfm/scripts/cluster
bash run_all_projects_in_parent_folder.sh -p /path/to/projects/folder
```

For more details on each step, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

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


### Other

There are many problems that can be recovered from by rerunning.
In theory, the snakemake pipeline is designed to be rerun from any step, and it should detect which steps are finished.
See [detailed pipeline steps](docs/detailed_pipeline_steps.md) to rerun individual steps or the full pipeline.


# More details

[Detailed pipeline steps](docs/detailed_pipeline_steps.md)

[Detailed installation instructions](docs/installation_instructions.md)

[Known issues](docs/known_issues.md)

[Folder organization](docs/data_folder_organization.md)

If you would like to contribute, see [how to contribute](docs/how_to_contribute.md)
