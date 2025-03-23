# Whole Brain Freely Moving Tracking and Trace extraction

Contains code for the pipeline, GUI, and analysis code for the paper:
[An intrinsic neuronal manifold underlies brain-wide hierarchical organization of behavior in C. elegans](https://www.biorxiv.org/content/10.1101/2025.03.09.642241v1)

This repository contains python code for analyzing raw volumetric images in two channels: red (tracking) and green (activity).


# Installation for running

This project is designed to be installed with Anaconda, and requires two external local packages to be installed.
However, there are different use cases, some of which have easier installation steps.
Please check all sections below to determine which is best for you.

## If you just want to run the GUI

See [GUI README](wbfm/gui/README.md)


## On the cluster

If you just want to run the code (most people), then you can use the pre-installed environments installed on the cluster, which can be activated using:
```
conda activate /lisc/scratch/neurobiology/zimmer/.conda/envs/wbfm/
```

## Local installation

In principle this is rare, and only for developers or if you want to run the full pipeline on your local machine.
See: [detailed installation instructions](docs/installation_instructions.md)

# Running the pipeline

## Preparation

See expected folder structure [here](docs/data_folder_organization.md).

1. Red channel (tracking) as an ndtiff
2. Green channel (signal) as an ndtiff file
3. 1 conda environment (see above for installation instructions, or use pre-installed versions on the cluster)

Note that bigtiff for the raw data may work, but is deprecated.

## Full pipeline

### Creating a single project (single recording)

Most people will create multiple projects at once (next section), but for a single recording, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

### Creating multiple projects

Each recording will generate one project, and you can use the create_projects_from_folder.sh script.

Important: the data folders must be organized in a specific way, see [here](docs/data_folder_organization.md).

```commandline
bash /path/to/this/code/wbfm/scripts/cluster/create_multiple_projects_from_data_parent_folder.sh -t /path/to/parent/data/folder -p /path/to/projects/folder
```

Where "/path/to/parent/data/folder" contains subfolders with red, green, and (optionally) behavioral data, and "/path/to/projects/folder" is a folder where the new projects will be generated. 
In principle "/path/to/projects/folder" should be empty, but this is not necessary.

As of September 2023, this is the proper path to this code on the cluster:
```commandline
ls /lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/cluster
```

For running projects, you will most likely want to run them all simultaneously instead of one-by-one.
See this [section](#running-the-rest-of-the-workflow-for-multiple-projects).

#### *IMPORTANT*

Creating and running the project on different operating systems will cause problems.
See [Summary of common problems](#summary-of-common-problems) for more details.

### Checklist of most important parameters to change or validate

You should check that the correct files were found, and that the z-metadata is correct.

1. project_config.yaml
   1. red_fname
   2. green_fname
   3. behavior_fname (optional)
2. dat/preprocessing_config.yaml
   1. raw_number_of_planes (before any removal)
   2. starting_plane (set 1 to remove a single plane)
3. segment_config.yaml
   1. max_number_of_objects (increase for immobilized projects)

For all other settings, the defaults should work well.


### Running the rest of the workflow for single project

Most people will run multiple projects at once (next section), but for a single recording, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

### Running the rest of the workflow for multiple projects

If you have many projects to run, you can use the run_all_projects_in_parent_folder.sh script.
This is especially useful if you created the projects using the create_multiple_projects_from_data_parent_folder.sh script.

```commandline
bash /path/to/this/code/wbfm/scripts/cluster/run_all_projects_in_parent_folder.sh -t /path/to/projects/folder
```

Note that you should run this script from a directory where you have permission to create files.
Otherwise the log files will be created in the directory where you ran the script, which will crash silently due to permission errors.

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

### Jobs are PENDING for a long time on the cluster

Sometimes the cluster is very busy, and we have single large jobs.
This can cause jobs to be pending for a long time, which is not a fundamental problem but can cause delays.
You can check the status of your jobs using the squeue command.
Example:
```commandline
squeue -u <your_username>
```

If you have many jobs pending, you can change them to the himem partition, which is often empty.
BUT you should check this! 
You can look at the main login screen when you ssh to the cluster to see the status of the partitions.
Alternatively you can use the sinfo command to see the status of the partitions.
```commandline
sinfo -p himem
```

If any nodes are idle, your jobs should immediately start running if you switch.
To switch, you can use the following command:
```commandline
bash /path/to/this/code/wbfm/scripts/cluster/move_pending_slurm_jobs_to_himem_partition.sh
```

Currently this script is located here:
```commandline
/lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/cluster/move_pending_slurm_jobs_to_himem_partition.sh
```

NOTE: this will change ALL pending jobs to the himem partition, so be careful if you are running other jobs.

### Raw data paths on Windows vs. cluster
The method of creating the project will determine the style of filepaths that are saved in the project.
Thus, if you create and run the project on different operating systems, you will probably need to manually update the path to the raw data.

Specifically be aware of these variables:
```yaml
red__fname
green_fname
```

in the main project file: config.yaml

### Moving location of raw data

This will not affect a completed project, but will cause problems if you try to rerun the pipeline.
This includes if you already ran the trace extraction steps, but then want to run the behavior analysis.
Make sure the the paths to raw data are correct in the project_config.yaml file, specifically:
```yaml
parent_data_folder
behavior_fname (optional)
red_fname
green_fname
```


### Creating project from Windows

In addition, if creating from a windows computer, you may need to use dos2unix to fix any files that you want to execute, specifically:
1. RUNME.sh

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
