# Whole Brain Freely Moving Tracking and Trace extraction

This repository contains python code for analyzing raw volumetric images in two channels: red (tracking) and green (activity).

The segmentation portion of the algorithm is in a sibling repository, but all GUIs are in this one:
https://github.com/Zimmer-lab/segmentation

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

# Installation for developing

### Install Anaconda

https://www.anaconda.com/products/individual

There are additional details in the computational protocols repository:
https://github.com/Zimmer-lab/protocols/tree/master/computational/zimmer_lab_code_pipeline

### Get the code

Download or clone these repositories:
1. DLC_for_WBFM (this repo): https://github.com/Zimmer-lab/dlc_for_wbfm
2. Segmentation: https://github.com/Zimmer-lab/segmentation
3. fDNC (leifer paper): https://github.com/Charles-Fieseler-Vienna/fDNC_Neuron_ID

### Install the environments

#### Pre-installed environments

Note: there are pre-installed environments living on the cluster, at:
/scratch/zimmer/.conda/envs/segmentation
/scratch/zimmer/.conda/envs/wbfm

They can be activated using:
```commandline
conda activate /scratch/zimmer/.conda/envs/wbfm
```

#### Installing new environments

Note: if you are just using the GUI, then you can use a simplified environment.
Detailed instructions can be found in the [README](DLC_for_WBFM/gui/README.md) file under the gui section

For this project, two separate environments are currently needed.
This is due to versioning interactions between opencv and tensorflow within deeplabcut and stardist. 
Both environments can be found in the folder conda-environments/

1. segmentation.yaml
2. wbfm.yaml

This installs the public packages, now we need to install our custom code.
Do `conda activate segmentation` and install the local code in the following way:

1. cd to the segmentation repository
2. run: `pip install -e .`
3. Repeat steps 1-2 for the other repository, DLC_for_WBFM
4. Repeat steps 1-3 for the other environment

#### Summary of installations

You will install 4 things: 2 environments and 3 custom packages

# Running a project: start from nothing

## Preparation:

1. Red channel (tracking) in a single bigtiff file
2. Green channel (signal) in a single bigtiff file
3. 2 conda environments (see above for installation instructions)

## Full workflow with commands and approximate times

All examples assume that you are in the main folder:

DLC_for_WBFM/

For simple reference, each command is in proper number order in DLC_for_WBFM/scripts

### Creating a project: easy

Working examples to create a project are available for 
[linux](DLC_for_WBFM/scripts/examples/0-create_new_project-linux-EXAMPLE.sh)
and [windows](DLC_for_WBFM/scripts/examples/0-create_new_project-windows-EXAMPLE.sh)

Currently, everything is designed to run on the command line.
However, if you have your data available locally, for the initial project creation you can use a gui:

```commandline
python gui/create_project_gui.py
```

### Creating a project: command-line details

Command:
```bash
RED_PATH="path/to/red/data"
GREEN_PATH="path/to/green/data"
PROJECT_DIR="path/to/new/project/location"

COMMAND="scripts/0a-create_new_project.py"

python $COMMAND with project_dir=$PROJECT_DIR red_bigtiff_fname=$RED_PATH green_bigtiff_fname=$GREEN_PATH
```

This is the most complicated step, and I recommend that you create a bash script.

Speed: Fast

Output: new project folder with project_config.yaml, and with 4 numbered subfolders

## Running full workflow (snakemake)

This code is designed in several different scripts, which can be running using two commands.
The organization between these steps uses the workflow manager [snakemake](https://snakemake.readthedocs.io/en/stable/).
Snakemake works by keeping track of output files with special names, and only reliably works for the first run.
If you are rerunning an old project, see the next section.

Once a project is made, the analysis can be run in the following way:
1. Activate the segmentation environment
2. cd to the /snakemake folder within the project
3. Do a dry run to catch any errors in initialization:
```bash
bash DRYRUN_segmentation.sh
```
4. If there are errors, see the next subsection
5. Run the relevant RUNME script, either cluster or local. Probably, you want the cluster version:
```bash
bash RUNME_cluster_segmentation.sh
```
6. Wait for this to finish. Depending on scheduling, it could take 4-12 hours.
7. Carefully check the log files (they will be in the /snakemake folder) to ensure success
8. Repeat steps 3-7 for the *_post_segmentation.sh DRYRUN and RUNME scripts

# Running a project: start from a previous project

TODO

## Running single steps on the cluster (sbatch)

Once the project is created, each step can be run via sbatch using:

```commandline
sbatch single_step_dispatcher.sbatch -s 1 -t /scratch/zimmer/Charles/dlc_stacks/worm10-gui_test/project_config.yaml
```

# More details on each step

### Segmentation

Preparation:
0. Make sure the project was initialized successfully!
1. Open project_config.yaml and change the variables marked CHANGE ME
..* In particular, manually check the video quality to make sure start_volume and num_frames are reasonable
..* For now, num_slices must be set manually
4. Open 1-segmentation/preprocessing_config.yaml and change the variables marked CHANGE ME
5. For now, 1-segmentation/segmentation_config.yaml should not need to be updated

Command:
```bash
python scripts/1-segment_video.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Slowest step; 3-12 hours

Output, in 1-segmentation:
1. masks.zarr
..* .zarr is the data type, similar to btf but faster
2. metadata.pickle
..* This is the centroids and brightnesses of the identified neurons

### Training data

Preparation:
1. Open 2-training_data/preprocessing_config.yaml and change the variables marked CHANGE ME
2. For now, 2-training_data/training_data_config.yaml should not need to be updated

Command:
```bash
python scripts/2ab-build_feature_and_match.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Fast, but depends on number of frames; 2-6 hours

Output, in 2-training_data/raw:
1. clust_dat_df.pickle
..* This is the dataframe that contains the partial "tracklets", i.e. tracked neurons in time
2. match_dat.pickle and frame_dat.pickle
..* These contain the matches between frames, and the frame objects themselves
..* This is mostly for debugging and visualizing, and is not used further



### Tracking

#### Part 1/2

Preparation:
1. Open 3-tracking/tracking_config.yaml and change the variables marked CHANGE ME

Command:
```bash
python scripts/3a-track_using_superglue.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: 1-3 hours

Output, in 3-tracking:
1. A dataframe with positions for all neurons

#### Part 2/2

Combine the tracks and tracklets

Command:
```bash
python scripts/3b-match_tracklets_and_tracks_using_neuron_initialization.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Long; ~6 hours

Output:
1. A dataframe with positions for each neuron, corrected by the tracklets

### Traces

Preparation: None, for now

Command:
```bash
python scripts/4-make_final_traces.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: ~30 minutes

Output, in 4-traces:
1. all_matches.pickle, the matches between the DLC tracking and the original segmentation
2. red_traces.h5 and green_traces.h5
..* This is the raw time series for all neurons in each channel
   
   
## Visualization of results

I made a gui for this purpose, with everything in the gui/ folder. Read the README in that folder for more detail

Command:
```bash
python gui/trace_explorer.py --project_path PATH-TO-YOUR-PROJECT
```

# Summary of GUIs

Initial creation of project:
```bash
python gui/create_project_gui.py --project_path PATH-TO-YOUR-PROJECT
```

Visualization of all steps in the analysis is also possible, and they can be accessed via the progress gui:

```bash
python gui/progress_gui.py --project_path PATH-TO-YOUR-PROJECT
```

Manual annotation and more detailed visualization:

```bash
python gui/trace_explorer.py --project_path PATH-TO-YOUR-PROJECT
```