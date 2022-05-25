# Whole Brain Freely Moving Tracking and Trace extraction

This repository contains python code for analyzing raw volumetric images in two channels: red (tracking) and green (activity).

The segmentation portion of the algorithm is in a sibling repository, but all GUIs are in this one:
https://github.com/Zimmer-lab/segmentation

# Installation

This project is designed to be installed with Anaconda, and requires one external local package to be installed.
The following are step-by-step instructions.

### Install Anaconda

https://www.anaconda.com/products/individual

### Get the code

Download or clone two repositories, both on the Zimmer group GitHub:
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

In total, you will install 4 things: 2 environments and 2 custom packages

# Running a project

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
An example can be found at:

Speed: Fast

Output: new project folder with project_config.yaml, and with 4 numbered subfolders

## Running each step on the cluster (sbatch)

Once the project is created, each step can be run via sbatch using:

```commandline
sbatch single_step_dispatcher.sbatch -s 1 -t /scratch/zimmer/Charles/dlc_stacks/worm10-gui_test/project_config.yaml
```


## More details on each step
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



### DeepLabCut

Note: this step is being phased out

#### Part 1/3

Preparation:
1. Open 3-tracking/tracking_config.yaml and change the variables marked CHANGE ME

Command:
```bash
python scripts/3a-initialize_dlc_stack.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: fast, but depends on number of frames and stacks; 10-30 minutes

Output, in 3-tracking:
1. A DeepLabCut folder for each of the 2d tracking networks
..* Within these folders, training data will be extracted as .pngs
..* In the labeled-data subfolder you can see the produced training data
   
#### Part 2/3

Command:
```bash
python scripts/3b-train_all_dlc_networks.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: LONG, but depends on network convergence and number of stacks; 1-2 days

Output, in each DLC project folder:
1. A trained network

NOTE: can be run on the cluster; see scripts/cluster/3b-train_all_dlc_networks_array.sh.
If using the cluster, the training time is reduced to ~12 hours

#### Part 3/3

Command:
```bash
python scripts/3c-make_full_tracks.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Fast if using GPU; <1 hour

Output, in 3-tracking:
1. full_3d_tracks.h5, a dataframe of the tracks combined into 3d from each 2d network

NOTE: can be run on the cluster; see scripts/cluster/3c-make-full_tracks.sbatch

### Traces

Preparation: None, for now

Command:
```bash
python scripts/4a-match_tracks_and_segmentation.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Medium; 1-3 hours

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

Visualization of other steps in the analysis is also possible, and they can be accessed via the progress gui:

Command:
```bash
python gui/progress_gui.py --project_path PATH-TO-YOUR-PROJECT
```