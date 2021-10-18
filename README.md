# Installation

This project is designed to be installed with Anaconda, and requires one external local package to be installed.
The following are step-by-step instructions.

### Install Anaconda

https://www.anaconda.com/products/individual

### Get the code

Download or clone two repositories, both on the Zimmer group GitHub:
1. DLC_for_WBFM (this repo)
2. Segmentation

### Install the environments

For this project, two separate environments are currently needed.
This is due to versioning interactions between opencv and tensorflow within deeplabcut and stardist. 
Both environments can be found in the folder conda-environments/

1. segmentation.yaml
2. DLC-for-WBFM.yaml OR DLC-for-WBFM-cluster.yaml, depending on the target machine

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
3. 2 conda environments (see main README for installation instructions)

## Full workflow with commands and approximate times

Currently, everything is designed to run on the command line.

All examples assume that you are in the main folder:

DLC_for_WBFM/

For simple reference, each command is in proper number order in DLC_for_WBFM/scripts

### Initializing

Perparation: see above

Command:
```bash
RED_PATH="D:\More-stabilized-wbfm\test2020-10-22_16-15-20_test4-channel-0-pco_camera1\test2020-10-22_16-15-20_test4-channel-0-pco_camera1bigtiff.btf"
GREEN_PATH="D:\More-stabilized-wbfm\test2020-10-22_16-15-20_test4-channel-1-pco_camera2\test2020-10-22_16-15-20_test4-channel-1-pco_camera2bigtiff.btf"
COMMAND="scripts/create_new_project.py"

python $COMMAND with project_dir=./scratch red_bigtiff_fname=$RED_PATH green_bigtiff_fname=$GREEN_PATH
```

This is the most complicated step, and I recommend that you create a bash script.
An example can be found at:

/DLC_for_WBFM/scripts/0-create_new_project-windows-EXAMPLE.sh

Speed: Fast

Output: new project folder with project_config.yaml, and with 4 numbered subfolders

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

Speed: Slowest step; 1 day - 1 week

Output, in 1-segmentation:
1. masks_N.zarr
..* .zarr is the data type, similar to btf but faster
..* N is the number of frames, as set in the project_config
2. metadata_N.pickle
..* This is the centroids and brightnesses of the identified neurons

### Training data

Preparation:
1. Open 2-training_data/preprocessing_config.yaml and change the variables marked CHANGE ME
2. For now, 2-training_data/training_data_config.yaml should not need to be updated

Command:
```bash
python scripts/2a-make_short_tracklets.py with project_path=PATH-TO-YOUR-PROJECT
```

Speed: Fast, but depends on number of frames; 1-2 hours

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