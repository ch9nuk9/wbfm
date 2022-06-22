# More details on each step

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
