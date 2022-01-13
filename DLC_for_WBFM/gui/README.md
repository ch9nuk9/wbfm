# GUIs for Whole Brain Freely Moving

## Starting out: progress_gui

All GUIs are designed to be accessed from the "progress gui".

This can be accessed by running "progress_gui.py", for example:

```commandline
python progress_gui.py --project_path ~/dlc_stacks/worm3-gui_demo/project_config.yaml
```

This command will bring a small GUI up that displays the status of the target project, with buttons to open more complex GUIs.
Those will be described in the next major section.

This command works with 3 assumptions:
1. You are in a terminal in this folder
2. You are in the proper conda environment
3. You have initialized a project at ~/dlc_stacks/worm3-gui_demo/project_config.yaml

Instructions to satisfy these assumptions are in the next sections.

### Terminal setup

If you are on linux or mac, a terminal is included. On Windows I suggest git bash

### Installation

#### Step 1
In the folder conda-environments, there is a specific environment for this purpose: "gui_only.yaml"

If you are in a terminal, you can use this to create the proper conda environment:

```commandline
conda create -f gui_only.yaml
```

#### Step 2
After the overall packages are installed, the zimmer group private packages need to be installed:

1. git clone dlc_for_wbfm and segmentation (from https://github.com/Zimmer-lab)
2. run ```pip install -e .``` in each main folder

### Project initialization

See the main README file for instructions, or use a pre-generated project

## More complex GUI: tracklet and segmentation correction

First, open the progress gui as described above.

Then, the most complex gui can be accessed using the "tracesVis" button.

This opens a new Napari window with several layers, designed to be used to view and modify the tracking and segmentation.

### Overall explanation

This GUI displays information at 2 scales:
1. An overview of the entire worm body
2. More detail about a specific neuron

In addition, there are 4 areas with different information:
1. Left - Napari layers
2. Center - Main 3d data
3. Right - Menus and buttons
4. Bottom - Matplotlib graph

First I will explain the Napari layers:

#### Top level - entire body

The following layers below to this level:
1. Red data - Raw mscarlet layer
2. Green data - Raw gcamp layer
3. Neuron IDs - Numbers displayed on top of the neurons
4. Colored segmentation - Segmentation as colored by tracks. This is a subset of the next layer 
5. Raw segmentation - Original segmentation, before tracking. Note that this layer can be interactive

#### Detailed level - Single neuron

The following layers below to this level, and are related to the currently selected neuron:
1. track_of_point - a single point on top of the currently selected neuron
2. final_track - a line showing the current and past positions of the neuron

In addition, if the user uses interactions to click on certain neurons, then more layers will be added that relate to the clicked neuron.

### Explanation of interactivity

By default, interactivity is off, and must be turned on with the checkbox in the top right.

There are two types of interactivity:
1. Tracklets 
2. Segmentation

#### Tracklet workflow

This is designed to correct the tracklets associated with neurons.
The basic steps are as follows:

1. Select a neuron under "Neuron selection"
2. Change to "tracklets" mode under "Channel and Mode selection"
   1. The graph at the bottom will display all tracklets that belong to the selected neuron
3. Find a problem (gap in tracking or error in signal)
   1. Use the plot at the bottom
4. Navigate to the time point with the problem
   1. Many shortcuts are provided
   2. Note that if the neuron is tracked, the main view will be centered on that neuron
5. Fix the problem (described in next section)
6. Save the current tracklet
7. Find a new problem, and repeat 4-6
8. When the neuron is fully tracked, save the manual annotations to disk
   1. NOTE: this will take some time (~10-20 seconds)
9. Choose a new neuron, and repeat 3-8

#### Categories of tracklet problems

Basically there are two types of problems, with two solutions:

1. A perfect tracklet was not assigned to the neuron
2. A tracklet had a mistake and needs to be split
3. A completely incorrect tracklet was assigned to the neuron

Use the following basic workflow to fix them:
1. With the "Raw segmentation layer" highlighted, click on a neuron
   1. This will load the tracklet associated with that segmentation
   2. In addition, a new Napari layer will appear showing the position of the tracklet across time
   3. NOTE: for the 3rd case, you are trying to click on the incorrect neuron, to select the incorrect tracklet for removal
2. Check the correctness of the tracklet
3. Case 1: the tracklet is perfect
   1. Use the button to save the current tracklet to the current neuron, and continue
   2. If there is a conflict, see below
4. Case 2: the tracklet jumps between neurons
   1. Use the "Split tracklet" buttons to remove the jumps
   2. When it is correct, save it to the current neuron
   3. If there is a conflict, see below
5. Case 3: the tracklet is added, but shouldn't be
   1. Use the "remove tracklet from all neurons" button
   2. No additional saving is needed

Note that many tracklet problems are in fact due to segmentation problems, described below.

#### Fixing tracklet conflicts

Currently, the gui will not allow you to save a tracklet if there are time points that overlap with other tracklets.
Thus, you must either a) shorten the current tracklet to fit, or b) remove conflicting tracklets.
Use the shortcut and buttons to do so, then save.

#### Segmentation workflow

This is designed to correct segmentation, and does not need to be related to the current neuron.
However, to keep track of which segmentations you have corrected, it makes sense to correct each neuron across time before moving to the next one.
In that case, the following workflow is suggested:

1. While correcting a neuron, find a segmentation problem
2. Control-click to attempt an automatic segmentation
   1. This will create a new window showing information about the splitting algorithm
3. Case 1: the automatic segmentation is good
   1. Simply save the candidate mask to RAM (button)
4. Case 2: the automatic segmentation is bad
   1. Using the pop-up or other information, choose where the neuron should actually be segmented
   2. Then alt-click to apply the manual segmentation
   3. Check that it is correct, and save to RAM (button)
5. As often as possible, Save to disk
   1. Note that this can take some time (~20 seconds)
   
### Known issues

See the GUI label on the main github repository 