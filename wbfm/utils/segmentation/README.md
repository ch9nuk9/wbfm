## IOUs

### Overview

Comparing two sets of masks.

1. Load the ground truth (GT) and cellpose (CP) algorithm output as 2d or 3d np.arrays()
2. Get all unique cells in GT
3. Loop through GT cells:
    1. Element-wise multiplication between this GT cell and full CP array
    2. Count the overlap
    3. Check if this is the largest overlap, and save it if it is
    4. Output of matches is like: [[0, 2]; [1, 15]; [2, 0] ...]
4. Divide each "best" overlap by the union (now that we have the same ID numbers in GT and CP)


### TODO

4. Get all IOUs across parameters/folders and save
    1. Refactor as a function
    2. Make a new bash script, that calls this script to be used on commandline. BASH job array!
    3. Core function working on one folder! Extra script iterates over subfolders containing the cellpose results

5. Plot
    1. What do we want to plot?
    2. How long do cellpose runs take on the cluster?
    3. Plot cellpose 2D/3D and STARDIST results together

6. Refactor to use Pandas dataframes
    1. After labmeeting > February

7. STARDIST
    1. Installation on cluster
    2. First test on volume
    3. Parameter & model testing
        1. First, the models
        2. Second, the parameters (n_rays, grid, nms, probability)
    4. Quantify


### DONE:

1. Divide matches (intersections) by unions to get full IoUs
2. Get all IOUs across planes
3. Save the IOUs