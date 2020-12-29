## IOUs

### Overview

Comparing two sets of masks.

1. Load the ground truth (GT) and cellpose (CP) algorithm output as 2d or 3d np.arrays()
2. Get all unique cells in GT
3. Loop through GT cells:
--1. Element-wise multiplication between this GT cell and full CP array
--2. Count the overlap
--3. Check if this is the largest overlap, and save it if it is
--4. Output of matches is like: [[0, 2]; [1, 15]; [2, 0] ...]
4. Divide each "best" overlap by the union (now that we have the same ID numbers in GT and CP)


### TODO

1. Divide matches (intersections) by unions to get full IOUs
2. Get all IOUs across planes
3. Save the IOUs
4. Get all IOUs across parameters and save
5. Plot
