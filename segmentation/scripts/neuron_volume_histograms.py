import segmentation.util.overlap as ol
import os
import numpy as np
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

# create list of cp results
cp_path = r'C:\Segmentation_working_area\stitched_3d_data'
sv_path = r'C:\Segmentation_working_area\cellpose_testdata'
files = [os.path.join(cp_path, f.name) for f in os.scandir(cp_path) if f.is_file()]