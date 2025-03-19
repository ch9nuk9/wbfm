'''
 This script test cellpose parameters (pixels and flow_threshold) with a given input by a shell script.
 Cellpose will run in 3D mode and only the masks are being saved (to save time; seq-file ~ 300mb/slice)
 It will use the test volume in /groups/zimmer/shared_projects/wbfm/cellpose_test_data/one_volume.tif
 
 Usage:
     cp_param_test_3D.py arg1 arg2
     
     arg1 = diameter in pixels; can be float(.1) or integer from 1-xx
     arg2 = flow_threshold; float(.2) from 0-1
     
     e.g. cp_param_test_3D.py 8 0.5
     
'''


# import section
import os,sys,timeit
import numpy as np
import time, os, sys
import tifffile as tiff
from cellpose import utils, io, models
import logging
import pickle

# logging
arg1 = round(float(sys.argv[1]) * 10)
arg2 = round(float(sys.argv[2]) * 100)
sv_base = "diam-" + str(arg1/10) + "_flow-" + str(arg2)
log_name = 'log_' + sv_base + '.log'

logging.basicConfig(filename=log_name, level=logging.DEBUG)
logging.info("START of python script")
print("START of python script")

# handling system arguments (=diameter and flow_threshold input)
# Args: (0 = script path)    1 = diameter in pixels     2 = flow_threshold 
if len(sys.argv) == 3:
    diam = round(float(sys.argv[1]), 1)
    flow_thr = round(float(sys.argv[2]), 2)
    logging.info('INPUTS: diameter = %.2f, flow threshold = %.2f' % (float(diam), float(flow_thr)))
else:
    logging.debug('INPUT ERROR! The inputs are %s' % (str(sys.argv)))
    



# saving base name
sv_base = "diam-" + str(arg1) + "_flow-" + str(arg2)

# original filepath 
# will use a test set in my home folder to not overwrite Charlie's results

vol_path = '/groups/zimmer/shared_projects/wbfm/cellpose_test_data/one_volume.tif'
#'/users/niklas.khoss/cp_test/one_volume/one_volume.tif'


# run cellpose (3D) on a tif volume 
# RUN CELLPOSE

# initializing cellpose model
model = models.Cellpose(gpu=False, model_type='nuclei')
channels = [0,0]

# load the volume and iterate over it calling cellpose on each slice separately
with tiff.TiffFile(vol_path) as vol:
    
    logging.info('Volume shape: %s', str(np.shape(vol.asarray())))
    img = vol.asarray()

    # running cellpose
    masks, flows, styles, diams = model.eval(img, diameter=diam, 
                                     flow_threshold=flow_thr,
                                     channels=channels, 
                                     net_avg=True, do_3D=True)

    # saving output
    print('-> saving output')
    png_sv_name = os.path.join(os.getcwd(),
                               "masks_3D_" + '_' + sv_base  )
    # save as tif
    tiff.imsave(png_sv_name, masks)

    # pickling the mask output
    sv_name = "masks_arrays_3D_" + '_' + sv_base + ".pickle"
    pickle.dump( masks, open( sv_name, "wb" ))

    np_sv = "np_masks_3D_" + "_" + sv_base
    np.save( np_sv, masks )
        
logging.info("--- Done! ---")