'''
 This script test cellpose parameters (pixels) with a given input by a shell script.
 Cellpose will run in 3D mode and only the masks are being saved (to save time; seq-file ~ 300mb/slice)
 It will use the test volume in /groups/zimmer/shared_projects/wbfm/cellpose_test_data/one_volume.tif

 Usage:
     cp_param_test_3D.py arg1 arg2

     arg1 = diameter in pixels; can be float(.1) or integer from 1-xx

     e.g. cp_param_test_3D.py 8

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
sv_base = "diam-" + str(arg1)
log_name = 'log_' + sv_base + '.log'

logging.basicConfig(filename=log_name, level=logging.DEBUG)
logging.info("START of python script")
print("START of python script")

# handling system arguments (=diameter)
# Args: (0 = script path)    1 = diameter in pixels
if len(sys.argv) == 2:
    diam = round(float(sys.argv[1]), 1)
    logging.info('INPUTS: diameter = %.2f' % (float(diam)))
else:
    logging.debug('INPUT ERROR! The inputs are %s' % (str(sys.argv)))

# saving base name
sv_base = "diam-" + str(arg1)

# original filepath
# will use a test set in my home folder to not overwrite Charlie's results

vol_path = '/users/niklas.khoss/stardist_test/preprocessed_volume.tif'
#'/users/niklas.khoss/cp_test/one_volume/one_volume.tif'


# run cellpose (3D) on a tif volume
# RUN CELLPOSE

# initializing cellpose model
model = models.Cellpose(gpu=False, model_type='nuclei')
channels = [0,0]

# load the volume and iterate over it calling cellpose on each slice separately
with tiff.TiffFile(vol_path) as vol:

    logging.info('Volume shape: %s', str(np.shape(vol.asarray())))
    img = vol.asarray()[1:]

    # running cellpose
    masks, flows, styles, diams = model.eval(img, diameter=diam,
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