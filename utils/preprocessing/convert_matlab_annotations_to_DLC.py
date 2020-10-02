import h5py
import os
import pandas as pd
import glob
import numpy as np
from pathlib import Path
from deeplabcut.utils import auxiliaryfunctions

def wb_tracker2dlc_format(path_config_file):
    """
    Converts Zimmer Whole-Brain tracker data for immobilized worms into training data for DeepLabCut
    """

    # Build filenames
    home = os.path.dirname(path_config_file)
    wb_fname = os.path.join(home,'wbstruct.mat')

    config_file = Path(path_config_file).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")

    # Note: the labeled-data subfolder has the entire video name as the folder name
    video_fname = [i for i in cfg['video_sets'].keys()][0] # Assume one video for now
    fname = Path(video_fname)
    output_path = os.path.join(Path(path_config_file).parents[0],'labeled-data',fname.stem)
    output_path = output_path
    print('Looking in folder {}'.format(output_path))

    # Get list of images
    imlist=[]
    imtype = '*.tif'
    imlist.extend([fn for fn in glob.glob(os.path.join(output_path,imtype)) if ('labeled.png' not in fn)])

    if len(imlist)==0:
        print("No images found!!")

    index = np.sort(imlist)
    print('Working on folder: {}'.format(os.path.split(str(output_path))[-1]))
    relativeimagenames=['labeled'+n.split('labeled')[1] for n in index]

    # Build dataset using pandas; copied from from labeling_toolbox.py
    scorer = cfg['scorer']
    with h5py.File(wb_fname, 'r') as mat:
        num_neurons = int(mat['simple']['nn'][0][0])
        x = mat['simple']['x'][:,0] # NOTE: a flip is required here; not sure why
        y = mat['simple']['y'][:,0]
        z = mat['simple']['z'][:]
        coords = np.empty((len(index),3,))
        dataFrame = None

        for i in range(num_neurons):
            bodypart = 'neuron{}'.format(i)

            # Note: this requires a flip in the y direction; not sure why
            x_sz = 272
            coords[:] = np.array([x_sz-x[i], y[i], z[i][0]])

            index = pd.MultiIndex.from_product([[scorer], [bodypart], ['x', 'y', 'z']],names=['scorer', 'bodyparts', 'coords'])
            #print(index)
            #print(coords)
            frame = pd.DataFrame(coords, columns = index, index = relativeimagenames)
            dataFrame = pd.concat([dataFrame, frame],axis=1)

    dataFrame.to_csv(os.path.join(output_path,"CollectedData_" + scorer + ".csv"))
    dataFrame.to_hdf(os.path.join(output_path,"CollectedData_" + scorer + '.h5'),'df_with_missing',format='table', mode='w')

    print('Finished')
