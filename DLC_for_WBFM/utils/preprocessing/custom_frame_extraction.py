#
# Basically this is a custom extract_frames() function
#

import os
import platform
from pathlib import Path
import tifffile
from deeplabcut.utils import auxiliaryfunctions


def extract_volumes_from_many_files(path_config_file, which_vol=None):
    """
    Assumes a folder of many different volumes, and copies some in the folder given by 'out_folder'
    """

    if platform.system() == 'Windows':
        is_windows = True
    else:
        is_windows = False

    config_file = Path(path_config_file).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")

    video_fnames = [i for i in cfg['video_sets'].keys()]
    print(type(video_fnames))
    print("Found {} volumes.".format(len(video_fnames)))

    # Read basic metadata

    # Get volume indices to save
    if which_vol is None:
        which_vol = [0]

    for i_vol in which_vol:
        print("Reading volume {}/{}".format(i_vol, len(which_vol)))

        this_fname = video_fnames[i_vol]

        # Read and make output name
        this_volume = tifffile.imread(this_fname)
        output_name = 'img{}.tif'.format(i_vol)

        # Save in output folder
        fname = Path(video_fnames[0])
        output_path = os.path.join(Path(path_config_file).parents[0], 'labeled-data', fname.stem)

        tifffile.imsave(os.path.join(str(output_path), output_name), this_volume)

        print('Saved volume to {}\\{}'.format(output_path, output_name))


def extract_volumes_from_MATLAB_output(path_config_file, which_vol=None):
    """
    Takes a video filename, which is a large ome-tiff file, and saves a volume in the folder given by 'out_folder'
    """

    if platform.system() == 'Windows':
        is_windows = True
    else:
        is_windows = False

    config_file = Path(path_config_file).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")

    video_fname = [i for i in cfg['video_sets'].keys()][0]  # Assume one video for now (will be giant, ~3GB)
    print(video_fname)

    # Read basic metadata to get 'nz' and 'nt'
    with tifffile.TiffFile(video_fname) as vid:
        # print(tifffile.xml2dict(vid.ome_metadata))
        print(vid.ome_metadata)
        ome_metadata = vid.ome_metadata
        if isinstance(ome_metadata, str):
            # Appears to be a bug that just returns a string... can fix manually though
            ome_metadata = tifffile.xml2dict(ome_metadata)['OME']
        print(ome_metadata.keys())
        mdat = ome_metadata['Image']['Pixels']
        nz, nt = mdat['SizeZ'], mdat['SizeT']

    # Get volume indices to save
    if which_vol is None:
        which_vol = [0]

    for i_vol in which_vol:
        print("Read volume {}/{}".format(i_vol, len(which_vol)))

        # Convert scalar volume label to the sequential frames
        # Note: this may change for future input videos!
        vol_indices = list(range(i_vol * nz, i_vol * nz + nz))

        # Read and make output name
        this_volume = tifffile.imread(video_fname, key=vol_indices)
        output_name = 'img{}.tif'.format(i_vol)

        # Save in output folder
        fname = Path(video_fname)
        output_path = os.path.join(Path(path_config_file).parents[0], 'labeled-data', fname.stem)

        tifffile.imsave(os.path.join(str(output_path), output_name), this_volume)

        print('Saved volume to {}\\{}'.format(output_path, output_name))


def extract_volume_from_tiff_in_dlc_project(path_config_file, nz,
                                            video_fname=None,
                                            which_vol=None,
                                            which_slice=None,
                                            actually_write=True):
    """
    Takes a video filename, which is a large ome-tiff file, and saves a volume in the 'labeled-data' folder
        Can also save

    Expects a DeepLabCut project directory, and the video to be present in the videos/ directory

    if which_slice is not None:
        saves single slices (instead of full volumes)
    """
    if not actually_write:
        print("NOT ACTUALLY WRITING")

    if which_slice is None:
        saving_str = "volume"
    else:
        saving_str = "slice"

    config_file = Path(path_config_file).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")

    dlc_video_fname = [i for i in cfg['video_sets'].keys()][0]  # Assume one video for now (will be giant, ~3GB)
    if video_fname is None:
        video_fname = dlc_video_fname  # Otherwise use custom video location
    # print(video_fname)

    # Get volume indices to save
    if which_vol is None:
        which_vol = [0]

    for i_vol in which_vol:
        print("Reading {} {}/{}".format(saving_str, i_vol, len(which_vol) - 1))

        # Convert scalar volume label to the sequential frames
        # Note: this may change for future input videos!
        if which_slice is None:
            vol_indices = list(range(i_vol * nz, i_vol * nz + nz))
        else:
            vol_indices = list(range(i_vol * nz + which_slice, i_vol * nz + 1 + which_slice))
        print("Converted {} indices: {} to {} (not including last frame)".format(saving_str, vol_indices[0],
                                                                                 vol_indices[-1]))

        # Read and make output name
        if actually_write:
            this_volume = tifffile.imread(video_fname, key=vol_indices)
        output_name = 'img{}.tif'.format(i_vol)

        # Save in output folder
        fname = Path(dlc_video_fname)
        output_path = os.path.join(Path(path_config_file).parents[0], 'labeled-data', fname.stem)

        if actually_write:
            tifffile.imsave(os.path.join(str(output_path), output_name), this_volume)

        print('Saved {} to {}'.format(saving_str, os.path.join(output_path, output_name)))
