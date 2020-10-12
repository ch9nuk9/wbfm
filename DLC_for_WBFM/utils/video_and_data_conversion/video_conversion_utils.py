import numpy as np
import tifffile
import os
from pathlib import Path
import platform
import cv2

##
## OME-TIFF
##

def write_video_from_ome_folder(num_frames, folder_name, out_fname,
                               out_dtype='uint8',
                               which_slice=None):
    """
    Write a video from a folder of ome-tiff files, where each one is a single volume
    
    'out_fname' should have te file extension included. Recommended: .avi
    """

    all_fnames = os.listdir(folder_name)
    all_fnames = sorted(all_fnames)[1:]

    # Load all into memory via appending, writing at the end
    for i, this_fname in enumerate(all_fnames):
        if i > num_frames-1:
            print("Finished writing {} files".format(num_frames))
            break

        full_fname = os.path.join(folder_name,this_fname)
        if i == 0:
            if which_slice is None:
                tmp = tifffile.imread(full_fname)
            else:
                tmp = tifffile.imread(full_fname)[which_slice,...]
            new_shape = [num_frames]
            new_shape.extend(list(tmp.shape))
            dat = np.zeros(new_shape, dtype=out_dtype)
            print("Final shape should be: ", new_shape)
            dat[i,...] = tmp
        elif which_slice is None:
            dat[i,...] = tifffile.imread(full_fname)
        else:
            dat[i,...] = tifffile.imread(full_fname)[which_slice,...]
        print("Reading frame {}/{}: ".format(i+1,num_frames))

    tifffile.imsave(out_fname, dat, dtype=out_dtype)

    return dat



def write_video_from_ome_file(num_frames, video_fname, out_fname, out_dtype='uint16', which_slice=None):
    """
    Takes a video filename, which is a single large ome-tiff file, and saves a smaller file in the folder given by 'out_fname'
    """

    # Read basic metadata
    with tifffile.TiffFile(video_fname) as vid:

        # Get size either using metadata or not
        ome_metadata = vid.ome_metadata
        if ome_metadata is not None:
            if isinstance(ome_metadata, str):
                # Appears to be a bug that just returns a string... can fix manually though
                ome_metadata = tifffile.xml2dict(ome_metadata)['OME']
#             print(ome_metadata.keys())
            mdat = ome_metadata['Image']['Pixels']
            nz, nt = mdat['SizeZ'], mdat['SizeT']
        else:
            # Just read it from the shape
            nt, nz, nx, ny = vid.series[0].shape

    for i_vol in range(num_frames):
        if i_vol%10 == 0:
            print("Read volume {}/{}".format(i_vol, num_frames))

        # Convert scalar volume label to the sequential frames
        # Note: this may change for future input videos!
        if which_slice is None:
            vol_indices = list(range(i_vol*nz, i_vol*nz + nz))
        else:
            vol_indices = i_vol*nz + which_slice

        # Actually read
        if i_vol == 0:
            if which_slice is None:
                tmp = tifffile.imread(video_fname, key=vol_indices)
            else:
                tmp = tifffile.imread(video_fname, key=vol_indices)
            new_shape = [num_frames]
            new_shape.extend(list(tmp.shape))
            dat = np.zeros(new_shape, dtype=out_dtype)
            print("Final shape should be: ", new_shape)
            dat[i_vol,...] = tmp
        elif which_slice is None:
            dat[i_vol,...] = tifffile.imread(video_fname, key=vol_indices)
        else:
            dat[i_vol,...] = tifffile.imread(video_fname, key=vol_indices)

        # Read and make output name
#         this_volume = tifffile.imread(video_fname, key=vol_indices)

    # Save in output folder
    output_name = os.path.join('.',out_fname)

    tifffile.imsave(output_name, dat, dtype=out_dtype)

    return dat



##
## .avi
##

# Following:
#  https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python/45868817
# Note: I'm not using the main answer, as that's for a captured video stream, not just an array
def write_numpy_as_avi(data, fname="output.avi", fps=10, dtype='uint16'):
    # Must have a numpy array (or hdf5 file?) named 'data'
    #  Frames should be stored in the first axis of 'data'
    if ".avi" not in fname:
        fname = fname + ".avi"

    # Make sure the whole range is used
    factor = np.max(data)/255.0 # Conversion from uint16 to uint8
    data = data / factor

    print("Writing to {}".format(fname))
    sz = data.shape
    writer = cv2.VideoWriter(fname,cv2.VideoWriter_fourcc(*"MJPG"), fps,(sz[2],sz[1]), isColor=False)

    for i_frame in range(sz[0]):
        f = data[i_frame,...].astype('uint8')
#         f = cv2.cvtColor(data[i_frame,...].astype(dtype),cv2.COLOR_GRAY2BGR)
#         f = ((f-np.min(f))/np.max(f)*255.0).astype('uint8')
#         if dtype is 'uint16':
#             f = (f/255.0).astype('uint8')
        writer.write(f)
#         writer.write(cv2.convertScaleAbs(f))
        if i_frame%10 == 0:
            print("Writing frame {}/{}".format(i_frame, sz[0]))

    writer.release()
    print("Finished")
