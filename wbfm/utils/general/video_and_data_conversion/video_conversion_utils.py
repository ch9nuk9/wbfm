import os
import warnings
import cv2
import numpy as np
import tifffile

##
## OME-TIFF
##

def write_video_from_ome_folder3d(num_frames: int,
                                  folder_name: str,
                                  out_fname: str,
                                  out_dtype: str = 'uint8',
                                  which_slice: np.array = None):
    """
    Write a video from a folder of ome-tiff files, where each one is a single volume

    Assumes that the tiff files are 3d; which_slice allows a subset of slices to be written

    'out_fname' should have te file extension included. Recommended: .avi

    Input-Output:
        /*ome.tiff -> .avi
    """

    all_fnames = os.listdir(folder_name)
    all_fnames = sorted(all_fnames)[1:]

    # Load all into memory via appending, writing at the end
    for i, this_fname in enumerate(all_fnames):
        if i > num_frames - 1:
            print("Finished writing {} files".format(num_frames))
            break

        full_fname = os.path.join(folder_name, this_fname)
        if i == 0:
            if which_slice is None:
                tmp = tifffile.imread(full_fname)
            else:
                tmp = tifffile.imread(full_fname)[which_slice, ...]
            new_shape = [num_frames]
            new_shape.extend(list(tmp.shape))
            dat = np.zeros(new_shape, dtype=out_dtype)
            print("Final shape should be: ", new_shape)
            dat[i, ...] = tmp
        elif which_slice is None:
            dat[i, ...] = tifffile.imread(full_fname)
        else:
            dat[i, ...] = tifffile.imread(full_fname)[which_slice, ...]
        print("Reading frame {}/{}: ".format(i + 1, num_frames))

    tifffile.imsave(out_fname, dat, dtype=out_dtype)

    return dat


def write_video_from_ome_folder2d(folder_name: str,
                                  out_fname: str,
                                  num_frames: int = None,
                                  out_dtype: str = 'uint8',
                                  verbose: int = 2):
    """
    Write a video from a folder of ome-tiff files, where each one is a single frame

    Input-Output:
        /*ome.tiff -> .avi

    Parameters
    ----------
    folder_name - target folder; should have only 2d ome-tiff files
    out_fname - output filename; should have the file extension included. Recommended: .avi
    num_frames - number of frames to write; default writes all
    out_dtype - output datatype; default is 'uint8', i.e. integers from 0-255
    verbose - how much to print; 0, 1, or 2
    """

    all_fnames = os.listdir(folder_name)
    all_fnames = sorted(all_fnames)[1:]
    if verbose >= 1:
        print(f"Found {len(all_fnames)} files; assuming they are all ome.tiff")

    # Load all into memory via appending, writing at the end
    for i, this_fname in enumerate(all_fnames):
        if i > num_frames - 1:
            if verbose >= 1:
                print("Finished writing {} files".format(num_frames))
            break

        full_fname = os.path.join(folder_name, this_fname)
        if i == 0:
            tmp = tifffile.imread(full_fname)
            new_shape = [num_frames]
            new_shape.extend(list(tmp.shape))
            dat = np.zeros(new_shape, dtype=out_dtype)
            if verbose >= 1:
                print("Final shape should be: ", new_shape)
            dat[i, ...] = tmp
        else:
            dat[i, ...] = tifffile.imread(full_fname)
        if verbose >= 2:
            print("Reading frame {}/{}: ".format(i + 1, num_frames))
    else:
        if verbose >= 1:
            print(f"Finished writing all frames")

    tifffile.imsave(out_fname, dat, dtype=out_dtype)

    return dat


def write_video_from_ome_file(num_frames, video_fname, out_fname, out_dtype='uint16', which_slice=None,
                              img_format="TZXY"):
    """
    Takes a video filename, which is a single large ome-tiff file, and saves a smaller file in the folder given by 'out_fname'
        Note: Reads the metadata first... which may take a long time
        Note: reads data into memory as np.array(), then saves at the end

    Input-Output:
        ome.tiff -> ome.tiff
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
            if img_format == "TZXY":
                nt, nz, nx, ny = vid.series[0].shape
            else:
                raise Exception

    for i_vol in range(num_frames):
        if i_vol % 10 == 0:
            print("Read volume {}/{}".format(i_vol, num_frames))

        # Convert scalar volume label to the sequential frames
        # Note: this may change for future input videos!
        if which_slice is None:
            vol_indices = list(range(i_vol * nz, i_vol * nz + nz))
        else:
            vol_indices = i_vol * nz + which_slice

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
            dat[i_vol, ...] = tmp
        elif which_slice is None:
            dat[i_vol, ...] = tifffile.imread(video_fname, key=vol_indices)
        else:
            dat[i_vol, ...] = tifffile.imread(video_fname, key=vol_indices)

        # Read and make output name
    #         this_volume = tifffile.imread(video_fname, key=vol_indices)

    # Save in output folder
    output_name = os.path.join('', out_fname)

    tifffile.imsave(output_name, dat, dtype=out_dtype)

    return dat


def write_video_from_ome_file_subset(input_fname, output_fname, which_slice=None,
                                     num_frames=None, fps=10, frame_width=None, frame_height=None,
                                     num_slices=33,
                                     alpha=None,
                                     actually_write=True):
    """
    Writes a video from a single ome-tiff file that is incomplete, i.e. cannot be read using tifffile.imread()

    Uses cv2 for video writing, which requires exact frame size information. This can be found by reading a single page of a TiffFile

    To get good output videos if the data is not uint8, 'alpha' will probably have to be set as max(data)/255.0
        By default 'alpha' is calculated from the first frame, but the first several may be outliers

    Writes sequentially, and only reads one frame at a time

    Input-Output:
        ome.tiff -> .avi
    """

    if not actually_write:
        print("NOT ACTUALLY WRITING A FILE")

    if frame_width is None:
        # Get the exact frame size from the first page=slice
        with tifffile.TiffFile(input_fname) as tif:
            frame_height, frame_width = tif.pages[0].shape
            print(f'Read shape of (H,W) = ({frame_height}, {frame_width})')
            if alpha is None:
                alpha = 0.9 * 255.0 / np.max(tif.pages[0].asarray())
                print(f'Calculated alpha as {alpha}')
        # Future: also get the number of z-slices

    if alpha is None:
        alpha = 1.0
    # Set up the video writer
    # ALSO NOT WORKKING: , FRWA, FRWD, IRAW, LAGS, LCW2, PIMJ, ASLC "-1",
    fourcc = 0
    video_out = cv2.VideoWriter(output_fname, fourcc=fourcc, fps=fps, frameSize=(frame_width, frame_height),
                                isColor=False)

    with tifffile.TiffFile(input_fname, multifile=False) as tif:
        for i, page in enumerate(tif.pages):
            if i % num_slices != which_slice:
                continue
            print(f'Page {i}/{len(tif.pages)}')
            # Bottleneck line
            img = page.asarray()
            # Convert to proper format, and write single frame
            img = (alpha * img).astype('uint8')
            if actually_write:
                video_out.write(img)
            if num_frames is not None and i > num_frames: break
    video_out.release()



##
## .avi
##

# Following:
#  https://stackoverflow.com/questions/29317262/opencv-video-saving-in-python/45868817
# Note: I'm not using the main answer, as that's for a captured video stream, not just an array
def write_numpy_as_avi(data,
                       out_fname="output.avi",
                       fps=10,
                       is_color=False,
                       verbose=0):
    """
    Assumes shape TXYC

    Can only write uint8

    Assumes all preprocessing is already done

    Assumes color is the last dimension; doesn't check for this

    Input-Output:
        np.array() -> .avi
    """

    if verbose >= 1:
        print("Writing to {}".format(out_fname))

    frame_width = data.shape[2]
    frame_height = data.shape[1]

    # Set up the video writer
    # On each system, different codecs are available
    # FOURCC is a 4-byte code used to specify the video codec.
    # The list of available codes can be found by setting fourcc = -1, which will print the list of available codecs
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    options = {'fourcc': fourcc,#-1,
               'fps': fps,
               'isColor': is_color,
               'frameSize': (frame_width, frame_height)}
    try:
        writer = cv2.VideoWriter(out_fname, **options)
        for i_frame in range(data.shape[0]):
            f = data[i_frame, ...].astype('uint8')
            writer.write(f)
    finally:
        writer.release()

    if verbose >= 1:
        print("Finished")
