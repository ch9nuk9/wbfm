import numpy as np
import tifffile
import os
import cv2
# from DLC_for_WBFM.config.class_configuration import load_config, build_avi_fnames

import warnings

##
## OME-TIFF
##

def write_video_from_ome_folder(num_frames, folder_name, out_fname,
                               out_dtype='uint8',
                               which_slice=None):
    """
    Write a video from a folder of ome-tiff files, where each one is a single volume

    'out_fname' should have te file extension included. Recommended: .avi

    Input-Output:
        /*ome.tiff -> .avi
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
            if img_format is "TZXY":
                nt, nz, nx, ny = vid.series[0].shape
            else:
                raise Exception

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



def write_video_from_ome_file_subset(input_fname, output_fname, which_slice=None,
                                     num_frames = None, fps=10, frame_width = None, frame_height = None,
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
                alpha =  0.9 * 255.0 / np.max(tif.pages[0].asarray())
                print(f'Calculated alpha as {alpha}')
        # Future: also get the number of z-slices

    if alpha is None:
        alpha = 1.0
    # Set up the video writer
    #ALSO NOT WORKKING: , FRWA, FRWD, IRAW, LAGS, LCW2, PIMJ, ASLC "-1",
    fourcc=0
    video_out = cv2.VideoWriter(output_fname, fourcc=fourcc, fps=fps, frameSize=(frame_width,frame_height), isColor=False)

    with tifffile.TiffFile(input_fname, multifile=False) as tif:
        for i, page in enumerate(tif.pages):
            if i % num_slices != which_slice:
                continue
            print(f'Page {i}/{len(tif.pages)}')
            # Bottleneck line
            img = page.asarray()
            # Convert to proper format, and write single frame
            img = (alpha*img).astype('uint8')
            if actually_write:
                video_out.write(img)
            if num_frames is not None and i > num_frames: break
    video_out.release()



## For use with config files
def write_minimax_projection_from_btf(config_file):
    """
    Note: writes videos to the same folder as the config file

    See also: write_video_projection_from_ome_file_subset
    """

    c = load_config(config_file)
    # Make sure the target filenames exist
    c = build_avi_fnames(c)

    # Get preprocessing settings
    frame_height, frame_width = c.datafiles.get_frame_size()

    params = dict(which_slices=c.preprocessing.which_slices(),
                start_volume=c.preprocessing.start_volume,
                num_frames=c.preprocessing.num_frames,
                frame_width=frame_width,
                frame_height=frame_height,
                num_slices=c.preprocessing.num_total_slices,
                alpha=c.preprocessing.alpha)

    # Do red (tracking) channel
    video_fname = c.datafiles.red_bigtiff_fname
    out_fname = c.datafiles.red_avi_fname

    write_video_projection_from_ome_file_subset(video_fname,
                                                out_fname,
                                                **params)

    # Do green (measurement) channel
    video_fname = c.datafiles.green_bigtiff_fname
    out_fname = c.datafiles.green_avi_fname

    write_video_projection_from_ome_file_subset(video_fname,
                                                out_fname,
                                                **params)

    return c



def write_video_projection_from_ome_file_subset(video_fname,
                                                out_fname,
                                                out_dtype='uint16',
                                                which_slices=None,
                                                start_volume=None,
                                                num_frames=None,
                                                fps=10,
                                                frame_width=608,
                                                frame_height=610,
                                                num_slices=33,
                                                alpha=None,
                                                flip_x=False,
                                                verbose=0):
    """
    Writes a video from a single ome-tiff file that is incomplete, i.e. cannot be read using tifffile.imread()
        This takes a max projection of the slices in 'which_slices', which is a FULL LIST of the desired frames

    Uses cv2 for video writing, which requires exact frame size information. This can be found by reading a single page of a TiffFile

    To get good output videos if the data is not uint8, 'alpha' will probably have to be set as max(data)/255.0

    Note that I skip the first volume by default, because it is significantly different

    Input-Output:
        ome.tiff -> .avi
    """
    # Set up the video writer
    fourcc=0
    video_out = cv2.VideoWriter(out_fname, fourcc=fourcc, fps=fps, frameSize=(frame_width,frame_height), isColor=False)

    assert num_slices%2==1, f"num_slices must be odd; was {num_slices}"

    # By default skip the first volume
    if start_volume is None:
        start_volume = num_slices
    if verbose >= 2 & (start_volume % num_slices != 0):
        print(f'Converting volume index {start_volume} to frame index {start_volume * num_slices}')
        start_volume = start_volume * num_slices

    # Set up the counting indices
    start_of_each_frame = which_slices[0]
    if verbose >= 1 and start_of_each_frame < 5:
        warnings.warn("As of 14.10.2020, the first several frames are very bad! Do you really mean to use these?")
    end_of_each_frame = which_slices[-1]
    # alpha *= 1.0 / len(which_slices) # Also takes a mean

    i_frame_count = 0
    img_tmp = np.zeros((len(which_slices), frame_height, frame_width))

    if verbose >= 2:
        print(f'Taking a max of {len(which_slices)} slices, starting at {start_of_each_frame}' )

    with tifffile.TiffFile(video_fname, multifile=False) as tif:
        for i_page, page in enumerate(tif.pages):
            i_slice_raw = i_page % num_slices
            i_slice_tmp = i_slice_raw - start_of_each_frame
            # Skip some frames
            if (i_page < start_volume) or (i_slice_raw not in which_slices):
                continue
            if verbose >= 2:
                print(f'Page {i_page}/{num_frames*num_slices}; a portion of slice {i_frame_count}/{num_frames} to tmp array index {i_slice_tmp}')

            img_tmp[i_slice_tmp,...] = page.asarray()

            if i_slice_raw == end_of_each_frame:
                # Take a mini-max projection
                if alpha is not None:
                    img_tmp *= alpha
                final_img = np.max(img_tmp, axis=0).astype('uint8')
                # final_img = np.max(img_tmp, axis=0).astype(out_dtype)
                # final_img = np.max((alpha*img_tmp), axis=0).astype(out_dtype)
                if flip_x:
                    # gcamp and mcherry are mirrored in the WBFM setup
                    final_img = np.flip(final_img, axis=1)
                video_out.write(final_img)
                i_frame_count += 1

            if num_frames is not None and i_frame_count > num_frames: break
    if verbose >= 1:
        print(f"Finished writing {i_frame_count} frames to {out_fname}")
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
                       frame_width=608,
                       frame_height=610,
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

    # Set up the video writer
    # writer = cv2.VideoWriter(fname,cv2.VideoWriter_fourcc(*"MJPG"), fps,(sz[2],sz[1]), isColor=isColor)
    opt = {'fourcc': 0,
           'fps': fps,
           'isColor': is_color,
           'frameSize': (frame_width, frame_height)}
    try:
        writer = cv2.VideoWriter(out_fname, **opt)
        for i_frame in range(data.shape[0]):
            f = data[i_frame, ...].astype('uint8')
            writer.write(f)
    finally:
        writer.release()

    if verbose >= 1:
        print("Finished")
