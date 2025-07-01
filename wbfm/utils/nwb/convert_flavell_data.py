import os
import nrrd
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import OpticalChannel
from ndx_multichannel_volume import MultiChannelVolumeSeries
from datetime import datetime
from hdmf.backends.hdf5.h5_utils import H5DataIO
import glob
import argparse
from wbfm.utils.nwb.utils_nwb_export import CustomDataChunkIterator
from wbfm.utils.nwb.utils_nwb_export import build_optical_channel_objects, _zimmer_microscope_device
import dask.array as da


def iter_volumes(base_dir, n_timepoints, channel):
    """Yield 3D volumes for a given channel, skipping missing files."""
    for t in range(n_timepoints):
        t_str = f"{t:04d}"
        pattern = f'{base_dir}/NRRD_cropped/*_t{t_str}_ch{channel}.nrrd'
        matches = glob.glob(pattern)
        if matches:
            path = matches[0]
        else:
            continue
        if os.path.exists(path):
            data, _ = nrrd.read(path)
            yield data
        else:
            continue


def iter_segmentations(base_dir, n_timepoints):
    """Yield 3D segmentation volumes, skipping missing files."""
    for t in range(n_timepoints):
        path = f'{base_dir}/img_roi_watershed/{t}.nrrd'
        if os.path.exists(path):
            data, _ = nrrd.read(path)
            yield data
        else:
            continue


def find_max_timepoint_volumes(base_dir, channel):
    """Find the maximum timepoint index for a given channel in NRRD_cropped."""
    pattern = f'{base_dir}/NRRD_cropped/*_ch{channel}.nrrd'
    matches = glob.glob(pattern)
    max_t = -1
    for path in matches:
        # Extract tXXXX from filename
        basename = os.path.basename(path)
        parts = basename.split('_')
        for part in parts:
            if part.startswith('t') and part[1:5].isdigit():
                t = int(part[1:5])
                if t > max_t:
                    max_t = t
    return max_t

def find_max_timepoint_segmentations(base_dir):
    """Find the maximum timepoint index for segmentations in img_roi_watershed."""
    pattern = f'{base_dir}/img_roi_watershed/*.nrrd'
    matches = glob.glob(pattern)
    max_t = -1
    for path in matches:
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        if name.isdigit():
            t = int(name)
            if t > max_t:
                max_t = t
    return max_t

def count_valid_volumes(base_dir, channel):
    """Count valid volumes for a channel."""
    pattern = f'{base_dir}/NRRD_cropped/*_ch{channel}.nrrd'
    matches = glob.glob(pattern)
    return len(matches)


def count_valid_segmentations(base_dir, n_timepoints):
    count = 0
    for t in range(n_timepoints):
        path = f'{base_dir}/img_roi_watershed/{t}.nrrd'
        if os.path.exists(path):
            count += 1
    return count


def dask_stack_volumes(volume_iter, n_frames, frame_shape):
    """Stack a generator of volumes into a dask array."""
    # Each block is a single volume (3D), stacked along axis=0 (time)
    return da.stack([da.from_array(vol, chunks=frame_shape) for vol in volume_iter], axis=0)


def create_nwb_file_only_images(session_description, identifier, session_start_time, device_name, imaging_rate):
    nwbfile = NWBFile(
        session_description=session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        lab='Flavell lab',
        institution='MIT'
    )
    device = nwbfile.create_device(name=device_name)
    opt_ch_green = OpticalChannel('GFP', 'green channel', 488.0)
    opt_ch_red = OpticalChannel('RFP', 'red channel', 561.0)
    imaging_plane = nwbfile.create_imaging_plane(
        name='ImagingPlane',
        optical_channel=[opt_ch_green, opt_ch_red],
        description='Dummy imaging plane',
        device=device,
        excitation_lambda=488.0,
        imaging_rate=imaging_rate,
        indicator='GFP/RFP',
        location='unknown'
    )
    return nwbfile, imaging_plane


def convert_flavell_to_nwb(
    base_dir,
    output_path,
    session_description='Dummy Flavell data conversion',
    identifier='flavell_dummy',
    device_name='Microscope',
    imaging_rate=1.0,
    DEBUG=False
):
    session_start_time = datetime.now()
    nwbfile, imaging_plane = create_nwb_file_only_images(
        session_description, identifier, session_start_time, device_name, imaging_rate
    )

    # Count valid frames for each channel
    if DEBUG:
        n_frames = 10
        print("DEBUG mode: limiting to first 10 time points")
    else:
        n_green = find_max_timepoint_volumes(base_dir, 1)
        n_red = find_max_timepoint_volumes(base_dir, 2)
        n_seg = find_max_timepoint_segmentations(base_dir)
        n_frames = max(n_green, n_red, n_seg)
        if n_frames == 0:
            raise RuntimeError("No valid frames found for all channels.")

    # Use the first valid green volume to get shape
    green_gen = iter_volumes(base_dir, n_frames, 1)
    try:
        first_green = next(green_gen)
    except StopIteration:
        raise RuntimeError("No green channel volumes found. Check your input data and n_frames value.")
    frame_shape = first_green.shape

    # Build dask arrays for each channel
    green_dask = dask_stack_volumes(iter_volumes(base_dir, n_frames, 1), n_frames, frame_shape)
    red_dask = dask_stack_volumes(iter_volumes(base_dir, n_frames, 2), n_frames, frame_shape)
    seg_dask = dask_stack_volumes(iter_segmentations(base_dir, n_frames), n_frames, frame_shape)

    # Make single multi-channel data series
    # Flavell data is already TXYZ
    green_red_dask = da.stack([green_dask, red_dask], axis=-1)

    chunk_seg = (1,) + frame_shape  # chunk along time only

    # Ensure chunk_video matches the number of dimensions in green_red_dask
    chunk_video = (1,) + green_red_dask.shape[1:-1] + (1,)
    print(f"Creating NWB file with chunk size {chunk_video} and size {green_red_dask.shape} for green/red data")
    green_red_data = H5DataIO(
        data=CustomDataChunkIterator(array=green_red_dask, chunk_shape=chunk_video),
        compression="gzip"
    )
    seg_data = H5DataIO(
        data=CustomDataChunkIterator(array=seg_dask, chunk_shape=chunk_seg),
        compression="gzip"
    )

    # Build metadata objects
    grid_spacing = (0.3, 0.3, 0.3)  # Flavell data is isotropic
    device = _zimmer_microscope_device(nwbfile)
    CalcImagingVolume, _ = build_optical_channel_objects(device, grid_spacing, ['red', 'green'])
    # Add directly to the file to prevent hdmf.build.errors.OrphanContainerBuildError
    nwbfile.add_imaging_plane(CalcImagingVolume)

    nwbfile.add_acquisition(MultiChannelVolumeSeries(
        name="CalciumImageSeries",
        description="Series of calcium imaging data",
        comments="Calcium imaging data from Flavell lab",
        data=green_red_data,  # data here should be series of indexed masks
        # Elements below can be kept the same as the CalciumImageSeries defined above
        device=device,
        unit="Voxel gray counts",
        scan_line_rate=2995.,
        # dimension=None, #  Gives a warning; what should this be?,
        resolution=1.,
        # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
        rate=imaging_rate,  # sampling rate in hz
        imaging_volume=CalcImagingVolume,
    ))
    # Add segmentation under the processed module
    calcium_imaging_module = nwbfile.create_processing_module(
        name='CalciumActivity',
        description='Calcium time series metadata, segmentation, and fluorescence data'
    )
    calcium_imaging_module.add(MultiChannelVolumeSeries(
        name="CalciumSeriesSegmentation",
        description="Series of indexed masks associated with calcium segmentation",
        comments="Segmentation masks for calcium imaging data from Flavell lab",
        data=seg_data,  # data here should be series of indexed masks
        # Elements below can be kept the same as the CalciumImageSeries defined above
        device=device,
        unit="Voxel gray counts",
        scan_line_rate=2995.,
        # dimension=None, #  Gives a warning; what should this be?,
        resolution=1.,
        # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
        rate=imaging_rate,  # sampling rate in hz
        imaging_volume=CalcImagingVolume,
    ))

    with NWBHDF5IO(output_path, 'w') as io:
        io.write(nwbfile)
    print(f"Done. NWB file written to {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Convert Flavell data to NWB format.")
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory containing input data')
    parser.add_argument('--output_path', type=str, required=False, help='Output NWB file path')
    parser.add_argument('--session_description', type=str, default='Flavell Lab Data', help='Session description')
    parser.add_argument('--identifier', type=str, default='flavell_001', help='NWB file identifier')
    parser.add_argument('--device_name', type=str, default='FlavellMicroscope', help='Device name')
    parser.add_argument('--imaging_rate', type=float, default=1.0, help='Imaging rate (Hz)')
    parser.add_argument('--debug', action='store_true', help='If set, only convert the first 10 time points')

    args = parser.parse_args()

    # If the output path is not an absolute path, make it absolute by joining with the base_dir
    if args.output_path is None:
        args.output_path = os.path.join(args.base_dir, 'flavell_data.nwb')
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(args.base_dir, args.output_path)

    convert_flavell_to_nwb(
        base_dir=args.base_dir,
        output_path=args.output_path,
        session_description=args.session_description,
        identifier=args.identifier,
        device_name=args.device_name,
        imaging_rate=args.imaging_rate,
        DEBUG=args.debug
    )
