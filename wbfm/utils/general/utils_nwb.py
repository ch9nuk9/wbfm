import logging
import os
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import OnePhotonSeries, OpticalChannel, ImageSegmentation, PlaneSegmentation, Fluorescence, RoiResponseSeries
from hdmf.data_utils import DataChunkIterator
from dateutil import tz
import pandas as pd
from datetime import datetime
from hdmf.backends.hdf5.h5_utils import H5DataIO
# ndx_mulitchannel_volume is the novel NWB extension for multichannel optophysiology in C. elegans
from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume
from tqdm.auto import tqdm

from wbfm.utils.projects.finished_project_data import ProjectData


def convert_project_data_to_nwb(project_data: ProjectData, include_image_data=False, output_folder=None):
    """
    Convert a ProjectData class to an NWB h5 file, optionally including all raw image data.

    Following: https://github.com/focolab/NWB/blob/main/NWB_tutorial.ipynb

    Parameters
    ----------
    project_data

    Returns
    -------

    """

    if output_folder is None:
        logging.warning("No output folder specified, will not save final output (this is a dry run)")

    # Unpack variables from project_data
    # TODO: proper date
    session_start_time = datetime(2022, 11, 27, 21, 41, 10, tzinfo=tz.gettz("Europe/Vienna"))
    # Convert the datetime to a string that can be used as a subject_id
    subject_id = session_start_time.strftime("%Y%m%d-%H-%M-%S")

    # Unpack traces
    gce_quant = project_data.red_traces.swaplevel(i=0, j=1, axis=1).copy()
    # Unpack video
    red_video = project_data.red_data

    # TODO: strain
    strain = "ZIM"

    # Initialize and populate the NWB file
    nwbfile = initialize_nwb_file(session_start_time, strain, subject_id)

    # TODO: Fix device
    device = nwbfile.create_device(
        name="Spinning disk confocal",
        description="Leica DMi8 Inverted Microscope with Yokogawa CSU-W1 SoRA, 40x WI objective 1.1 NA",
        manufacturer="Leica, Yokagawa"
    )

    calcium_image_series, imaging_plane, CalcOptChanRefs = initialize_imaging_channels(
        red_video, nwbfile, device
    )
    ImSeg, fluor = convert_traces_to_nwb_format(
        gce_quant, imaging_plane, calcium_image_series
    )

    ophys = nwbfile.create_processing_module(
        name='CalciumActivity',
        description='Calcium time series metadata, segmentation, and fluorescence data'
    )

    # Finish
    ophys.add(ImSeg)
    ophys.add(fluor)
    ophys.add(CalcOptChanRefs)

    # specify the file path you want to save this NWB file to
    if output_folder:
        fname = os.path.join(output_folder, subject_id + '.nwb')
        logging.info(f"Saving NWB file to {fname}")
        with NWBHDF5IO(fname, mode='w') as io:
            io.write(nwbfile)
        logging.info(f"Saving successful!")


def initialize_nwb_file(session_start_time, strain, subject_id):
    nwbfile = NWBFile(
        session_description='Add a description for the experiment/session. Can be just long form text',
        # Can use any identity marker that is specific to an individual trial. We use date-time to specify trials
        identifier='20221127-21-41-10',
        # Specify date and time of trial. Datetime entries are in order Year, Month, Day, Hour, Minute, Second. Not all entries are necessary
        session_start_time=session_start_time,
        lab='Zimmer lab',
        institution='University of Vienna',
        related_publications=''
    )
    # TODO: Fix dates and strain
    nwbfile.subject = CElegansSubject(
        # This is the same as the NWBFile identifier for us, but does not have to be. It should just identify the subject for this trial uniquely.
        subject_id=subject_id,
        # Age is optional but should be specified in ISO 8601 duration format similarly to what is shown here for growth_stage_time
        # age = pd.Timedelta(hours=2, minutes=30).isoformat(),
        # Date of birth is a required field but if you do not know or if it's not relevant, you can just use the current date or the date of the experiment
        date_of_birth=session_start_time,
        # Specify growth stage of worm - should be one of two-fold, three-fold, L1-L4, YA, OA, dauer, post-dauer L4, post-dauer YA, post-dauer OA
        growth_stage='YA',
        # Optional: specify time in current growth stage
        # growth_stage_time=pd.Timedelta(hours=2, minutes=30).isoformat(),
        # Specify temperature at which animal was cultivated
        cultivation_temp=20.,
        description="free form text description, can include whatever you want here",
        # Currently using the ontobee species link until NWB adds support for C. elegans
        species="http://purl.obolibrary.org/obo/NCBITaxon_6239",
        # Currently just using O for other until support added for other gender specifications
        sex="O",
        strain=strain
    )
    return nwbfile


def initialize_imaging_channels(red_video, nwbfile, device):
    # The DataChunkIterator wraps the data generator function and will stitch together the chunks as it iteratively reads over the full file
    data = DataChunkIterator(
        data=_iter_volumes(red_video),
        # this will be the max shape of the final image. Can leave blank or set as the size of your full data if you know that ahead of time
        maxshape=None,
        buffer_size=10,
    )

    # TODO: get correct excitation
    CalcOptChan = OpticalChannelPlus(
        name='GFP-GCaMP',
        description='GFP/GCaMP channel, 488 excitation, 525/50m emission',
        excitation_lambda=488.,
        excitation_range=[488 - 1.5, 488 + 1.5],
        emission_range=[525 - 25, 525 + 25],
        emission_lambda=525.
    )
    # This object just contains references to the order of channels because OptChannels does not preserve ordering
    CalcOptChanRefs = OpticalChannelReferences(
        name='OpticalChannelRefs',
        channels=[CalcOptChan]
    )

    # TODO: get grid spacing
    CalcImagingVolume = ImagingVolume(
        name='CalciumImVol',
        description='Imaging plane used to acquire calcium imaging data',
        optical_channel_plus=[CalcOptChan],
        order_optical_channels=CalcOptChanRefs,
        device=device,
        location='Worm head',
        grid_spacing=[0.3208, 0.3208, 1.5],
        grid_spacing_unit='um',
        reference_frame='Worm head'
    )
    wrapped_data = H5DataIO(data=data, compression="gzip", compression_opts=4)

    # TODO: Are the dimensions correct?
    # Use OnePhotonSeries object to store calcium imaging series data
    # See https://pynwb.readthedocs.io/en/stable/pynwb.ophys.html for other optional fields to include here.
    calcium_image_series = OnePhotonSeries(
        name="CalciumImageSeries",
        data=wrapped_data,
        unit="n/a",
        scan_line_rate=0.5,
        dimension=(650, 900, 22),
        rate=4.0,
        imaging_plane=CalcImagingVolume,
    )
    nwbfile.add_imaging_plane(CalcImagingVolume)

    # TODO: get correct excitation
    optical_channel = OpticalChannel(
        name='GFP-GCaMP',
        description='GFP/GCaMP channel, 488 excitation, 525/50m emission',
        emission_lambda=525.
    )

    # TODO: get correct excitation and grid
    imaging_plane = nwbfile.create_imaging_plane(
        name='CalciumImPlane',
        description='Imaging plane used to acquire calcium imaging data',
        optical_channel=optical_channel,
        device=device,
        excitation_lambda=488.,
        indicator='GFP-GCaMP',
        location='Worm head',
        grid_spacing=[0.3208, 0.3208, 1.5],
        grid_spacing_unit='um',
        reference_frame='Worm head'
    )

    # TODO: get correct dimensions and frame rate
    # Use OnePhotonSeries object to store calcium imaging series data
    calcium_image_series = OnePhotonSeries(
        name="CalciumImageSeries",
        data=data,
        unit="n/a",
        scan_line_rate=0.5,
        dimension=(1667, 650, 900, 22),
        rate=3.5,
        imaging_plane=imaging_plane,
    )
    return calcium_image_series, imaging_plane, CalcOptChanRefs


# define a data generator function that will yield a single data entry, in our case we are iterating over time points and creating a Z stack of images for each time point
def _iter_volumes(video_data):
    # Will return a 4d image: ZXYC

    # We iterate through all of the timepoints and yield each timepoint back to the DataChunkIterator
    for i in range(video_data.shape[0]):
        # Make sure array ends up as the correct dtype coming out of this function (the dtype that your data was collected as)
        vol = video_data[i]
        yield np.transpose(vol, [1, 2, 0])
    return


def convert_traces_to_nwb_format(gce_quant, imaging_plane,  calcium_image_series,
                                 DEBUG=False):
    # Copy the label column to be blob_ix, but need to manually create the multiindex because it is multiple columns
    new_columns = pd.MultiIndex.from_tuples([('blob_ix', c[1]) for c in gce_quant[['label']].columns])
    gce_quant[new_columns] = gce_quant['label'].copy()

    gce_quant.loc[:, ('blob_ix', slice(None))] = gce_quant.loc[:, ('blob_ix', slice(None))].fillna(
        method='bfill').fillna(method='ffill')

    # Expects a long single-level dataframe
    gce_quant = gce_quant.stack(level=1, dropna=False).reset_index(level=1, drop=True)  # .dropna(how='all')
    gce_quant.reset_index(inplace=True)

    # Replace NaN with 0's, because it has to be int in the end
    gce_quant = gce_quant.fillna(0)

    # Rename columns to be the format of this file
    gce_quant = gce_quant.rename(
        columns={'x': 'X', 'y': 'Y', 'z': 'Z', 'intensity_image': 'gce_quant', 'label': 'ID', 'index': 'T',
                 'blob_ix': 'blob_ix'})
    if DEBUG:
        print(len(gce_quant['blob_ix'].unique()))  # Count the number of unique blobs in this file
        print(len(gce_quant['T'].unique()))  # Count the number of unique time points in this file

    quant = gce_quant[['X', 'Y', 'Z', 'gce_quant', 'ID', 'T', 'blob_ix']]  # Reorder columns to order we want

    blobquant = None
    for idx in tqdm(gce_quant['blob_ix'].unique()):
        blob = quant[quant['blob_ix'] == idx]
        blobarr = np.asarray(blob[['X', 'Y', 'Z', 'gce_quant', 'ID']])
        blobarr = blobarr[np.newaxis, :, :]
        if blobquant is None:
            blobquant = blobarr

        else:
            blobquant = np.vstack((blobquant, blobarr))

    print(
        blobquant.shape)  # Now dimensions are blob_ix, time, and data columns (X, Y, Z, gce_quant, ID). We are now ready to add this data to NWB objects.

    planesegs = []

    for t in tqdm(range(blobquant.shape[1])):
        planeseg = PlaneSegmentation(
            name='Seg_tpoint_' + str(t),
            description='Neuron segmentation for time point ' + str(t) + ' in calcium image series',
            imaging_plane=imaging_plane,
            reference_images=calcium_image_series,
        )

        for i in range(blobquant.shape[0]):
            voxel_mask = blobquant[i, t, 0:3]  # X, Y, Z columns

            voxel_mask = np.hstack((voxel_mask, 1))
            voxel_mask = voxel_mask[np.newaxis, :]

            planeseg.add_roi(voxel_mask=voxel_mask)

        planesegs.append(planeseg)

    ImSeg = ImageSegmentation(
        name='CalciumSeriesSegmentation',
        plane_segmentations=planesegs
    )

    gce_data = np.transpose(
        blobquant[:, :, 3])  # Take only gce quantification column and transpose so time is in the first dimension

    rt_region = planesegs[0].create_roi_table_region(
        description='All segmented neurons associated with calcium image series',
        region=list(np.arange(blobquant.shape[0]))
    )

    RoiResponse = RoiResponseSeries(
        # See https://pynwb.readthedocs.io/en/stable/pynwb.ophys.html#pynwb.ophys.RoiResponseSeries for additional key word argument options
        name='CalciumImResponseSeries',
        description='Fluorescence activity for calcium imaging data',
        data=gce_data,
        rois=rt_region,
        unit='',
        rate=4.0
    )

    fluor = Fluorescence(
        name='CalciumFluorTimeSeries',
        roi_response_series=RoiResponse
    )

    return ImSeg, fluor
