import logging
import os
import re
from pathlib import Path

import mat73
import numpy as np
from MultiChannelVol import MultiChannelVolume
from pynwb import NWBFile, NWBHDF5IO
from pynwb.ophys import OnePhotonSeries, OpticalChannel, ImageSegmentation, PlaneSegmentation, Fluorescence, RoiResponseSeries
from hdmf.data_utils import DataChunkIterator
from dateutil import tz
import pandas as pd
from datetime import datetime
from hdmf.backends.hdf5.h5_utils import H5DataIO
# ndx_mulitchannel_volume is the novel NWB extension for multichannel optophysiology in C. elegans
from ndx_multichannel_volume import CElegansSubject, OpticalChannelReferences, OpticalChannelPlus, ImagingVolume
from tifffile import tifffile
from tqdm.auto import tqdm

from wbfm.utils.projects.finished_project_data import ProjectData


def nwb_using_project_data(project_data: ProjectData, include_image_data=False, output_folder=None):
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

    nwb_file = nwb_with_traces_from_components(red_video, gce_quant, session_start_time, subject_id, strain, output_folder)
    return nwb_file


def nwb_from_matlab_tracker(matlab_fname, output_folder=None):
    """
    Convert a matlab tracker file to an NWB h5 file.

    Parameters
    ----------
    matlab_fname

    Returns
    -------

    """
    if output_folder is None:
        logging.warning("No output folder specified, will not save final output (this is a dry run)")

    mat = mat73.loadmat(matlab_fname)

    # Unpack variables from matlab file
    session_start_time = mat['added']['dateAdded'][0]
    # This is a string like '26-Jan-2019 10:35:22'; convert to datetime
    session_start_time = datetime.strptime(session_start_time, '%d-%b-%Y %H:%M:%S')
    # Convert the datetime to a string that can be used as a subject_id
    subject_id = session_start_time.strftime("%Y%m%d-%H-%M-%S")

    # Define a regular expression pattern to match "ZIM" followed by numbers
    pattern = r'ZIM(\d+)'
    match = re.search(pattern, matlab_fname)
    if match:
        # Extract the matched substring
        strain = match.group(0)
    else:
        print(f"Pattern 'ZIM' not found in the input string.")
        raise NotImplementedError

    # Unpack traces
    id_names = mat["ID1"]
    raw_colnames = [f"neuron_{i:03d}" for i in range(len(id_names))]
    # colnames = [dummy if ID is None else ID for dummy, ID in zip(raw_colnames, id_names)]
    colnames = raw_colnames
    gce_quant = pd.DataFrame(mat["deltaFOverF"], columns=colnames)
    # Add a new level to the columns to specify the type of data (here, just 'intensity_image')
    gce_quant.columns = pd.MultiIndex.from_product([['intensity_image'], gce_quant.columns])
    gce_dict = gce_quant.to_dict()
    # Also needs to have the additional columns that my freely moving projects do:
    #   ['x', 'y', 'z', 'intensity_image', 'label', 'index']
    # But actually build a dictionary and then convert to dataframe to avoid fragmentation warnings
    n = len(gce_quant)
    t_vec = list(gce_quant.index)
    for name in raw_colnames:
        # The label column has to be correct, i.e. each neuron should have a label such as 'neuron_001' -> 1
        idx = int(name.split('_')[1])
        gce_dict[('label', name)] = {t: idx for t in t_vec}
        # TODO: proper x, y, z, index
        gce_dict[('x', name)] = {t: 1 for t in t_vec}
        gce_dict[('y', name)] = {t: 1 for t in t_vec}
        gce_dict[('z', name)] = {t: 1 for t in t_vec}
        gce_dict[('index', name)] = {t: t for t in t_vec}
    gce_quant = pd.DataFrame(gce_dict)

    # Unpack video
    red_video = None

    nwb_file = nwb_with_traces_from_components(red_video, gce_quant, session_start_time, subject_id, strain, output_folder)
    return nwb_file


def nwb_with_traces_from_components(red_video, gce_quant, session_start_time, subject_id, strain, output_folder):
    # Initialize and populate the NWB file
    nwbfile = initialize_nwb_file(session_start_time, strain, subject_id)

    device = nwbfile.create_device(
        name="Spinning disk confocal",
        description="Zeiss Observer.Z1 Inverted Microscope with Yokogawa CSU-X1, "
                    "Zeiss LD LCI Plan-Apochromat 40x WI objective 1.2 NA",
        manufacturer="Zeiss"
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

    if output_folder:
        fname = os.path.join(output_folder, subject_id + '.nwb')
        logging.info(f"Saving NWB file to {fname}")
        with NWBHDF5IO(fname, mode='w') as io:
            io.write(nwbfile)
        logging.info(f"Saving successful!")

    return nwbfile


def initialize_nwb_file(session_start_time, strain, subject_id):
    nwbfile = NWBFile(
        session_description='Add a description for the experiment/session. Can be just long form text',
        # Can use any identity marker that is specific to an individual trial. We use date-time to specify trials
        identifier=session_start_time.strftime("%Y%m%d-%H-%M-%S"),
        # Specify date and time of trial. Datetime entries are in order Year, Month, Day, Hour, Minute, Second. Not all entries are necessary
        session_start_time=session_start_time,
        lab='Zimmer lab',
        institution='University of Vienna',
        related_publications=''
    )
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


def initialize_imaging_channels(video_data, nwbfile, device):
    # if red_video is None:
    #     return None, None, None
    # TODO: properly switch between red and green
    # FREE
    # grid_spacing = [0.325, 0.325, 1.5]
    # rate = 3.5  # Volumes per second
    # IMMOBILIZED
    grid_spacing = [0.4, 0.4, 2]
    rate = 2.0  # Volumes per second # TODO: get correct rate
    # GREEN
    emission_lambda = 525.
    emission_delta = 50.
    excitation_lambda = 488.
    # RED
    # emission_lambda = 617.
    # emission_delta = 73.
    # excitation_lambda = 561.
    video_shape = video_data.shape if video_data is not None else None

    # The DataChunkIterator wraps the data generator function and will stitch together the chunks as it iteratively reads over the full file
    if video_data is not None:
        data = DataChunkIterator(
            data=_iter_volumes(video_data),
            # this will be the max shape of the final image. Can leave blank or set as the size of your full data if you know that ahead of time
            maxshape=None,
            buffer_size=10,
        )
    else:
        data = H5DataIO(np.zeros((1, 1, 1, 1), dtype=np.uint8), compression='gzip')

    CalcOptChan = OpticalChannelPlus(
        name='GFP-GCaMP',
        description=f'GFP/GCaMP channel, f{excitation_lambda} excitation, {emission_lambda}/{emission_delta}m emission',
        excitation_lambda=excitation_lambda,
        excitation_range=[excitation_lambda - 1.5, excitation_lambda + 1.5],
        emission_range=[emission_lambda - emission_delta/2, emission_lambda + emission_delta/2],
        emission_lambda=emission_lambda
    )
    # This object just contains references to the order of channels because OptChannels does not preserve ordering
    CalcOptChanRefs = OpticalChannelReferences(
        name='OpticalChannelRefs',
        channels=[CalcOptChan]
    )

    CalcImagingVolume = ImagingVolume(
        name='CalciumImVol',
        description='Imaging plane used to acquire calcium imaging data',
        optical_channel_plus=[CalcOptChan],
        order_optical_channels=CalcOptChanRefs,
        device=device,
        location='Worm head',
        grid_spacing=grid_spacing,
        grid_spacing_unit='um',
        reference_frame='Worm head'
    )
    nwbfile.add_imaging_plane(CalcImagingVolume)

    optical_channel = OpticalChannel(
        name='GFP-GCaMP',
        description='GFP/GCaMP channel, 488 excitation, 525/50m emission',
        emission_lambda=emission_lambda
    )

    imaging_plane = nwbfile.create_imaging_plane(
        name='CalciumImPlane',
        description='Imaging plane used to acquire calcium imaging data',
        optical_channel=optical_channel,
        device=device,
        excitation_lambda=excitation_lambda,
        indicator='GFP-GCaMP',
        location='Worm head',
        grid_spacing=grid_spacing,
        grid_spacing_unit='um',
        reference_frame='Worm head'
    )

    # Use OnePhotonSeries object to store calcium imaging series data
    calcium_image_series = OnePhotonSeries(
        name="CalciumImageSeries",
        data=data,
        unit="n/a",
        scan_line_rate=0.5,
        dimension=video_shape,
        rate=rate,
        imaging_plane=imaging_plane,
    )
    return calcium_image_series, imaging_plane, CalcOptChanRefs


# define a data generator function that will yield a single data entry, in our case we are iterating over time points and creating a Z stack of images for each time point
def _iter_volumes(video_data):
    # Will return a 4d image: ZXYC
    if video_data is None:
        return None

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

    # print(
    #     blobquant.shape)  # Now dimensions are blob_ix, time, and data columns (X, Y, Z, gce_quant, ID). We are now ready to add this data to NWB objects.

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


def nwb_from_pedro_format(folder_name: str):
    """
    Pedro's manually organized format, which contains:
    1. ome.tif for the neuropal stacks
    2. .xlsx for the neuropal ID, with two sheets separated for head and tail
    3. .txt for fiji-exported positions of the ID'ed neurons. This is XY; Z is in the .xlsx file
    4. Optional: ome.tif for the head (if traces are calculated)
    5. Optional: ome.tif for the tail (if traces are calculated)
    6. Optional: .mat for the traces (if traces are calculated)

    Parameters
    ----------
    folder_name

    Returns
    -------

    """
    # Unpack the folder name into parts, which will form the basis of the input and output file names
    folder_dirname = Path(folder_name).name
    folder_parts = folder_dirname.split('_')
    all_files = os.listdir(folder_name)
    all_files = [f for f in all_files if not f.startswith('.')]

    # Unpack these parts into metadata for the NWB file
    date_str = folder_parts[0]
    day = int(date_str[:2])
    month = int(date_str[2:4])
    year = int("20" + date_str[4:])
    session_start_time = datetime(year, month, day)

    strain_id = folder_parts[1]

    subject_id = session_start_time.strftime("%Y%m%d-%H-%M-%S")

    # First load the ID and position data
    # File should be .xlsx and contain "NeuroPAL" in the name
    fname = [f for f in all_files if ('NeuroPAL' in f) and f.endswith('.xlsx')][0]
    fname = os.path.join(folder_name, fname)

    sheet_name_base = f"{folder_parts[0]}_{folder_parts[2]}"
    all_dfs_excel = []
    for suffix in ['head', 'tail']:
        sheet_name = f"{sheet_name_base}_{suffix}"
        df = pd.read_excel(fname, sheet_name=sheet_name)
        df.loc[~(df['neuron ID'].isnull()), 'body_part'] = suffix
        all_dfs_excel.append(df)
    df = all_dfs_excel[0].copy()
    df.update(all_dfs_excel[1])

    # Now load the XY position data
    fname = [f for f in all_files if ('NeuroPAL' in f) and f.endswith('.txt')][0]
    fname = os.path.join(folder_name, fname)
    df_xy = pd.read_csv(fname, sep='\t', header=None)
    df_xy.columns = ['X', 'Y']
    df = pd.concat([df_xy, df], axis=1)

    # Add column for the z position, which is contained in the comments
    def _convert_comment_to_z(entry):
        if isinstance(entry, str):
            return int(entry.split(' ')[1])
        else:
            return np.nan

    df['Z'] = df['comments'].apply(_convert_comment_to_z)

    # Load the neuropal stack (image)
    fname = [f for f in all_files if ('NeuroPAL' in f) and f.endswith('.ome.tif')][0]
    fname = os.path.join(folder_name, fname)
    with tifffile.TiffFile(fname) as f:
        neuropal_stacks = f.asarray()

    # Pack everything as a NWB file
    nwbfile = initialize_nwb_file(session_start_time, strain_id, subject_id)

    # TODO: is the microscope the same as the one used for the freely moving experiments?
    device = nwbfile.create_device(
        name="Spinning disk confocal",
        description="Zeiss Observer.Z1 Inverted Microscope with Yokogawa CSU-X1, "
                    "Zeiss LD LCI Plan-Apochromat 40x WI objective 1.2 NA",
        manufacturer="Zeiss"
    )

    # Neuropal and segmentation
    nwbfile = add_neuropal_stacks_to_nwb(nwbfile, device, neuropal_stacks)



    if output_folder:
        fname = os.path.join(output_folder, subject_id + '.nwb')
        logging.info(f"Saving NWB file to {fname}")
        with NWBHDF5IO(fname, mode='w') as io:
            io.write(nwbfile)
        logging.info(f"Saving successful!")

    return nwbfile


def add_neuropal_stacks_to_nwb(nwbfile, device, neuropal_stacks):
    """
    Add a neuropal stack to an existing NWB file.

    Parameters
    ----------
    nwbfile
    device
    neuropal_stacks

    Returns
    -------

    """

    # First, the metadata

    # Channels is a list of tuples where each tuple contains the fluorophore used, the specific emission filter used, and a short description
    # structured as "excitation wavelength - emission filter center point- width of emission filter in nm"
    # TODO: Make sure this list is in the same order as the channels in your data
    channels = [("mTagBFP2", " BrightLine HC 447/60", "405-447-60m"),  # UV excited, observe in blue
                ("CyOFP1", "BrightLine HC 617/73", "488-617-73m"),  # excited with blue, observe in red
                #("CyOFP1-high filter", "Chroma ET 700/75", "488-700-75m"),
                ("GFP-GCaMP", " BrightLine HC 525/50", "488-525-50m"),
                ("mNeptune 2.5", "Chroma ET 647/57", "561-647-57m"),
                ("Tag RFP-T", "Chroma ET 586/20", "561-586-20m"),
                #("mNeptune 2.5-far red", "Chroma ET 700/75", "639-700-75m")
                ]
    # We also have mScarlet, which is not normally in neuropal

    OptChannels = []
    OptChanRefData = []
    # The loop below takes the list of channels and converts it into a list of OpticalChannelPlus objects which hold the metadata
    # for the optical channels used in the experiment
    for fluor, des, wave in channels:
        excite = float(wave.split('-')[0])
        emiss_mid = float(wave.split('-')[1])
        emiss_range = float(wave.split('-')[2][:-1])
        OptChan = OpticalChannelPlus(
            name=fluor,
            description=des,
            excitation_lambda=excite,
            excitation_range=[excite - 1.5, excite + 1.5],
            emission_range=[emiss_mid - emiss_range / 2, emiss_mid + emiss_range / 2],
            emission_lambda=emiss_mid
        )

        OptChannels.append(OptChan)
        OptChanRefData.append(wave)

    # This object just contains references to the order of channels because OptChannels does not preserve ordering by itself
    OpticalChannelRefs = OpticalChannelReferences(
        name='OpticalChannelRefs',
        channels=OptChanRefData
    )

    ImagingVol = ImagingVolume(
        name='NeuroPALImVol',
        # Add connections to the OptChannels and OpticalChannelRefs objects
        optical_channel_plus=OptChannels,
        order_optical_channels=OpticalChannelRefs,
        # Free form description of what is being imaged in this volume
        description='NeuroPAL image of C. elegans brain',
        # Reference the device created earlier that was used to acquire this data
        device=device,
        # Specifies where in the C. elegans body the image is being taken of
        location="Head and Tail",
        # Specifies the voxel spacing in x, y, z respectively. The values specified should be how many micrometers of physical
        # distance are covered by a single pixel in each dimension
        # TODO: grid spacing
        grid_spacing=[0.3208, 0.3208, 0.75],
        grid_spacing_unit='micrometers',
        # Origin coords, origin coords unit, and reference frames are carry over fields from other model organisms where you
        # are likely only looking at a small portion of the brain. These fields are unfortunately required but feel free to put
        # whatever feels right here
        origin_coords=[0, 0, 0],
        origin_coords_unit="micrometers",
        reference_frame="Worm head"
    )

    nwbfile.add_imaging_plane(ImagingVol)  # add this ImagingVol to the nwbfile

    # Then, the data
    # TODO: Data is in order C, Z, Y, X when reading in??
    data = neuropal_stacks
    RGBW_channels = [0, 1, 4, 6]

    Image = MultiChannelVolume(
        name='NeuroPALImageRaw',
        # This is the same OpticalChannelRefs used in the associated Imaging Volume
        order_optical_channels=OpticalChannelRefs,
        description='free form description of image',
        # Specifies which channels in the image are associated with the RGBW channels - should be a list of channel indices as shown above
        RGBW_channels=RGBW_channels,
        # This is the raw data numpy array that we loaded above
        data=H5DataIO(data=data, compression=True),
        # This is a reference to the Imaging Volume object we defined previously
        imaging_volume=ImagingVol
    )

    nwbfile.add_acquisition(Image)

    return nwbfile
