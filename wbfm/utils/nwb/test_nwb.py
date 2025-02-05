from pynwb import NWBFile, NWBHDF5IO


class TestNWB:
    """Copied from: https://github.com/focolab/NWBelegans/blob/main/check_NWB.py"""

    def __init__(self, nwbfile):
        has_neuropal = False
        has_calcium_imaging = False
        has_calcium_traces = False
        has_segmentation = False
        with NWBHDF5IO(nwbfile, mode='r', load_namespaces=True) as io:
            if isinstance(io, NWBFile):
                print('NWB file loaded successfully')
                read_nwbfile = io
            else:
                read_nwbfile = io.read()

            subject = read_nwbfile.subject  # get the metadata about the experiment subject
            growth_stage = subject.growth_stage
            try:
                image = read_nwbfile.acquisition['NeuroPALImageRaw'].data[:]  # get the neuroPAL image as a np array
                channels = read_nwbfile.acquisition['NeuroPALImageRaw'].RGBW_channels[
                           :]  # get which channels of the image correspond to which RGBW pseudocolors
                im_vol = read_nwbfile.acquisition[
                    'NeuroPALImageRaw'].imaging_volume  # get the metadata associated with the imaging acquisition

                # get the locations of neuron centers
                seg = read_nwbfile.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons'].voxel_mask[:]
                labels = read_nwbfile.processing['NeuroPAL']['NeuroPALSegmentation']['NeuroPALNeurons']['ID_labels'][:]
                # get information about all of the optical channels used in acquisition
                optchans = im_vol.optical_channel_plus[:]
                # get the order of the optical channels in the image
                chan_refs = read_nwbfile.processing['NeuroPAL']['OpticalChannelRefs'].channels[:]
                has_neuropal = True
            except KeyError as e:
                print(e)

            try:
                # load the first 15 frames of the calcium images
                calcium_frames = read_nwbfile.acquisition['CalciumImageSeries'].data[0:15, :, :, :]
                size = read_nwbfile.acquisition['CalciumImageSeries'].data.shape
                print(f"Size of calcium imaging data: {size}")
                has_calcium_imaging = True
            except KeyError as e:
                print(e)

            try:
                try:
                    fluor = read_nwbfile.processing['CalciumActivity']['SignalFluorescence']['SignalCalciumImResponseSeries'].data[:]
                except KeyError:
                    fluor = read_nwbfile.processing['CalciumActivity']['SignalDFoF'][
                                'SignalCalciumImResponseSeries'].data[:]
                print(f"Size of calcium imaging traces: {fluor.shape}")
                has_calcium_traces = True

            except KeyError as e:
                print(e)

            try:
                calc_seg = read_nwbfile.processing['CalciumActivity']['CalciumSeriesSegmentation'][
                               'Seg_tpoint_0'].voxel_mask[:]
                has_segmentation = True
            except KeyError as e:
                print(e)
            except TypeError:
                # Then it is a fully saved voxel video
                try:
                    calc_seg = read_nwbfile.processing['CalciumActivity']['SegmentationVol0'][
                                   'Seg_tpoint_0'].voxel_mask[:]
                    has_segmentation = True
                except KeyError as e:
                    print(e)

        print(f"Found the following data in the NWB file: \n"
              f"NeuroPAL image:         {has_neuropal}\n"
              f"Video calcium imaging:  {has_calcium_imaging}\n"
              f"Video calcium traces:   {has_calcium_traces}\n"
              f"Video segmentation:     {has_segmentation}")
