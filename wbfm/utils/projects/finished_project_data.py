import concurrent
import logging
from collections import defaultdict
from pathlib import Path

from matplotlib import pyplot as plt

from wbfm.utils.external.utils_jupyter import executing_in_notebook
from wbfm.utils.external.utils_zarr import zarr_reader_folder_or_zipstore
from wbfm.utils.general.custom_errors import NoMatchesError
from wbfm.utils.general.postures.centerline_classes import WormFullVideoPosture, shade_using_behavior
from wbfm.utils.general.preprocessing.utils_preprocessing import PreprocessingSettings
from wbfm.utils.neuron_matching.class_reference_frame import ReferenceFrame
from wbfm.utils.neuron_matching.matches_class import MatchesWithConfidence, get_mismatches
from wbfm.utils.projects.utils_neuron_names import int2name_neuron
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Union, List
import napari
import numpy as np
import pandas as pd
import zarr
from tqdm.auto import tqdm

from wbfm.utils.external.utils_pandas import dataframe_to_numpy_zxy_single_frame, df_to_matches, \
    get_column_name_from_time_and_column_value
from wbfm.utils.neuron_matching.class_frame_pair import FramePair
from wbfm.utils.projects.physical_units import PhysicalUnitConversion
from wbfm.utils.tracklets.utils_tracklets import fix_global2tracklet_full_dict, check_for_unmatched_tracklets
from sklearn.neighbors import NearestNeighbors
from wbfm.utils.tracklets.tracklet_class import DetectedTrackletsAndNeurons
from wbfm.utils.projects.plotting_classes import TracePlotter, TrackletAndSegmentationAnnotator
from segmentation.util.utils_metadata import DetectedNeurons
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from wbfm.utils.projects.utils_filenames import read_if_exists, pickle_load_binary, \
    load_file_according_to_precedence, pandas_read_any_filetype, get_sequential_filename
from wbfm.utils.projects.utils_project import safe_cd
# from functools import cached_property # Only from python>=3.8
from backports.cached_property import cached_property


@dataclass
class ProjectData:
    project_dir: str
    project_config: ModularProjectConfig  # Custom class

    red_data: zarr.Array = None  # Actual video (~100 GB)
    green_data: zarr.Array = None  # Actual video (~100 GB)

    raw_segmentation: zarr.Array = None  # Full-sized segmentation (before tracking)
    segmentation: zarr.Array = None  # Full-sized segmentation (after tracking -> colors are aligned)
    segmentation_metadata: DetectedNeurons = None  # Easy conversion between segmentation ID and position

    # OLD
    df_training_tracklets: pd.DataFrame = None
    reindexed_masks_training: zarr.Array = None

    # Traces as calculated from the segmentation
    red_traces: pd.DataFrame = None
    green_traces: pd.DataFrame = None

    # For plotting and visualization
    worm_posture_class: WormFullVideoPosture = None  # Allows coloring the traces (currently, done manually)
    background_per_pixel: float = None  # Simple version of background correction
    likelihood_thresh: float = None  # When plotting, plot gaps for low-confidence time points

    all_used_fnames: list = None  # Save a list of all paths to raw data
    verbose: int = 2  # How much to print when running methods

    # Precedence when multiple are available
    precedence_global2tracklet: list = None
    precedence_df_tracklets: list = None
    precedence_tracks: list = None

    # Will be set as loaded
    final_tracks_fname: str = None
    global2tracklet_fname: str = None
    df_all_tracklets_fname: str = None
    force_tracklets_to_be_sparse: bool = True  # TODO: pass as arg

    _custom_frame_indices: list = None

    # Classes for more functionality
    _trace_plotter: TracePlotter = None
    physical_unit_conversion: PhysicalUnitConversion = None

    # Values for ground truth annotation (reading from excel or .csv)
    finished_neurons_column_name: str = "Finished?"

    # EXPERIMENTAL (but tested)
    use_custom_padded_dataframe: bool = False

    def __post_init__(self):
        """
        Load values from disk config files if the user did not set them
        """
        track_cfg = self.project_config.get_tracking_config()
        if self.precedence_global2tracklet is None:
            self.precedence_global2tracklet = track_cfg.config['precedence_global2tracklet']
        if self.precedence_df_tracklets is None:
            self.precedence_df_tracklets = track_cfg.config['precedence_df_tracklets']
        if self.precedence_tracks is None:
            self.precedence_tracks = track_cfg.config['precedence_tracks']

    @cached_property
    def intermediate_global_tracks(self) -> pd.DataFrame:
        """
        Dataframe of tracks produced by the global tracker alone (no tracklets)

        Names are aligned with the final traces
        """
        tracking_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        fname = tracking_cfg.config['leifer_params']['output_df_fname']
        fname = tracking_cfg.resolve_relative_path(fname, prepend_subfolder=False)
        self.all_used_fnames.append(fname)

        global_tracks = read_if_exists(fname)
        return global_tracks

    @cached_property
    def final_tracks(self) -> pd.DataFrame:
        """
        Dataframe of tracks produced by combining the global tracker and tracklets

        Names are aligned with the final traces
        """
        tracking_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        fname = tracking_cfg.config['leifer_params']['output_df_fname']
        possible_fnames = dict(automatic=tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df'),
                               imputed=tracking_cfg.resolve_relative_path_from_config('missing_data_imputed_df'),
                               fdnc=tracking_cfg.resolve_relative_path(fname, prepend_subfolder=False))

        fname_precedence = self.precedence_tracks
        final_tracks, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                this_reader=read_if_exists)
        self.final_tracks_fname = fname
        self.all_used_fnames.append(fname)
        return final_tracks

    def get_final_tracks_only_finished_neurons(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        See get_ground_truth_annotations()

        Dataframe with subset of columns (neurons)
        """
        df_gt, finished_neurons = self.get_list_of_finished_neurons()

        return df_gt.loc[:, finished_neurons], finished_neurons

    def get_list_of_finished_neurons(self):
        """Get the finished neurons and dataframe that will be subset-ed"""
        df_gt = self.final_tracks
        finished_neurons = self.finished_neuron_names
        return df_gt, finished_neurons

    @cached_property
    def global2tracklet(self) -> dict:
        """Dictionary of matches between neuron names (str) and tracklets (str)"""
        tracking_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        possible_fnames = dict(
            manual=tracking_cfg.resolve_relative_path_from_config('manual_correction_global2tracklet_fname'),
            automatic=tracking_cfg.resolve_relative_path_from_config('global2tracklet_matches_fname'))

        fname_precedence = self.precedence_global2tracklet
        global2tracklet, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                   this_reader=pickle_load_binary)
        self.global2tracklet_fname = fname
        if global2tracklet is not None:
            global2tracklet = fix_global2tracklet_full_dict(self.df_all_tracklets, global2tracklet)
        return global2tracklet

    @cached_property
    def raw_frames(self) -> List[ReferenceFrame]:
        """
        List of ReferenceFrame objects

        This can become desynced if the user modifies segmentation
        """
        self.logger.info("First time loading the raw frames, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'frame_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        frames = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return frames

    @cached_property
    def raw_matches(self) -> Dict[tuple, FramePair]:
        """
        Dict of FramePair objects

        This can become desynced if the user modifies segmentation
        """
        self.logger.info("First time loading the raw matches, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'match_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        matches = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return matches

    @cached_property
    def _raw_clust(self) -> pd.DataFrame:
        """
        Legacy custom dataframe format, before transforming into tracklets.

        Use not suggested
        """
        self.logger.info("First time loading the raw cluster dataframe, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'clust_df_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        clust = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return clust

    @cached_property
    def df_all_tracklets(self) -> pd.DataFrame:
        """Sparse Dataframe of all tracklets"""
        self.logger.info("First time loading all the tracklets, may take a while...")
        train_cfg = self.project_config.get_training_config()
        track_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        possible_fnames = dict(
            manual=track_cfg.resolve_relative_path_from_config('manual_correction_tracklets_df_fname'),
            wiggle=track_cfg.resolve_relative_path_from_config('wiggle_split_tracklets_df_fname'),
            automatic=train_cfg.resolve_relative_path_from_config('df_3d_tracklets'))
        fname_precedence = self.precedence_df_tracklets
        df_all_tracklets, fname = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                                    this_reader=pandas_read_any_filetype)
        self.df_all_tracklets_fname = fname
        self.all_used_fnames.append(fname)

        if self.force_tracklets_to_be_sparse:
            # if not check_if_fully_sparse(df_all_tracklets):
            if True:
                self.logger.warning("Casting tracklets as sparse, may take a minute")
                # df_all_tracklets = to_sparse_multiindex(df_all_tracklets)
                df_all_tracklets = df_all_tracklets.astype(pd.SparseDtype("float", np.nan))
            else:
                self.logger.info("Found sparse matrix")
        self.logger.info("Finished loading tracklets")

        return df_all_tracklets

    @cached_property
    def tracklet_annotator(self) -> TrackletAndSegmentationAnnotator:
        """Custom class that implements manual modification of tracklets and segmentation"""
        tracking_cfg = self.project_config.get_tracking_config()
        training_cfg = self.project_config.get_training_config()
        # fname = tracking_cfg.resolve_relative_path_from_config('global2tracklet_matches_fname')

        obj = TrackletAndSegmentationAnnotator(
            self.tracklets_and_neurons_class,
            self.global2tracklet,
            segmentation_metadata=self.segmentation_metadata,
            tracking_cfg=tracking_cfg,
            training_cfg=training_cfg,
            z_to_xy_ratio=self.physical_unit_conversion.z_to_xy_ratio,
            buffer_masks=zarr.zeros_like(self.segmentation),
            logger=self.logger
        )

        obj.initialize_gt_model_mismatches(self)
        return obj

    @cached_property
    def tracklets_and_neurons_class(self) -> DetectedTrackletsAndNeurons:
        """Class that connects tracklets with raw neuron segmentation"""
        self.logger.info("Loading tracklets")
        _ = self.df_all_tracklets  # Make sure it is loaded
        return DetectedTrackletsAndNeurons(self.df_all_tracklets, self.segmentation_metadata,
                                           dataframe_output_filename=self.df_all_tracklets_fname,
                                           use_custom_padded_dataframe=self.use_custom_padded_dataframe)

    @property
    def logger(self) -> logging.Logger:
        return self.project_config.logger

    def load_tracklet_related_properties(self):
        _ = self.df_all_tracklets
        # _ = self._raw_clust

    def load_interactive_properties(self):
        _ = self.tracklet_annotator

    def load_frame_related_properties(self):
        _ = self.raw_frames
        _ = self.raw_matches

    def load_segmentation_related_properties(self):
        _ = self.segmentation_metadata.segmentation_metadata
        self.all_used_fnames.append(self.segmentation_metadata.segmentation_metadata.detection_fname)

    @cached_property
    def num_frames(self) -> int:
        """Note that this is cached so that a user can overwrite the number of frames"""
        return self.project_config.config['dataset_params']['num_frames']

    def custom_frame_indices(self) -> list:
        """For overriding the normal iterator over frames, for skipping problems etc."""
        if self._custom_frame_indices is not None:
            return self._custom_frame_indices
        else:
            return list(range(self.num_frames))

    def get_frame_index_generator(self):
        """Generator yielding values from custom_frame_indices"""
        for val in self.custom_frame_indices():
            yield val

    @property
    def which_training_frames(self) -> list:
        train_cfg = self.project_config.get_training_config()
        return train_cfg.config['training_data_3d']['which_frames']

    @property
    def num_training_frames(self) -> int:
        return len(self.which_training_frames)

    def check_data_desyncing(self, raise_error=True):
        """
        Checks desynchronization in the tracklet-neuron matching database

        See: check_for_unmatched_tracklets
        """
        self.logger.info("Checking for database desynchronization")
        unmatched_tracklets = check_for_unmatched_tracklets(self.df_all_tracklets, self.global2tracklet,
                                                            raise_error=raise_error)

    @staticmethod
    def unpack_config_file(project_path: Union[str, ModularProjectConfig]):
        """Unpack config file into its components"""
        if isinstance(project_path, ModularProjectConfig):
            cfg = project_path
        else:
            opt = {'log_to_file': not executing_in_notebook()}
            cfg = ModularProjectConfig(project_path, **opt)

        project_dir = cfg.project_dir

        segment_cfg = cfg.get_segmentation_config()
        train_cfg = cfg.get_training_config()
        tracking_cfg = cfg.get_tracking_config()
        traces_cfg = cfg.get_traces_config()

        return cfg, segment_cfg, train_cfg, tracking_cfg, traces_cfg, project_dir

    @staticmethod
    def _load_data_from_configs(cfg: ModularProjectConfig,
                                segment_cfg: SubfolderConfigFile,
                                train_cfg: SubfolderConfigFile,
                                tracking_cfg: SubfolderConfigFile,
                                traces_cfg: SubfolderConfigFile,
                                project_dir,
                                to_load_tracklets=False,
                                to_load_interactivity=False,
                                to_load_frames=False,
                                to_load_segmentation_metadata=False,
                                initialization_kwargs=None):
        """Load all data (Dataframes, etc.) from disk using filenames defined in config files"""
        # Initialize object in order to use cached properties
        if initialization_kwargs is None:
            initialization_kwargs = {}
        else:
            cfg.logger.info(f"Initialized project with custom settings: {initialization_kwargs}")

        obj = ProjectData(project_dir, cfg)
        for k, v in initialization_kwargs.items():
            obj.k = v

        obj.all_used_fnames = []
        preprocessing_settings = PreprocessingSettings.load_from_config(cfg, do_background_subtraction=False)

        red_dat_fname = cfg.config['preprocessed_red']
        green_dat_fname = cfg.config['preprocessed_green']
        red_traces_fname = traces_cfg.resolve_relative_path(traces_cfg.config['traces']['red'])
        green_traces_fname = traces_cfg.resolve_relative_path(traces_cfg.config['traces']['green'])

        # df_training_tracklets_fname = train_cfg.resolve_relative_path_from_config('df_training_3d_tracks')
        # reindexed_masks_training_fname = train_cfg.resolve_relative_path_from_config('reindexed_masks')

        final_tracks_fname = tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
        seg_fname_raw = segment_cfg.resolve_relative_path_from_config('output_masks')
        seg_fname = traces_cfg.resolve_relative_path_from_config('reindexed_masks')

        # Metadata uses class from segmentation package, which does lazy loading itself
        seg_metadata_fname = segment_cfg.resolve_relative_path_from_config('output_metadata')
        obj.segmentation_metadata = DetectedNeurons(seg_metadata_fname)

        obj.physical_unit_conversion = cfg.get_physical_unit_conversion_class()

        # Read ahead of time because they may be needed for classes in the threading environment
        _ = obj.final_tracks

        behavior_reader = lambda: WormFullVideoPosture.load_from_config(cfg)
        zarr_reader_readwrite = lambda fname: zarr.open(fname, mode='r+')

        # Note: when running on the cluster the raw data isn't (for now) accessible
        with safe_cd(cfg.project_dir):

            cfg.logger.info("Starting threads to read data...")
            with concurrent.futures.ThreadPoolExecutor() as ex:
                if to_load_tracklets:
                    ex.submit(obj.load_tracklet_related_properties)
                if to_load_interactivity:
                    ex.submit(obj.load_interactive_properties)
                if to_load_frames:
                    ex.submit(obj.load_frame_related_properties)
                if to_load_segmentation_metadata:
                    ex.submit(obj.load_segmentation_related_properties)
                red_data = ex.submit(read_if_exists, red_dat_fname, zarr_reader_folder_or_zipstore).result()
                green_data = ex.submit(read_if_exists, green_dat_fname, zarr_reader_folder_or_zipstore).result()
                red_traces = ex.submit(read_if_exists, red_traces_fname).result()
                green_traces = ex.submit(read_if_exists, green_traces_fname).result()
                # TODO: don't open this as read-write by default
                raw_segmentation = ex.submit(read_if_exists, seg_fname_raw, zarr_reader_readwrite).result()
                segmentation = ex.submit(read_if_exists, seg_fname, zarr_reader_folder_or_zipstore).result()
                worm_posture_class = ex.submit(behavior_reader).result()

            if red_traces is not None:
                red_traces.replace(0, np.nan, inplace=True)
                green_traces.replace(0, np.nan, inplace=True)

        obj.all_used_fnames.extend([red_dat_fname, green_dat_fname, red_traces_fname, green_traces_fname,
                                    seg_fname_raw, seg_fname])
        cfg.logger.info(f"Read all data from files: {obj.all_used_fnames}")

        background_per_pixel = preprocessing_settings.background_per_pixel
        likelihood_thresh = traces_cfg.config['visualization']['likelihood_thresh']

        # Return a full object
        obj.red_data = red_data
        obj.green_data = green_data
        obj.raw_segmentation = raw_segmentation
        obj.segmentation = segmentation
        obj.red_traces = red_traces
        obj.green_traces = green_traces
        obj.worm_posture_class = worm_posture_class
        obj.background_per_pixel = background_per_pixel
        obj.likelihood_thresh = likelihood_thresh
        cfg.logger.info(obj)
        return obj

    @staticmethod
    def load_final_project_data_from_config(project_path: Union[str, os.PathLike, ModularProjectConfig],
                                            **kwargs):
        """
        Main constructor that accepts multiple input formats
        This includes an already initialized ProjectData class, in which case this function returns

        valid kwargs are:
            to_load_tracklets=False,
            to_load_interactivity=False,
            to_load_frames=False,
            to_load_segmentation_metadata=False,
            initialization_kwargs=None
        """
        if isinstance(project_path, (str, os.PathLike)):
            if Path(project_path).is_dir():
                project_path = Path(project_path).joinpath('project_config.yaml')
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args, **kwargs)
        elif isinstance(project_path, ModularProjectConfig):
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args, **kwargs)
        elif isinstance(project_path, ProjectData):
            return project_path
        else:
            raise TypeError("Must pass pathlike or already loaded project data")

    def calculate_traces(self, channel_mode: str,
                         calculation_mode: str,
                         neuron_name: str,
                         remove_outliers: bool = False,
                         filter_mode: str = 'no_filtering',
                         min_confidence: float = None) -> Tuple[list, list]:
        """
        Uses TracePlotter class to calculate traces

        In other words, creates a class, then uses a method from that class

        Parameters
        ----------
        channel_mode - red, green, or ratio
        calculation_mode - integration (raw sum of pixels), volume, mean, z, likelihood (from the tracks dataframe)
        neuron_name - example: 'neuron_001'
        remove_outliers - try to remove spiking outliers
        filter_mode - try to filter; not currently working
        min_confidence - if confidence below this, plot a gap

        Returns
        -------

        time (as vector), y (as vector)

        """
        # Todo: don't recreate object every time
        self._trace_plotter = TracePlotter(
            self.red_traces,
            self.green_traces,
            self.final_tracks,
            channel_mode,
            calculation_mode,
            remove_outliers,
            filter_mode,
            min_confidence,
            self.background_per_pixel
        )
        y = self._trace_plotter.calculate_traces(neuron_name)
        return self._trace_plotter.tspan, y

    def plot_neuron_with_kymograph(self, neuron_name):
        t, y = self.calculate_traces(channel_mode='ratio', calculation_mode='integration',
                                     neuron_name=neuron_name)
        df_kymo = self.worm_posture_class.curvature_fluorescence_fps

        fig, axes = plt.subplots(nrows=2, figsize=(30, 10), sharex=True)
        axes[0].imshow(df_kymo.T, origin="upper", cmap='seismic', extent=[0, df_kymo.shape[0], df_kymo.shape[1], 0],
                       aspect='auto', vmin=-0.06, vmax=0.06)
        axes[0].set_ylabel("Segment (0=nose)")
        axes[1].plot(t, y)
        axes[1].set_ylabel("Ratio (green/red)")
        plt.xlabel("Time (frames)")
        plt.xlim(0, self.num_frames)
        self.shade_axis_using_behavior()

    def save_fig_in_project(self, suffix=''):
        out_fname = f'fig-{suffix}.png'
        foldername = self.project_config.get_visualization_dir()
        out_fname = os.path.join(foldername, out_fname)
        out_fname = get_sequential_filename(out_fname)

        plt.savefig(out_fname)

    def calculate_tracklets(self, neuron_name) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, str]:
        """
        Calculates tracklets using the tracklet_annotator class

        Returns all tracklets already attached, as well as any currently selected tracklets
        """
        y_dict, y_current, y_current_name = self.tracklet_annotator.calculate_tracklets_for_neuron(neuron_name)
        return y_dict, y_current, y_current_name

    def modify_confidences_of_frame_pair(self, pair, gamma, mode) -> list:
        """
        Postprocessing function to be applied to frame pairs that incorporates similarity in image space

        Parameters
        ----------
        pair
        gamma
        mode

        Returns
        -------

        """
        frame_match = self.raw_matches[pair]

        matches = frame_match.modify_confidences_using_image_features(self.segmentation_metadata,
                                                                      gamma=gamma,
                                                                      mode=mode)
        frame_match.final_matches = matches
        return matches

    def modify_confidences_of_all_frame_pairs(self, gamma, mode):
        """
        Loops modify_confidences_of_frame_pair across all frames

        Parameters
        ----------
        gamma
        mode

        Returns
        -------

        """
        frame_matches = self.raw_matches
        opt = dict(metadata=self.segmentation_metadata, gamma=gamma, mode=mode)
        for pair, obj in frame_matches.items():
            matches = obj.modify_confidences_using_image_features(**opt)
            obj.final_matches = matches

    def modify_segmentation_using_manual_correction(self, t=None, new_mask=None):
        """
        Modifies single segmentation, but does NOT update the disk, but rather a buffer zarr array in the
        tracklet_annotator class

        NOTE: will invalidate raw_frames and raw_matches!!

        Parameters
        ----------
        t
        new_mask

        Returns
        -------

        """
        # TODO: save the list of split neurons in separate pickle
        if new_mask is None or t is None:
            new_mask = self.tracklet_annotator.candidate_mask
            t = self.tracklet_annotator.time_of_candidate
        if new_mask is None:
            self.logger.warning("Modification attempted, but no valid candidate mask exists; aborting")
            self.logger.warning("HINT: if you produce a mask but then click different neurons, it invalidates the mask!")
            return
        affected_masks = self.tracklet_annotator.indices_of_original_neurons
        # this_seg = self.raw_segmentation[t, ...]
        # affected_masks = np.unique(this_seg[(this_seg - new_mask) != 0])

        self.logger.info(f"Updating raw segmentation at t = {t}; affected masks={affected_masks}")
        self.tracklet_annotator.modify_buffer_segmentation(t, new_mask)
        # self.raw_segmentation[t, ...] = new_mask

        self.logger.info(f"Updating metadata at t, but NOT writing to disk...")
        red_volume = self.red_data[t, ...]
        self.segmentation_metadata.modify_segmentation_metadata(t, new_mask, red_volume)

        self.logger.info("Updating affected tracklets, but NOT writing to disk")
        for m in affected_masks:
            # Explicitly check to see if there actually was a tracklet before the segmentation was changed
            # Note that this metadata refers to the old masks, even if the mask is deleted above
            tracklet_name = self.tracklets_and_neurons_class.get_tracklet_from_segmentation_index(t, m)
            if tracklet_name is not None:
                self.tracklets_and_neurons_class.update_tracklet_metadata_using_segmentation_metadata(
                    t, tracklet_name=tracklet_name, mask_ind=m, likelihood=1.0, verbose=1
                )
                self.logger.info(f"Updating {tracklet_name} corresponding to segmentation {m}")
            else:
                self.logger.info(f"No tracklet corresponding to segmentation {m}; not updated")
        self.logger.debug("Segmentation and tracklet metadata modified successfully")

    def modify_segmentation_on_disk_using_buffer(self):
        """
        Modifies single segmentation ON DISK using tracklet_annotator class

        NOTE: invalidates raw_frames and raw_matches!!

        See: modify_segmentation_using_manual_correction

        """

        for t in self.tracklet_annotator.t_buffer_masks:
            self.raw_segmentation[t, ...] = self.tracklet_annotator.buffer_masks[t, ...]

    def shade_axis_using_behavior(self, ax=None, behaviors_to_ignore='none'):
        """
        Shades the currently active matplotlib axis using externally annotated behavior annotation

        Note: only works if self.behavior_annotations is found

        Example with a loaded project:

        t, y = project_data.calculate_traces(channel_mode='ratio',
                                             calculation_mode='integration',
                                             neuron_name='neuron_001')
        plt.plot(t, y)
        project_data.shade_axis_using_behavior()
        plt.show()

        Parameters
        ----------
        ax - optional; takes the current one by default
        behaviors_to_ignore - integer of behaviors to ignore when shading

        Returns
        -------

        """
        self.worm_posture_class.shade_using_behavior(ax=ax, behaviors_to_ignore=behaviors_to_ignore)

    def get_centroids_as_numpy(self, i_frame):
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        return self.segmentation_metadata.detect_neurons_from_file(i_frame)

    def get_centroids_as_numpy_training(self, i_frame: int, is_relative_index=True) -> np.ndarray:
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        assert is_relative_index, "Only relative supported"

        return dataframe_to_numpy_zxy_single_frame(self.df_training_tracklets, t=i_frame)

    def get_distance_to_closest_neuron(self, i_frame, target_pt, nbr_obj=None) -> float:
        """
        Get the 3d distance between a target neuron and its closest neighbor

        Parameters
        ----------
        i_frame
        target_pt
        nbr_obj

        Returns
        -------

        """
        # TODO: refactor to segmentation class?
        if nbr_obj is None:
            # TODO: cache these neighbor objects?
            segmented_pts = self.get_centroids_as_numpy(i_frame)
            nbr_obj = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(segmented_pts)

        # Get point
        # target_pt = df_interp2[which_neuron].iloc[i_frame].to_numpy()[:3]

        # Get closest neighbor
        if not any(np.isnan(target_pt)):
            imputed_dist, _ = nbr_obj.kneighbors([target_pt], n_neighbors=1)
            dist = imputed_dist[0][0]
        else:
            dist = np.nan

        return dist

    def correct_relative_training_index(self, i) -> int:
        """Converts a relative index within the training set into a global index"""
        return self.which_training_frames[i]

    def napari_of_single_match(self, pair, which_matches='final_matches', this_match: FramePair = None,
                               rigidly_align_volumetric_images=False, min_confidence=0.0) -> napari.Viewer:
        """
        Visualize the matches between two volumes, including the raw segmentation

        Parameters
        ----------
        pair
        which_matches
        this_match
        rigidly_align_volumetric_images
        min_confidence

        Returns
        -------

        """
        from wbfm.utils.visualization.napari_from_project_data_class import NapariLayerInitializer
        v = NapariLayerInitializer.napari_of_single_match(self, pair, which_matches, this_match,
                                                          rigidly_align_volumetric_images, min_confidence)
        return v

    def napari_tracks_layer_of_single_neuron_match(self, neuron_name, t):
        neuron_ind_in_list = self.final_tracks.loc[t, (neuron_name, 'raw_neuron_ind_in_list')]
        if np.isnan(neuron_ind_in_list):
            self.logger.debug(f"No match for {neuron_name} at t={t}")
        else:
            neuron_ind_in_list = int(neuron_ind_in_list)

        this_pair = self.raw_matches[(t, t + 1)]
        list_of_matches = this_pair.final_matches
        matches_class = MatchesWithConfidence.matches_from_array(list_of_matches)
        # If match is not found, use -1
        this_match = [[neuron_ind_in_list, matches_class.get_mapping_0_to_1(unique=True).get(neuron_ind_in_list, -1)]]
        tracks = this_pair.napari_tracks_of_matches(this_match)
        return tracks

    def add_layers_to_viewer(self, viewer=None, which_layers: Union[str, List[str]] = 'all',
                             to_remove_flyback=False, check_if_layers_exist=False,
                             dask_for_segmentation=True) -> napari.Viewer:
        """
        Add layers corresponding to any analysis steps to a napari viewer object

        If no viewer is passed, then this creates a new one.

        By default, these layers are added to the viewer:
            ['Red data',
            'Green data',
            'Raw segmentation',
            'Colored segmentation',
            'Neuron IDs',
            'Intermediate global IDs']

        An additional option that is not added:
            'GT IDs' (only exists if ground truth annotation is present)

        Additional special type, which should be passed as a tuple:
            ('heatmap', 'count_nonnan')
            ('heatmap', 'max_of_red')
            ('heatmap', 'max_of_green')
            ('heatmap', 'std_of_green')

        See: NapariPropertyHeatMapper for up-to-date list of options (all methods are valid)

        Parameters
        ----------
        viewer
        which_layers
        to_remove_flyback
        check_if_layers_exist
        dask_for_segmentation

        Returns
        -------

        """
        from wbfm.utils.visualization.napari_from_project_data_class import NapariLayerInitializer
        v = NapariLayerInitializer.add_layers_to_viewer(self, viewer, which_layers,
                                                        to_remove_flyback, check_if_layers_exist,
                                                        dask_for_segmentation)
        return v

    def get_desynced_seg_and_frame_object_frames(self, verbose=1) -> List[int]:
        """Return frame objects that are obviously desynced from the segmentation"""
        desynced_frames = []
        for t in range(self.num_frames):
            pts_from_seg = self.get_centroids_as_numpy(t)
            pts_from_frame = self.raw_frames[t].neuron_locs

            if pts_from_seg.shape != pts_from_frame.shape:
                desynced_frames.append(t)
        if verbose >= 1:
            self.logger.warning(f"Found {len(desynced_frames)} desynchronized frames")
        return desynced_frames

    @cached_property
    def df_manual_tracking(self) -> pd.DataFrame:
        """Load a dataframe corresponding to manual tracking, i.e. which neurons have been manually corrected"""
        # TODO: do not hardcode
        track_cfg = self.project_config.get_tracking_config()
        fname = track_cfg.resolve_relative_path("manual_annotation/manual_tracking.csv", prepend_subfolder=True)
        df_manual_tracking = read_if_exists(fname, reader=pd.read_csv)
        # df_manual_tracking = pd.read_csv(fname)
        return df_manual_tracking

    @cached_property
    def finished_neuron_names(self) -> List[str]:
        """
        Uses df_manual_tracking to get a list of the neuron names that have been fully corrected
        """
        df_manual_tracking = self.df_manual_tracking
        if df_manual_tracking is None:
            return []

        try:
            neurons_finished_mask = df_manual_tracking[self.finished_neurons_column_name]
            if neurons_finished_mask.dtype != bool:
                self.logger.warning("Found non-boolean entries in manual annotation column; this may be a data error: "
                                    f"{np.unique(neurons_finished_mask)}")
                neurons_finished_mask = neurons_finished_mask.astype(bool)
            neurons_that_are_finished = list(df_manual_tracking[neurons_finished_mask]['Neuron ID'])

            # Filter to make sure they are the proper format
            tmp = []
            for col_name in neurons_that_are_finished:
                if isinstance(col_name, str):
                    tmp.append(col_name)
                else:
                    self.logger.warning(f"Found and removed improper column name in manual annotation : {col_name}")
            neurons_that_are_finished = tmp
        except KeyError:
            neurons_that_are_finished = []

        return neurons_that_are_finished

    def __repr__(self):
        return f"=======================================\n\
Project data for directory:\n\
{self.project_dir} \n\
Found the following data files:\n\
============Raw========================\n\
red_data:                 {self.red_data is not None}\n\
green_data:               {self.green_data is not None}\n\
============Annotations================\n\
manual_tracking:          {self.df_manual_tracking is not None}\n\
============Segmentation===============\n\
raw_segmentation:         {self.raw_segmentation is not None}\n\
colored_segmentation:     {self.segmentation is not None}\n\
============Traces=====================\n\
red_traces:               {self.red_traces is not None}\n\
green_traces:             {self.green_traces is not None}\n"


def napari_of_training_data(cfg: ModularProjectConfig) -> Tuple[napari.Viewer, np.ndarray, np.ndarray]:

    project_data = ProjectData.load_final_project_data_from_config(cfg)
    training_cfg = cfg.get_training_config()

    z_dat = project_data.red_data
    raw_seg = project_data.raw_segmentation
    z_seg = project_data.reindexed_masks_training

    # Training data doesn't usually start at i=0, so align
    num_frames = training_cfg.config['training_data_3d']['num_training_frames']
    i_seg_start = training_cfg.config['training_data_3d']['which_frames'][0]
    i_seg_end = i_seg_start + num_frames
    z_dat = z_dat[i_seg_start:i_seg_end, ...]
    raw_seg = raw_seg[i_seg_start:i_seg_end, ...]

    cfg.logger.info(f"Size of reindexed_masks: {z_dat.shape}")

    viewer = napari.view_labels(z_seg, ndisplay=3)
    viewer.add_labels(raw_seg, visible=False)
    viewer.add_image(z_dat)
    viewer.show()

    return viewer, z_dat, z_seg


def template_matches_to_dataframe(project_data: ProjectData,
                                  all_matches: list,
                                  null_value=-1):
    """Correct null value within all_matches is []"""
    num_frames = len(all_matches)
    coords = ['z', 'x', 'y', 'likelihood', 'raw_neuron_ind_in_list']
    sz = (num_frames, len(coords))
    neuron_arrays = defaultdict(lambda: np.zeros(sz))

    for i_frame, these_matches in enumerate(tqdm(all_matches, leave=False)):
        pts = project_data.get_centroids_as_numpy(i_frame)
        # For each match, save location
        for m in these_matches:
            this_unscaled_pt = pts[m[1]]
            this_template_idx = m[0]

            # These columns must match the order of 'coords' above
            if null_value in m:
                neuron_arrays[this_template_idx][i_frame, :] = np.nan
            else:
                neuron_arrays[this_template_idx][i_frame, :3] = this_unscaled_pt
                neuron_arrays[this_template_idx][i_frame, 3] = m[2]  # Match confidence
                neuron_arrays[this_template_idx][i_frame, 4] = m[1]  # Match index

    # Convert to pandas multiindexing formatting
    new_dict = {}
    for i_template, data in neuron_arrays.items():
        for i_col, coord_name in enumerate(coords):
            # NOTE: these neuron names are final for all subsequent steps
            k = (int2name_neuron(i_template + 1), coord_name)
            new_dict[k] = data[:, i_col]

    df = pd.DataFrame(new_dict)

    return df


def calc_mismatch_between_ground_truth_and_pairs(all_matches: Dict[tuple, FramePair],
                                                 df_gt: pd.DataFrame,
                                                 t0: int,
                                                 minimum_confidence: float):
    pair = (t0, t0 + 1)

    model_matches = all_matches[pair].final_matches
    # model_matches = all_matches[pair].feature_matches
    gt_matches = df_to_matches(df_gt, t0)

    model_obj = MatchesWithConfidence.matches_from_array(model_matches, minimum_confidence=minimum_confidence)
    gt_obj = MatchesWithConfidence.matches_from_array(gt_matches, 1)
    correct_matches, gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model = \
        get_mismatches(gt_obj, model_obj)

    return gt_matches_different_model, model_matches_different_gt, model_matches_no_gt, gt_matches_no_model


def calc_all_mismatches_between_ground_truth_and_pairs(project_data: ProjectData,
                                                       minimum_confidence=0.0) -> Dict[str, list]:
    """
    Calculates all mismatches between the ground truth and the matches class

    Return a dictionary indexed by a tuple:
        (time, neuron_name, tracklet_name) = bool

    (In principle this could be a list)

    Parameters
    ----------
    minimum_confidence
    project_data

    Returns
    -------

    """
    all_mismatches = defaultdict(list)
    all_matches = project_data.raw_matches
    df_gt, finished_neurons = project_data.get_final_tracks_only_finished_neurons()
    if len(finished_neurons) == 0:
        return all_mismatches

    project_data.logger.info("Calculating mismatches between ground truth and automatic matches")
    for t0 in tqdm(range(project_data.num_frames - 2)):
        # Only need the first of the pair here, i.e. the index on t0, not t0+1
        try:
            gt_matches_different_model, model_matches_different_gt, _, _ = \
                calc_mismatch_between_ground_truth_and_pairs(all_matches, df_gt, t0,
                                                             minimum_confidence=minimum_confidence)
        except NoMatchesError:
            continue

        if gt_matches_different_model:
            for gt_mismatch, model_mismatch in zip(gt_matches_different_model, model_matches_different_gt):
                raw_neuron_ind_in_list = gt_mismatch[0]
                ind, neuron_name = get_column_name_from_time_and_column_value(df_gt, t0, raw_neuron_ind_in_list,
                                                                              col_name='raw_neuron_ind_in_list')

                # Convert to segmentation to use the tracklet class pre-allocated dict
                mask_ind = project_data.segmentation_metadata.i_in_array_to_mask_index(t0, raw_neuron_ind_in_list)
                tracklet_name = project_data.tracklets_and_neurons_class.get_tracklet_from_segmentation_index(t0,
                                                                                                              mask_ind)
                all_mismatches[neuron_name].append((t0, tracklet_name, model_mismatch, gt_mismatch))

    logging.info(f"Found {sum(map(len, all_mismatches.values()))} mismatches")

    return all_mismatches
