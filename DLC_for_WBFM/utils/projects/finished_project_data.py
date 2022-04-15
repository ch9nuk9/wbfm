import concurrent
import logging
from collections import defaultdict

from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name_neuron

logger = logging.getLogger('projectDataLogger')
logger.setLevel(logging.INFO)
import os
from dataclasses import dataclass
from typing import Tuple, List

import dask.array as da
import napari
import numpy as np
import pandas as pd
import zarr
from tqdm.auto import tqdm

from DLC_for_WBFM.utils.external.utils_pandas import dataframe_to_numpy_zxy_single_frame, check_if_fully_sparse
from DLC_for_WBFM.utils.neuron_matching.class_frame_pair import FramePair
from DLC_for_WBFM.utils.projects.physical_units import PhysicalUnitConversion
from DLC_for_WBFM.utils.tracklets.utils_tracklets import fix_global2tracklet_full_dict, check_for_unmatched_tracklets
from sklearn.neighbors import NearestNeighbors
from DLC_for_WBFM.utils.tracklets.tracklet_class import DetectedTrackletsAndNeurons
from DLC_for_WBFM.utils.projects.plotting_classes import TracePlotter, TrackletAndSegmentationAnnotator
from DLC_for_WBFM.utils.visualization.napari_from_config import napari_labels_from_frames
from DLC_for_WBFM.utils.visualization.napari_utils import napari_labels_from_traces_dataframe
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior
from segmentation.util.utils_metadata import DetectedNeurons
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from DLC_for_WBFM.utils.projects.utils_filenames import read_if_exists, pickle_load_binary, \
    load_file_according_to_precedence, pandas_read_any_filetype
from DLC_for_WBFM.utils.projects.utils_project import safe_cd
# from functools import cached_property # Only from python>=3.8
from backports.cached_property import cached_property


@dataclass
class ProjectData:
    project_dir: str
    project_config: ModularProjectConfig

    red_data: zarr.Array = None
    green_data: zarr.Array = None

    raw_segmentation: zarr.Array = None
    segmentation: zarr.Array = None
    segmentation_metadata: DetectedNeurons = None

    df_training_tracklets: pd.DataFrame = None
    reindexed_masks_training: zarr.Array = None

    red_traces: pd.DataFrame = None
    green_traces: pd.DataFrame = None
    # final_tracks: pd.DataFrame = None

    behavior_annotations: pd.DataFrame = None
    background_per_pixel: float = None
    likelihood_thresh: float = None

    all_used_fnames: list = None
    verbose: int = 2

    # Precedence when multiple are available
    precedence_global2tracklet: list = None
    precedence_df_tracklets: list = None
    precedence_tracks: list = None

    # Will be set as loaded
    final_tracks_fname: str = None
    global2tracklet_fname: str = None
    df_all_tracklets_fname: str = None
    force_tracklets_to_be_sparse: bool = True  # TODO: pass as arg

    # Classes for more functionality
    trace_plotter: TracePlotter = None
    physical_unit_conversion: PhysicalUnitConversion = None

    # Values for ground truth annotation (reading from excel or .csv)
    finished_neurons_column_name: str = "Finished?"

    def __post_init__(self):
        track_cfg = self.project_config.get_tracking_config()
        if self.precedence_global2tracklet is None:
            self.precedence_global2tracklet = track_cfg.config['precedence_global2tracklet']
        if self.precedence_df_tracklets is None:
            self.precedence_df_tracklets = track_cfg.config['precedence_df_tracklets']
        if self.precedence_tracks is None:
            self.precedence_tracks = track_cfg.config['precedence_tracks']

    @cached_property
    def intermediate_global_tracks(self) -> pd.DataFrame:
        tracking_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        fname = tracking_cfg.config['leifer_params']['output_df_fname']
        fname = tracking_cfg.resolve_relative_path(fname, prepend_subfolder=False)

        global_tracks = read_if_exists(fname)
        return global_tracks

    @cached_property
    def final_tracks(self) -> pd.DataFrame:
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

    def get_final_tracks_only_finished_neurons(self, finished_neurons_column_name=None) -> pd.DataFrame:
        """See get_ground_truth_annotations()"""
        if finished_neurons_column_name is None:
            finished_neurons_column_name = self.finished_neurons_column_name
        df_gt = self.final_tracks
        _, df_manual_tracking = self.get_ground_truth_annotations()
        finished_neurons = list(
            df_manual_tracking[df_manual_tracking[finished_neurons_column_name]]['Neuron ID'])

        return df_gt.loc[:, finished_neurons]

    @cached_property
    def global2tracklet(self) -> dict:
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
    def raw_frames(self):
        logger.info("First time loading the raw frames, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'frame_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        frames = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return frames

    @cached_property
    def raw_matches(self) -> List[FramePair]:
        logger.info("First time loading the raw matches, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'match_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        matches = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return matches

    @cached_property
    def raw_clust(self):
        logger.info("First time loading the raw cluster dataframe, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'clust_df_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        clust = pickle_load_binary(fname)
        self.all_used_fnames.append(fname)
        return clust

    @cached_property
    def df_all_tracklets(self):
        logger.info("First time loading the all tracklets, may take a while...")
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
            if not check_if_fully_sparse(df_all_tracklets):
                logger.warning("Casting tracklets as sparse, may take a minute")
                # df_all_tracklets = to_sparse_multiindex(df_all_tracklets)
                df_all_tracklets = df_all_tracklets.astype(pd.SparseDtype("float", np.nan))
            else:
                logger.info("Found sparse matrix")
        logging.info("Finished loading tracklets")

        return df_all_tracklets

    @cached_property
    def tracklet_annotator(self):
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
            buffer_masks=zarr.zeros_like(self.segmentation)
        )
        return obj

    @cached_property
    def tracklets_and_neurons_class(self):
        _ = self.df_all_tracklets  # Make sure it is loaded
        return DetectedTrackletsAndNeurons(self.df_all_tracklets, self.segmentation_metadata,
                                           dataframe_output_filename=self.df_all_tracklets_fname)

    @cached_property
    def df_fdnc_tracks(self):
        train_cfg = self.project_config.get_tracking_config()
        fname = os.path.join('postprocessing', 'leifer_tracks.h5')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        df_fdnc_tracks = read_if_exists(fname)
        return df_fdnc_tracks

    def _load_tracklet_related_properties(self):
        _ = self.df_all_tracklets
        _ = self.raw_clust

    def _load_interactive_properties(self):
        _ = self.tracklet_annotator

    def _load_frame_related_properties(self):
        _ = self.raw_frames
        _ = self.raw_matches

    def _load_segmentation_related_properties(self):
        _ = self.segmentation_metadata.segmentation_metadata
        self.all_used_fnames.append(self.segmentation_metadata.segmentation_metadata.detection_fname)

    @property
    def num_frames(self):
        return self.project_config.config['dataset_params']['num_frames']

    @property
    def which_training_frames(self):
        train_cfg = self.project_config.get_training_config()
        return train_cfg.config['training_data_3d']['which_frames']

    @property
    def num_training_frames(self):
        return len(self.which_training_frames)

    def check_data_desyncing(self, raise_error=True):
        logging.info("Checking for database desynchronization")
        unmatched_tracklets = check_for_unmatched_tracklets(self.df_all_tracklets, self.global2tracklet,
                                                            raise_error=raise_error)

    @staticmethod
    def unpack_config_file(project_path):
        cfg = ModularProjectConfig(project_path)
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
                                to_load_segmentation_metadata=False):
        # Initialize object in order to use cached properties
        obj = ProjectData(project_dir, cfg)
        obj.all_used_fnames = []

        red_dat_fname = cfg.config['preprocessed_red']
        green_dat_fname = cfg.config['preprocessed_green']
        red_traces_fname = traces_cfg.resolve_relative_path(traces_cfg.config['traces']['red'])
        green_traces_fname = traces_cfg.resolve_relative_path(traces_cfg.config['traces']['green'])

        df_training_tracklets_fname = train_cfg.resolve_relative_path_from_config('df_training_3d_tracks')
        reindexed_masks_training_fname = train_cfg.resolve_relative_path_from_config('reindexed_masks')

        final_tracks_fname = tracking_cfg.resolve_relative_path_from_config('final_3d_tracks_df')
        seg_fname_raw = segment_cfg.resolve_relative_path_from_config('output_masks')
        seg_fname = traces_cfg.resolve_relative_path_from_config('reindexed_masks')

        # Metadata uses class from segmentation package, which does lazy loading itself
        seg_metadata_fname = segment_cfg.resolve_relative_path_from_config('output_metadata')
        obj.segmentation_metadata = DetectedNeurons(seg_metadata_fname)

        obj.physical_unit_conversion = cfg.get_physical_unit_conversion_class()

        # Read ahead of time because they may be needed for classes in the threading environment
        _ = obj.final_tracks
        # obj.final_tracks = read_if_exists(final_tracks_fname)

        # TODO: do not hardcode
        behavior_fname = "3-tracking/manual_annotation/manual_behavior_annotation.xlsx"
        behavior_fname = cfg.resolve_relative_path(behavior_fname)
        if not os.path.exists(behavior_fname):
            behavior_fname = "3-tracking/postprocessing/manual_behavior_annotation.xlsx"
            behavior_fname = cfg.resolve_relative_path(behavior_fname)

        zarr_reader_readonly = lambda fname: zarr.open(fname, mode='r')
        zarr_reader_readwrite = lambda fname: zarr.open(fname, mode='r+')
        excel_reader = lambda fname: pd.read_excel(fname, sheet_name='behavior')['Annotation']

        # Note: when running on the cluster the raw data isn't (for now) accessible
        with safe_cd(cfg.project_dir):

            logger.info("Starting threads to read data...")
            with concurrent.futures.ThreadPoolExecutor() as ex:
                if to_load_tracklets:
                    ex.submit(obj._load_tracklet_related_properties)
                if to_load_interactivity:
                    ex.submit(obj._load_interactive_properties)
                if to_load_frames:
                    ex.submit(obj._load_frame_related_properties)
                if to_load_segmentation_metadata:
                    ex.submit(obj._load_segmentation_related_properties)
                red_data = ex.submit(read_if_exists, red_dat_fname, zarr_reader_readonly).result()
                green_data = ex.submit(read_if_exists, green_dat_fname, zarr_reader_readonly).result()
                red_traces = ex.submit(read_if_exists, red_traces_fname).result()
                green_traces = ex.submit(read_if_exists, green_traces_fname).result()
                df_training_tracklets = ex.submit(read_if_exists, df_training_tracklets_fname).result()
                reindexed_masks_training = ex.submit(read_if_exists, reindexed_masks_training_fname, zarr_reader_readonly).result()
                # reindexed_metadata_training = ex.submit(read_if_exists,
                #                                         reindexed_metadata_training_fname, pickle_load_binary).result()
                # final_tracks = ex.submit(read_if_exists, final_tracks_fname).result()
                # TODO: don't open this as read-write by default
                raw_segmentation = ex.submit(read_if_exists, seg_fname_raw, zarr_reader_readwrite).result()
                segmentation = ex.submit(read_if_exists, seg_fname, zarr_reader_readonly).result()
                # seg_metadata: dict = ex.submit(pickle_load_binary, seg_metadata_fname).result()
                behavior_annotations = ex.submit(read_if_exists, behavior_fname, excel_reader).result()

            if red_traces is not None:
                red_traces.replace(0, np.nan, inplace=True)
                green_traces.replace(0, np.nan, inplace=True)
            logger.info("Read all data")

        obj.all_used_fnames.extend([red_dat_fname, green_dat_fname, red_traces_fname, green_traces_fname,
                                    df_training_tracklets_fname, reindexed_masks_training_fname,
                                    seg_fname_raw, seg_fname, behavior_fname])

        background_per_pixel = traces_cfg.config['visualization']['background_per_pixel']
        likelihood_thresh = traces_cfg.config['visualization']['likelihood_thresh']

        # Return a full object
        obj.red_data = red_data
        obj.green_data = green_data
        obj.raw_segmentation = raw_segmentation
        obj.segmentation = segmentation
        obj.df_training_tracklets = df_training_tracklets
        obj.reindexed_masks_training = reindexed_masks_training
        obj.red_traces = red_traces
        obj.green_traces = green_traces
        obj.behavior_annotations = behavior_annotations
        obj.background_per_pixel = background_per_pixel
        obj.likelihood_thresh = likelihood_thresh
        print(obj)

        return obj

    @staticmethod
    def load_final_project_data_from_config(project_path, **kwargs):
        if isinstance(project_path, (str, os.PathLike)):
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args, **kwargs)
        elif isinstance(project_path, ModularProjectConfig):
            project_path = project_path.self_path
            args = ProjectData.unpack_config_file(project_path)
            return ProjectData._load_data_from_configs(*args, **kwargs)
        elif isinstance(project_path, ProjectData):
            return project_path
        else:
            raise TypeError("Must pass pathlike or already loaded project data")

    def calculate_traces(self, channel_mode: str, calculation_mode: str, neuron_name: str,
                         remove_outliers: bool = False,
                         filter_mode: str = 'no_filtering',
                         min_confidence: float = None):
        # Todo: don't recreate object every time
        self.trace_plotter = TracePlotter(
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
        y = self.trace_plotter.calculate_traces(neuron_name)
        return self.trace_plotter.tspan, y

    def calculate_tracklets(self, neuron_name):
        y, y_current = self.tracklet_annotator.calculate_tracklets_for_neuron(neuron_name)
        return y, y_current

    def modify_confidences_of_frame_pair(self, pair, gamma, mode):
        frame_match = self.raw_matches[pair]

        matches = frame_match.modify_confidences_using_image_features(self.segmentation_metadata,
                                                                      gamma=gamma,
                                                                      mode=mode)
        frame_match.final_matches = matches
        return matches

    def modify_confidences_of_all_frame_pairs(self, gamma, mode):
        frame_matches = self.raw_matches
        opt = dict(metadata=self.segmentation_metadata, gamma=gamma, mode=mode)
        for pair, obj in frame_matches.items():
            matches = obj.modify_confidences_using_image_features(**opt)
            obj.final_matches = matches

    def modify_segmentation_using_manual_correction(self, t=None, new_mask=None):
        # TODO: save the list of split neurons in separate pickle
        if new_mask is None or t is None:
            new_mask = self.tracklet_annotator.candidate_mask
            t = self.tracklet_annotator.time_of_candidate
        if new_mask is None:
            logger.warning("Modification attempted, but no valid candidate mask exists; aborting")
            logger.warning("HINT: if you produce a mask but then click different neurons, it invalidates the mask!")
            return
        affected_masks = self.tracklet_annotator.indices_of_original_neurons
        # this_seg = self.raw_segmentation[t, ...]
        # affected_masks = np.unique(this_seg[(this_seg - new_mask) != 0])

        print(f"Updating raw segmentation at t = {t}; affected masks={affected_masks}")
        self.tracklet_annotator.modify_buffer_segmentation(t, new_mask)
        # self.raw_segmentation[t, ...] = new_mask

        print(f"Updating metadata at t, but NOT writing to disk...")
        red_volume = self.red_data[t, ...]
        self.segmentation_metadata.modify_segmentation_metadata(t, new_mask, red_volume)

        print("Updating affected tracklets, but NOT writing to disk")
        for m in affected_masks:
            # Explicitly check to see if there actually was a tracklet before the segmentation was changed
            # Note that this metadata refers to the old masks, even if the mask is deleted above
            tracklet_name = self.tracklets_and_neurons_class.get_tracklet_from_segmentation_index(t, m)
            if tracklet_name is not None:
                self.tracklets_and_neurons_class.update_tracklet_metadata_using_segmentation_metadata(
                    t, tracklet_name=tracklet_name, mask_ind=m, likelihood=1.0, verbose=1
                )
                print(f"Updating {tracklet_name} corresponding to segmentation {m}")
            else:
                print(f"No tracklet corresponding to segmentation {m}; not updated")
        logger.debug("Segmentation and tracklet metadata modified successfully")

    def modify_segmentation_on_disk_using_buffer(self):
        for t in self.tracklet_annotator.t_buffer_masks:
            self.raw_segmentation[t, ...] = self.tracklet_annotator.buffer_masks[t, ...]

    def shade_axis_using_behavior(self, ax=None, behaviors_to_ignore='none'):
        if self.behavior_annotations is None:
            pass
            # logger.warning("No behavior annotations present; skipping")
        else:
            shade_using_behavior(self.behavior_annotations, ax, behaviors_to_ignore)

    def get_centroids_as_numpy(self, i_frame):
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        return self.segmentation_metadata.detect_neurons_from_file(i_frame)

    def get_centroids_as_numpy_training(self, i_frame: int, is_relative_index=True) -> np.ndarray:
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        assert is_relative_index, "Only relative supported"

        return dataframe_to_numpy_zxy_single_frame(self.df_training_tracklets, t=i_frame)

    def get_distance_to_closest_neuron(self, i_frame, target_pt, nbr_obj=None):
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

    def correct_relative_training_index(self, i):
        return self.which_training_frames[i]

    def napari_of_single_match(self, pair, which_matches='final_matches', this_match: FramePair = None,
                               rigidly_align_volumetric_images=False):
        import napari
        from DLC_for_WBFM.utils.visualization.napari_from_config import napari_tracks_from_match_list
        if this_match is None:
            this_match: FramePair = self.raw_matches[pair]

        dat0, dat1 = self.red_data[pair[0], ...], self.red_data[pair[1], ...]
        this_match.load_raw_data(dat0, dat1)
        if rigidly_align_volumetric_images:
            # Ensure that both point cloud and data have rotations
            this_match.preprocess_data(force_rotation=True)
            # Load the rotated versions
            n0_zxy = this_match.pts0_preprocessed  # May be rotated
            dat0 = this_match.dat0_preprocessed
        else:
            # Keep the non-rotated versions
            n0_zxy = this_match.pts0

        n1_zxy = this_match.pts1
        raw_red_data = np.stack([dat0, dat1])

        list_of_matches = getattr(this_match, which_matches)
        all_tracks_list = napari_tracks_from_match_list(list_of_matches, n0_zxy, n1_zxy)

        v = napari.view_image(raw_red_data, ndisplay=3)
        v.add_points(n0_zxy, size=3, face_color='green', symbol='x', n_dimensional=True)
        v.add_points(n1_zxy, size=3, face_color='blue', symbol='o', n_dimensional=True)
        v.add_tracks(all_tracks_list, head_length=2, name=which_matches)

        # Add text overlay; temporarily change the neuron locations on the frame
        original_zxy = this_match.frame0.neuron_locs
        this_match.frame0.neuron_locs = n0_zxy
        frames = {0: this_match.frame0, 1: this_match.frame1}
        options = napari_labels_from_frames(frames, num_frames=2, to_flip_zxy=False)
        v.add_points(**options)
        this_match.frame0.neuron_locs = original_zxy

        return v

    def add_layers_to_viewer(self, viewer=None, which_layers='all',
                             to_remove_flyback=False, check_if_layers_exist=False,
                             dask_for_segmentation=True):
        if viewer is None:
            import napari
            viewer = napari.Viewer(ndisplay=3)
        if which_layers == 'all':
            which_layers = ['Red data', 'Green data', 'Raw segmentation', 'Colored segmentation',
                            'Neuron IDs', 'Intermediate global IDs']
        if check_if_layers_exist:
            # NOTE: only works if the layer names are the same as these convinience names
            new_layers = set(which_layers) - set([layer.name for layer in viewer.layers])
            which_layers = list(new_layers)

        logger.info(f"Finished loading data, adding following layers: {which_layers}")
        z_to_xy_ratio = self.physical_unit_conversion.z_to_xy_ratio
        if to_remove_flyback:
            clipping_list = [{'position': [2*z_to_xy_ratio, 0, 0], 'normal': [1, 0, 0], 'enabled': True}]
        else:
            clipping_list = []

        raw_chunk = self.red_data.chunks
        dask_chunk = list(raw_chunk).copy()
        dask_chunk[0] = 50

        if 'Red data' in which_layers:
            red_dask = da.from_zarr(self.red_data, chunk=dask_chunk)
            viewer.add_image(red_dask, name="Red data", opacity=0.5, colormap='red',
                             contrast_limits=[0, 200],
                             scale=(1.0, z_to_xy_ratio, 1.0, 1.0),
                             experimental_clipping_planes=clipping_list)
        if 'Green data' in which_layers:
            green_dask = da.from_zarr(self.green_data, chunk=dask_chunk)
            viewer.add_image(green_dask, name="Green data", opacity=0.5, colormap='green', visible=False,
                             contrast_limits=[0, 200],
                             scale=(1.0, z_to_xy_ratio, 1.0, 1.0),
                             experimental_clipping_planes=clipping_list)
        if 'Raw segmentation' in which_layers:
            if dask_for_segmentation:
                seg_array = da.from_zarr(self.raw_segmentation, chunk=dask_chunk)
            else:
                seg_array = zarr.array(self.raw_segmentation)
            viewer.add_labels(seg_array, name="Raw segmentation",
                              scale=(1.0, z_to_xy_ratio, 1.0, 1.0), opacity=0.8, visible=False,
                              rendering='translucent')
        if 'Colored segmentation' in which_layers and self.segmentation is not None:
            viewer.add_labels(self.segmentation, name="Colored segmentation",
                              scale=(1.0, z_to_xy_ratio, 1.0, 1.0), opacity=0.4, visible=False)

        # Add a text overlay
        if 'Neuron IDs' in which_layers:
            df = self.red_traces
            options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
            viewer.add_points(**options)

        if 'GT IDs' in which_layers:
            # Not added by default!
            df = self.final_tracks
            neurons_that_are_finished, _ = self.get_ground_truth_annotations()
            neuron_name_dict = {name: f"GT_{name.split('_')[1]}" for name in neurons_that_are_finished}
            options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio,
                                                          neuron_name_dict=neuron_name_dict)
            options['name'] = 'GT IDs'
            options['text']['color'] = 'red'
            viewer.add_points(**options)

        if 'Intermediate global IDs' in which_layers and self.intermediate_global_tracks is not None:
            df = self.intermediate_global_tracks
            options = napari_labels_from_traces_dataframe(df, z_to_xy_ratio=z_to_xy_ratio)
            options['name'] = 'Intermediate global IDs'
            options['text']['color'] = 'green'
            options['visible'] = False
            viewer.add_points(**options)

        if 'Point Cloud' in which_layers:
            # Note: performance is horrible here
            raise NotImplementedError
            def make_time_vector(zxy, i):
                out = np.array([i]*zxy.shape[0])
                out = np.expand_dims(out, axis=-1)
                return out
            pts_data = [self.get_centroids_as_numpy(i) for i in tqdm(range(self.num_frames), leave=False)]
            pts_data = [np.hstack([make_time_vector(zxy, i), zxy]) for i, zxy in enumerate(pts_data) if len(zxy) > 0]
            pts_data = np.vstack(pts_data)

            options = dict(data=pts_data, name="Point Cloud", size=1, blending='opaque')
            viewer.add_points(**options)

        logger.info(f"Finished adding layers {which_layers}")

        return viewer

    def get_ground_truth_annotations(self):
        # TODO: do not hardcode
        track_cfg = self.project_config.get_tracking_config()
        fname = track_cfg.resolve_relative_path("manual_annotation/manual_tracking.csv", prepend_subfolder=True)
        df_manual_tracking = pd.read_csv(fname)
        neurons_that_are_finished = list(df_manual_tracking[df_manual_tracking['Finished?']]['Neuron ID'])
        return neurons_that_are_finished, df_manual_tracking

    def __repr__(self):
        return f"=======================================\n\
Project data for directory:\n\
{self.project_dir} \n\
Found the following data files:\n\
============Raw========================\n\
red_data:                 {self.red_data is not None}\n\
green_data:               {self.green_data is not None}\n\
============Annotations================\n\
behavior_annotations:     {self.behavior_annotations is not None}\n\
============Training================\n\
df_training_tracklets:    {self.df_training_tracklets is not None}\n\
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

    logger.info(f"Size of reindexed_masks: {z_dat.shape}")

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
