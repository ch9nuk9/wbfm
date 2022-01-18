import concurrent
import logging
import os
from dataclasses import dataclass
from typing import Tuple, List
import napari
import numpy as np
import pandas as pd
import zarr

from DLC_for_WBFM.utils.feature_detection.class_frame_pair import FramePair
from DLC_for_WBFM.utils.feature_detection.utils_tracklets import fix_global2tracklet_full_dict
from sklearn.neighbors import NearestNeighbors
from DLC_for_WBFM.utils.pipeline.tracklet_class import DetectedTrackletsAndNeurons
from DLC_for_WBFM.utils.projects.plotting_classes import TracePlotter, TrackletAndSegmentationAnnotator
from DLC_for_WBFM.utils.visualization.napari_from_config import napari_labels_from_frames
from DLC_for_WBFM.utils.visualization.napari_utils import napari_labels_from_traces_dataframe
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior
from scipy.spatial.distance import cdist
from segmentation.util.utils_metadata import DetectedNeurons, get_metadata_dictionary
from DLC_for_WBFM.utils.projects.utils_filepaths import ModularProjectConfig, read_if_exists, pickle_load_binary, \
    SubfolderConfigFile, load_file_according_to_precedence
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
    reindexed_metadata_training: DetectedNeurons = None

    red_traces: pd.DataFrame = None
    green_traces: pd.DataFrame = None
    # final_tracks: pd.DataFrame = None

    behavior_annotations: pd.DataFrame = None
    background_per_pixel: float = None
    likelihood_thresh: float = None

    verbose: int = 2

    # Precedence when multiple are available
    precedence_global2tracklet: list = None
    precedence_df_tracklets: list = None
    precedence_tracks: list = None

    # Classes for more functionality
    trace_plotter: TracePlotter = None

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
        final_tracks = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                         this_reader=read_if_exists)
        return final_tracks

    @cached_property
    def global2tracklet(self) -> dict:
        tracking_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        possible_fnames = dict(
            manual=tracking_cfg.resolve_relative_path_from_config('manual_correction_global2tracklet_fname'),
            automatic=tracking_cfg.resolve_relative_path_from_config('global2tracklet_matches_fname'))

        fname_precedence = self.precedence_global2tracklet
        global2tracklet = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                            this_reader=pickle_load_binary)
        if global2tracklet is not None:
            global2tracklet = fix_global2tracklet_full_dict(self.df_all_tracklets, global2tracklet)
        return global2tracklet

    @cached_property
    def raw_frames(self):
        logging.info("First time loading the raw frames, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'frame_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        frames = pickle_load_binary(fname)
        return frames

    @cached_property
    def raw_matches(self) -> List[FramePair]:
        logging.info("First time loading the raw matches, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'match_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        matches = pickle_load_binary(fname)
        return matches

    @cached_property
    def raw_clust(self):
        logging.info("First time loading the raw cluster dataframe, may take a while...")
        train_cfg = self.project_config.get_training_config()
        fname = os.path.join('raw', 'clust_df_dat.pickle')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        clust = pickle_load_binary(fname)
        return clust

    @cached_property
    def df_all_tracklets(self):
        logging.info("First time loading the all tracklets, may take a while...")
        train_cfg = self.project_config.get_training_config()
        track_cfg = self.project_config.get_tracking_config()

        # Manual annotations take precedence by default
        possible_fnames = dict(
            manual=track_cfg.resolve_relative_path_from_config('manual_correction_tracklets_df_fname'),
            wiggle=track_cfg.resolve_relative_path_from_config('wiggle_split_tracklets_df_fname'),
            automatic=train_cfg.resolve_relative_path_from_config('df_3d_tracklets'))
        fname_precedence = self.precedence_df_tracklets
        df_all_tracklets = load_file_according_to_precedence(fname_precedence, possible_fnames,
                                                             this_reader=read_if_exists)

        return df_all_tracklets

    @cached_property
    def tracklet_annotator(self):
        tracking_cfg = self.project_config.get_tracking_config()
        training_cfg = self.project_config.get_training_config()
        # fname = tracking_cfg.resolve_relative_path_from_config('global2tracklet_matches_fname')

        tracklet_obj = DetectedTrackletsAndNeurons(self.df_all_tracklets, self.segmentation_metadata)

        obj = TrackletAndSegmentationAnnotator(
            tracklet_obj,
            self.global2tracklet,
            segmentation_metadata=self.segmentation_metadata,
            tracking_cfg=tracking_cfg,
            training_cfg=training_cfg
        )
        return obj

    @cached_property
    def df_fdnc_tracks(self):
        train_cfg = self.project_config.get_tracking_config()
        fname = os.path.join('postprocessing', 'leifer_tracks.h5')
        fname = train_cfg.resolve_relative_path(fname, prepend_subfolder=True)
        df_fdnc_tracks = read_if_exists(fname)
        return df_fdnc_tracks

    def _load_tracklet_related_properties(self):
        _ = self.df_all_tracklets
        _ = self.tracklet_annotator
        _ = self.raw_clust

    def _load_frame_related_properties(self):
        _ = self.raw_frames
        _ = self.raw_matches

    def _load_segmentation_related_properties(self):
        _ = self.segmentation_metadata.segmentation_metadata

    @property
    def num_frames(self):
        return self.project_config.config['dataset_params']['num_frames']

    @property
    def which_training_frames(self):
        # TODO: change this to just load on init?
        train_cfg = self.project_config.get_training_config()
        return train_cfg.config['training_data_3d']['which_frames']

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
                                to_load_frames=False,
                                to_load_segmentation_metadata=False):
        # Initialize object in order to use cached properties
        obj = ProjectData(project_dir, cfg)

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
        reindexed_metadata_training_fname = train_cfg.resolve_relative_path_from_config('reindexed_metadata')
        obj.reindexed_metadata_training = DetectedNeurons(reindexed_metadata_training_fname)

        # Read ahead of time because they may be needed for classes in the threading environment
        _ = obj.final_tracks
        # obj.final_tracks = read_if_exists(final_tracks_fname)

        # TODO: do not hardcode
        behavior_fname = "3-tracking/postprocessing/manual_behavior_annotation.xlsx"
        behavior_fname = cfg.resolve_relative_path(behavior_fname)

        zarr_reader_readonly = lambda fname: zarr.open(fname, mode='r')
        zarr_reader_readwrite = lambda fname: zarr.open(fname, mode='r+')
        excel_reader = lambda fname: pd.read_excel(fname, sheet_name='behavior')['Annotation']

        # Note: when running on the cluster the raw data isn't (for now) accessible
        with safe_cd(cfg.project_dir):

            logging.info("Starting threads to read data...")
            with concurrent.futures.ThreadPoolExecutor() as ex:
                if to_load_tracklets:
                    ex.submit(obj._load_tracklet_related_properties)
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
                raw_segmentation = ex.submit(read_if_exists, seg_fname_raw, zarr_reader_readwrite).result()
                segmentation = ex.submit(read_if_exists, seg_fname, zarr_reader_readonly).result()
                # seg_metadata: dict = ex.submit(pickle_load_binary, seg_metadata_fname).result()
                behavior_annotations = ex.submit(read_if_exists, behavior_fname, excel_reader).result()

            if red_traces is not None:
                red_traces.replace(0, np.nan, inplace=True)
                green_traces.replace(0, np.nan, inplace=True)
            logging.info("Read all data")

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
        # if self.tracklet_annotator is None:
        #     self.tracklet_annotator = TrackletAnnotator(
        #         self.df_all_tracklets,
        #         self.global2tracklet
        #     )
        y = self.tracklet_annotator.calculate_tracklets_for_neuron(neuron_name)
        return y

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

    def modify_segmentation_using_manual_correction(self):
        # TODO: save the list of split neurons in separate pickle
        new_mask = self.tracklet_annotator.candidate_mask
        t = self.tracklet_annotator.time_of_candidate

        print(f"Updating raw segmentation at t = {t}...")
        self.raw_segmentation[t, ...] = new_mask

        print("Updating metadata, but NOT writing to disk...")
        red_volume = self.red_data[t, ...]
        self.segmentation_metadata.modify_segmentation_metadata(t, new_mask, red_volume)

        logging.debug("Metadata modified successfully")

    def shade_axis_using_behavior(self, ax=None, behaviors_to_ignore='none'):
        if self.behavior_annotations is None:
            pass
            # logging.warning("No behavior annotations present; skipping")
        else:
            shade_using_behavior(self.behavior_annotations, ax, behaviors_to_ignore)

    def get_centroids_as_numpy(self, i_frame):
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        return self.segmentation_metadata.detect_neurons_from_file(i_frame)

    def get_centroids_as_numpy_training(self, i_frame: int, is_relative_index=True) -> np.ndarray:
        """Original format of metadata is a dataframe of tuples; this returns a normal np.array"""
        if is_relative_index:
            i_frame = self.correct_relative_index(i_frame)
        return self.reindexed_metadata_training.detect_neurons_from_file(i_frame)

    def get_centroids_as_numpy_training_with_unmatched(self, i_rel: int):
        i_abs = self.correct_relative_index(i_rel)
        matched_pts = self.reindexed_metadata_training.detect_neurons_from_file(i_abs)
        all_pts = self.segmentation_metadata.detect_neurons_from_file(i_abs)

        # Any points that do not have a near-identical match in matched_pts are unmatched
        # These will be appended
        tol = 2.0
        ind_unmatched = ~np.any(cdist(all_pts, matched_pts) < tol, axis=1)

        pts_to_add = all_pts[ind_unmatched, :]
        final_pts = np.vstack([matched_pts, pts_to_add])
        return final_pts

    # def calc_matched_point_clouds(self, pair):
    #     match = self.raw_matches[pair]
    #     pts0, pts1 = [], []
    #     n0, n1 = self.get_centroids_as_numpy(pair[0]), self.get_centroids_as_numpy(pair[1])
    #     for m in match.final_matches:
    #         pts0.append(n0[m[0]])
    #         pts1.append(n1[m[1]])
    #
    #     pts0, pts1 = np.array(pts0), np.array(pts1)
    #     return pts0, pts1

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

    def correct_relative_index(self, i):
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

    def add_layers_to_viewer(self, viewer, which_layers='all', to_remove_flyback=True):
        if which_layers == 'all':
            which_layers = ['red', 'green', 'Raw segmentation', 'Colored segmentation',
                            'Neuron ID', 'Intermediate global ID']
        logging.info(f"Finished loading data, adding following layers: {which_layers}")
        if to_remove_flyback:
            clipping_list = [{'position': [2, 0, 0], 'normal': [1, 0, 0], 'enabled': True}]
        else:
            clipping_list = []

        if 'red' in which_layers:
            viewer.add_image(self.red_data, name="Red data", opacity=0.5, colormap='red',
                             contrast_limits=[0, 110],
                             experimental_clipping_planes=clipping_list)
        if 'green' in which_layers:
            viewer.add_image(self.green_data, name="Green data", opacity=0.5, colormap='green', visible=False,
                             experimental_clipping_planes=clipping_list)
        if 'Raw segmentation' in which_layers:
            viewer.add_labels(self.raw_segmentation, name="Raw segmentation", opacity=0.8, visible=False)
        if 'Colored segmentation' in which_layers and self.segmentation is not None:
            viewer.add_labels(self.segmentation, name="Colored segmentation", opacity=0.4, visible=False)

        # Add a text overlay
        if 'Neuron ID' in which_layers:
            df = self.red_traces
            options = napari_labels_from_traces_dataframe(df)
            viewer.add_points(**options)

        if 'Intermediate global ID' in which_layers and self.intermediate_global_tracks is not None:
            df = self.intermediate_global_tracks
            options = napari_labels_from_traces_dataframe(df)
            options['name'] = 'Intermediate global IDs'
            options['text']['color'] = 'green'
            options['visible'] = False
            viewer.add_points(**options)

        logging.info("Finished adding layers to napari")

    def __repr__(self):
        return f"=======================================\n\
Project data for directory:\n\
{self.project_dir} \n\
=======================================\n\
Found the following raw data files:\n\
red_data:                 {self.red_data is not None}\n\
green_data:               {self.green_data is not None}\n\
============Segmentation===============\n\
raw_segmentation:         {self.raw_segmentation is not None}\n\
segmentation:             {self.segmentation is not None}\n\
============Tracklets==================\n\
df_training_tracklets:    {self.df_training_tracklets is not None}\n\
reindexed_masks_training: {self.reindexed_masks_training is not None}\n\
============Traces=====================\n\
red_traces:               {self.red_traces is not None}\n\
green_traces:             {self.green_traces is not None}\n\
final_tracks:             {self.final_tracks is not None}\n\
behavior_annotations:     {self.behavior_annotations is not None}\n"


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

    logging.info(f"Size of reindexed_masks: {z_dat.shape}")

    viewer = napari.view_labels(z_seg, ndisplay=3)
    viewer.add_labels(raw_seg, visible=False)
    viewer.add_image(z_dat)
    viewer.show()

    return viewer, z_dat, z_seg
