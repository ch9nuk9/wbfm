import copy
import glob
import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from skimage import transform
from sklearn.decomposition import PCA
from backports.cached_property import cached_property
from sklearn.neighbors import NearestNeighbors

from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_filenames import resolve_mounted_path_in_current_os, read_if_exists
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df


@dataclass
class WormFullVideoPosture:
    """
    Class for everything to do with Behavior videos

    Specifically collects centerline, curvature, and behavioral annotation information.
    Implements basic pca visualization of the centerlines

    Also knows the frame-rate conversion between the behavioral and fluorescence videos
    """

    filename_curvature: str = None
    filename_x: str = None
    filename_y: str = None
    filename_beh_annotation: str = None

    filename_table_position: str = None

    # This will be true for old manual annotations
    beh_annotation_already_converted_to_fluorescence_fps: bool = False
    beh_annotation_is_stable_style: bool = False
    _beh_annotation: pd.Series = None

    pca_i_start: int = 10
    pca_i_end: int = -10

    bigtiff_start_volume: int = 0
    frames_per_volume: int = 32  # TODO: make sure this is synchronized with z_slices

    project_config: ModularProjectConfig = None

    def __post_init__(self):
        self.fix_temporary_annotation_format()

        if self.filename_curvature is not None:
            self.filename_curvature = resolve_mounted_path_in_current_os(self.filename_curvature, verbose=0)
            self.filename_x = resolve_mounted_path_in_current_os(self.filename_x, verbose=0)
            self.filename_y = resolve_mounted_path_in_current_os(self.filename_y, verbose=0)

        if self.filename_table_position is None and self.filename_curvature is not None:
            # Try to find in the parent folder
            main_folder = Path(self.filename_curvature).parents[1]
            fnames = [fn for fn in glob.glob(os.path.join(main_folder, '*TablePosRecord.txt'))]
            if len(fnames) != 1:
                logging.warning(f"Did not find stage position file in {main_folder}")
            else:
                self.filename_table_position = fnames[0]

    @cached_property
    def pca_projections(self):
        pca = PCA(n_components=3, whiten=True)
        curvature_nonan = self.curvature.replace(np.nan, 0.0)
        pca_proj = pca.fit_transform(curvature_nonan.iloc[:, self.pca_i_start:self.pca_i_end])

        return pca_proj

    @cached_property
    def centerlineX(self):
        return read_if_exists(self.filename_x, reader=pd.read_csv, header=None)

    @cached_property
    def centerlineY(self):
        return read_if_exists(self.filename_y, reader=pd.read_csv, header=None)

    @cached_property
    def curvature(self):
        return read_if_exists(self.filename_curvature, reader=pd.read_csv, header=None)

    @cached_property
    def stage_position(self):
        df = pd.read_csv(self.filename_table_position, index_col='time')
        df.index = pd.DatetimeIndex(df.index)
        return df

    @property
    def beh_annotation(self):
        """Name is shortened to avoid US-UK spelling confusion"""
        if self._beh_annotation is None:
            self._beh_annotation = get_manual_behavior_annotation(behavior_fname=self.filename_beh_annotation)
        return self._beh_annotation

    def plot_pca(self):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        c = np.arange(self.curvature.shape[0]) / 1e6
        plt.scatter(self.pca_projections[:, 0], self.pca_projections[:, 1], self.pca_projections[:, 2], c=c)
        plt.colorbar()

    def get_centerline_for_time(self, t):
        c_x = self.centerlineX.iloc[t * self.frames_per_volume]
        c_y = self.centerlineY.iloc[t * self.frames_per_volume]
        return np.vstack([c_x, c_y]).T

    @staticmethod
    def load_from_config(project_config: ModularProjectConfig):
        # Get the relevant foldernames from a config file
        # The exact files may not be in the config, so try to find them

        # Before anything, load metadata
        frames_per_volume = get_behavior_fluorescence_fps_conversion(project_config)
        bigtiff_start_volume = project_config.config['dataset_params'].get('bigtiff_start_volume', 0)
        opt = dict(frames_per_volume=frames_per_volume,
                   bigtiff_start_volume=bigtiff_start_volume,
                   project_config=project_config)

        # First, get the folder that contains all behavior information
        # Try 1: read from config file
        behavior_fname = project_config.config.get('behavior_bigtiff_fname', None)
        if behavior_fname is None:
            # Try 2: look in the parent folder of the red raw data
            project_config.logger.debug("behavior_fname not found; searching")
            behavior_subfolder, flag = project_config.get_behavior_raw_parent_folder_from_red_fname()
            if not flag:
                project_config.logger.warning("behavior_fname search failed; All calculations with curvature (kymograph) will fail")
                behavior_subfolder = None
        else:
            behavior_subfolder = Path(behavior_fname).parent

        if behavior_subfolder is not None:

            # Second get the centerline-specific files
            filename_curvature, filename_x, filename_y = None, None, None
            for file in Path(behavior_subfolder).iterdir():
                if not file.is_file():
                    continue
                if file.name == 'skeleton_spline_K.csv':
                    filename_curvature = str(file)
                elif file.name == 'skeleton_spline_X_coords.csv':
                    filename_x = str(file)
                elif file.name == 'skeleton_spline_Y_coords.csv':
                    filename_y = str(file)
            all_files = dict(filename_curvature=filename_curvature,
                             filename_x=filename_x,
                             filename_y=filename_y)
        else:
            all_files = dict()

        # Third, get the automatic behavior annotations
        # Might exist even as manual annotation even if the behavior_subfolder wasn't found
        try:
            filename_beh_annotation, is_stable_style = get_manual_behavior_annotation_fname(project_config)
            opt.update(dict(beh_annotation_already_converted_to_fluorescence_fps=is_stable_style,
                       beh_annotation_is_stable_style=is_stable_style))
        except FileNotFoundError:
            # Many projects won't have either annotation
            project_config.logger.warning("Did not find behavioral annotations")
            filename_beh_annotation = None
        all_files['filename_beh_annotation'] = filename_beh_annotation

        # Fourth, get the table stage position (Should always exist)
        filename_table_position = None
        fnames = [fn for fn in glob.glob(os.path.join(behavior_subfolder.parent, '*TablePosRecord.txt'))]
        if len(fnames) != 1:
            logging.warning(f"Did not find stage position file in {behavior_subfolder}")
        else:
            filename_table_position = fnames[0]
        all_files['filename_table_position'] = filename_table_position

        # Even if no files found, at least save the fps
        return WormFullVideoPosture(**all_files, **opt)

    def shade_using_behavior(self, **kwargs):
        """Takes care of fps conversion and new vs. old annotation format"""
        bh = self.behavior_annotations_fluorescence_fps
        if bh is not None:
            shade_using_behavior(bh, **kwargs)

    def fix_temporary_annotation_format(self):
        """
        Temporary types:
            nan - Invalid data (no shade)
            -1 - FWD (no shade)
            0 - Turn (unknown)
            1 - REV (gray)
            [no quiescent for now]
        Returns
        -------

        """
        if self.beh_annotation_is_stable_style:
            return self.beh_annotation
        if self.beh_annotation is None:
            return None

        # Define a lookup table from tmp to stable
        def lut(val):
            _lut = {-1: 0, 0: -1, 1: 1}
            if not np.isscalar(val):
                val = val[0]
            if np.isnan(val):
                return -1
            else:
                return _lut[val]
        try:
            vec_lut = np.vectorize(lut)

            self._beh_annotation = pd.Series(np.squeeze(vec_lut(self.beh_annotation.to_numpy())))
            self.beh_annotation_is_stable_style = True
            return self.beh_annotation
        except KeyError:
            logging.warning("Could not correct behavior annotations; returning them as they are")
            self.beh_annotation_is_stable_style = True
            return self.beh_annotation

    @property
    def behavior_annotations_fluorescence_fps(self):
        if self.beh_annotation is None:
            return None
        if self.beh_annotation_already_converted_to_fluorescence_fps:
            return self.beh_annotation
        else:
            return self.beh_annotation.loc[self.subsample_indices]

    @property
    def curvature_fluorescence_fps(self):
        if self.curvature is not None:
            return self.curvature.iloc[self.subsample_indices, :]
        else:
            return None

    @property
    def stage_position_fluorescence_fps(self):
        return self.stage_position.iloc[self.subsample_indices, :]

    @cached_property
    def worm_speed(self):
        df = self.stage_position
        speed = np.sqrt(np.gradient(df['X']) ** 2 + np.gradient(df['Y']) ** 2)

        tdelta = pd.Series(df.index).diff().mean()
        tdelta_s = tdelta.delta / 1e9
        speed_mm_per_s = speed / tdelta_s

        return speed_mm_per_s

    @cached_property
    def worm_speed_fluorescence_fps(self):
        # Don't subset the speed directly, but go back to the positions
        df = self.stage_position_fluorescence_fps
        speed = np.sqrt(np.gradient(df['X']) ** 2 + np.gradient(df['Y']) ** 2)

        tdelta = pd.Series(df.index).diff().mean()
        tdelta_s = tdelta.delta / 1e9
        speed_mm_per_s = speed / tdelta_s

        return speed_mm_per_s

    @property
    def worm_speed_fluorescence_fps_signed(self):
        """Just sets the speed to be negative when the behavior is annotated as reversal"""
        speed = self.worm_speed_fluorescence_fps
        rev_ind = self.behavior_annotations_fluorescence_fps == 1
        velocity = copy.copy(speed)
        velocity[rev_ind] *= -1

        return velocity

    @property
    def worm_speed_smoothed(self):
        window = 50
        return pd.Series(self.worm_speed).rolling(window=window, center=True).mean()

    @property
    def worm_speed_smoothed_fluorescence_fps(self):
        window = 5
        return pd.Series(self.worm_speed_fluorescence_fps).rolling(window=window, center=True).mean()

    @property
    def leifer_curvature_from_kymograph(self):
        # Signed average over segments 10 to 90
        return self.curvature_fluorescence_fps.loc[:, 5:90].mean(axis=1)

    @property
    def subsample_indices(self):
        # Note: sometimes the curvature and beh_annotations are different length, if one is manually created
        return range(self.bigtiff_start_volume*self.frames_per_volume, len(self.worm_speed), self.frames_per_volume)

    def __repr__(self):
        return f"=======================================\n\
Posture class with the following files:\n\
============Centerline====================\n\
filename_x:                 {self.filename_x is not None}\n\
filename_y:                 {self.filename_y is not None}\n\
filename_curvature:         {self.filename_curvature is not None}\n\
============Annotations================\n\
filename_beh_annotation:    {self.filename_beh_annotation is not None}\n\
============Stage Position================\n\
filename_table_position:    {self.filename_table_position is not None}\n"


def get_behavior_fluorescence_fps_conversion(project_config):
    # TODO: in new config files, there should be a way to read this directly
    preprocessing_cfg = project_config.get_preprocessing_config()
    final_number_of_planes = project_config.config['dataset_params']['num_slices']
    raw_number_of_planes = preprocessing_cfg.config.get('raw_number_of_planes', final_number_of_planes)
    # True for older datasets, i.e. I had to remove it in postprocessing
    was_flyback_saved = final_number_of_planes != raw_number_of_planes
    if not was_flyback_saved:
        # Example: 22 saved fluorescence planes correspond to 24 behavior frames
        # UPDATE: as of August 2022, we remove 2 flyback planes
        raw_number_of_planes += 2
    return raw_number_of_planes


def get_manual_behavior_annotation_fname(cfg: ModularProjectConfig):
    """First tries to read from the config file, and if that fails, goes searching"""

    # Initial checks are all in project local folders
    is_stable_style = False
    try:
        behavior_cfg = cfg.get_behavior_config()
        behavior_fname = behavior_cfg.config.get('manual_behavior_annotation', None)
        if behavior_fname is not None and \
                Path(behavior_fname).exists() and \
                not Path(behavior_fname).is_absolute():
            # Assume it is in this project's behavior folder
            behavior_fname = behavior_cfg.resolve_relative_path(behavior_fname, prepend_subfolder=True)
            if str(behavior_fname).endswith('.xlsx'):
                # This means the user probably did it by hand... but is a fragile check
                is_stable_style = True
    except FileNotFoundError:
        # Old style project
        behavior_fname = None

    if behavior_fname is not None:
        return behavior_fname, is_stable_style

    # Otherwise, check for other places I used to put it
    is_stable_style = True
    behavior_fname = "3-tracking/manual_annotation/manual_behavior_annotation.xlsx"
    behavior_fname = cfg.resolve_relative_path(behavior_fname)
    if not os.path.exists(behavior_fname):
        behavior_fname = "3-tracking/postprocessing/manual_behavior_annotation.xlsx"
        behavior_fname = cfg.resolve_relative_path(behavior_fname)
    if not os.path.exists(behavior_fname):
        behavior_fname = None
    if behavior_fname is not None:
        return behavior_fname, is_stable_style

    # Final checks are all in raw behavior data folders, implying they are not the stable style
    is_stable_style = False
    raw_behavior_folder, flag = cfg.get_behavior_raw_parent_folder_from_red_fname()
    if not flag:
        return behavior_fname, is_stable_style

    behavior_fname = "beh_annotation.csv"
    behavior_fname = os.path.join(raw_behavior_folder, behavior_fname)
    if not os.path.exists(behavior_fname):
        behavior_fname = None

    return behavior_fname, is_stable_style


def get_manual_behavior_annotation(cfg: ModularProjectConfig = None, behavior_fname: str = None):
    if behavior_fname is None:
        if cfg is not None:
            behavior_fname, is_old_style = get_manual_behavior_annotation_fname(cfg)
        else:
            # Only None was passed
            return None
    if behavior_fname is not None:
        if str(behavior_fname).endswith('.csv'):
            behavior_annotations = pd.read_csv(behavior_fname, header=1, names=['annotation'], index_col=0)
            if behavior_annotations.shape[1] > 1:
                # Sometimes there is a messed up extra column
                behavior_annotations = pd.Series(behavior_annotations.iloc[:, 0])
        else:
            try:
                behavior_annotations = pd.read_excel(behavior_fname, sheet_name='behavior')['Annotation']
            except (PermissionError, FileNotFoundError):
                logging.warning("Permission or File error; do you have the excel sheet open elsewhere?")
                behavior_annotations = None
    else:
        behavior_annotations = None

    return behavior_annotations


@dataclass
class WormReferencePosture:

    reference_posture_ind: int
    all_postures: WormFullVideoPosture

    posture_radius: int = 0.7
    frames_per_volume: int = 32

    @property
    def pca_projections(self):
        return self.all_postures.pca_projections

    @property
    def reference_posture(self):
        return self.pca_projections[[self.reference_posture_ind], :]

    @cached_property
    def nearest_neighbor_obj(self):
        neigh = NearestNeighbors(n_neighbors=3)
        neigh.fit(self.pca_projections)

        return neigh

    @cached_property
    def all_dist_from_reference_posture(self):
        return np.linalg.norm(self.pca_projections[:, :3] - self.reference_posture, axis=1)

    @cached_property
    def indices_close_to_reference(self):
        # Converts to volume space using frames_per_volume

        pts, neighboring_ind = self.nearest_neighbor_obj.radius_neighbors(self.reference_posture,
                                                                          radius=self.posture_radius)
        neighboring_ind = neighboring_ind[0]
        # Use the behavioral posture corresponding to the middle (usually plane 15) of the fluorescence recording
        offset = int(self.frames_per_volume / 2)
        neighboring_ind = np.round((neighboring_ind + offset) / self.frames_per_volume).astype(int)
        neighboring_ind = list(set(neighboring_ind))
        neighboring_ind.sort()
        return neighboring_ind

    def get_next_close_index(self, i_start):
        for i in self.indices_close_to_reference:
            if i > i_start:
                return i
        else:
            logging.warning(f"Found no close indices after the query ({i_start})")
            return None


@dataclass
class WormSinglePosture:
    """
    Class for more detailed analysis of the posture at a single time point

    See also WormFullVideoPosture
    """

    neuron_zxy: np.ndarray
    centerline: np.ndarray

    centerline_neighbors: NearestNeighbors = None
    neuron_neighbors: NearestNeighbors = None

    def __post_init__(self):
        self.centerline_neighbors = NearestNeighbors(n_neighbors=2).fit(self.centerline)
        self.neuron_neighbors = NearestNeighbors(n_neighbors=5).fit(self.neuron_zxy)

    def get_closest_centerline_point(self, anchor_pt: Union[np.array, list]):
        """

        Parameters
        ----------
        anchor_pt - zxy of the desired point

        Returns
        -------

        """
        n_neighbors = 1
        closest_centerline_dist, closest_centerline_ind = self.centerline_neighbors.kneighbors(
            anchor_pt[1:].reshape(1, -1), n_neighbors)
        closest_centerline_pt = self.centerline[closest_centerline_ind[0][0], :]

        return closest_centerline_pt, closest_centerline_ind

    def get_transformation_using_centerline_tangent(self, anchor_pt):
        closest_centerline_pt, closest_centerline_ind = self.get_closest_centerline_point(anchor_pt)

        centerline_tangent = self.centerline[closest_centerline_ind[0][0] + 1, :] - closest_centerline_pt
        angle = np.arctan2(centerline_tangent[0], centerline_tangent[1])
        matrix = transform.EuclideanTransform(rotation=angle)

        return matrix

    def get_neighbors(self, anchor_pt, n_neighbors):
        neighbor_dist, neighbor_ind = self.neuron_neighbors.kneighbors(anchor_pt.reshape(1, -1), n_neighbors + 1)
        # Closest neighbor is itself
        neighbor_dist = neighbor_dist[0][1:]
        neighbor_ind = neighbor_ind[0][1:]
        neighbors_zxy = self.neuron_zxy[neighbor_ind, :]

        return neighbors_zxy, neighbor_ind

    def get_neighbors_in_local_coordinate_system(self, i_anchor, n_neighbors=10):
        anchor_pt = self.neuron_zxy[i_anchor]
        neighbors_zxy, neighbor_ind = self.get_neighbors(anchor_pt, n_neighbors)

        matrix = self.get_transformation_using_centerline_tangent(anchor_pt)
        new_pts = transform.matrix_transform(neighbors_zxy[:, 1:] - anchor_pt[1:], matrix.params)

        new_pts_zxy = np.zeros_like(neighbors_zxy)
        new_pts_zxy[:, 0] = neighbors_zxy[:, 0]
        new_pts_zxy[:, 1] = new_pts[:, 0]
        new_pts_zxy[:, 2] = new_pts[:, 1]
        return new_pts_zxy

    def get_all_neurons_in_local_coordinate_system(self, i_anchor):
        anchor_pt = self.neuron_zxy[i_anchor]

        matrix = self.get_transformation_using_centerline_tangent(anchor_pt)
        new_pts = transform.matrix_transform(self.neuron_zxy[:, 1:] - anchor_pt[1:], matrix.params)

        new_pts_zxy = np.zeros_like(self.neuron_zxy)
        new_pts_zxy[:, 0] = self.neuron_zxy[:, 0]
        new_pts_zxy[:, 1] = new_pts[:, 0]
        new_pts_zxy[:, 2] = new_pts[:, 1]

        return new_pts_zxy


def shade_using_behavior(bh, ax=None, behaviors_to_ignore='none',
                         cmap=None,
                         DEBUG=False):
    """
    Type one:
        Shades current plot using a 3-code behavioral annotation:
        -1 - Invalid data (no shade)
        0 - FWD (no shade)
        1 - REV (gray)
        2 - Turn (red)
        3 - Quiescent (light blue)

    """

    if cmap is None:
        cmap = {0: None,
                1: 'lightgray',
                2: 'pink',
                3: 'lightblue'}
    if ax is None:
        ax = plt.gca()
    bh = np.array(bh)

    block_final_indices = np.where(np.diff(bh))[0]
    block_final_indices = np.concatenate([block_final_indices, np.array([len(bh) - 1])])
    block_values = bh[block_final_indices]
    if DEBUG:
        print(block_values)
        print(block_final_indices)

    if behaviors_to_ignore != 'none':
        for b in behaviors_to_ignore:
            cmap[b] = None

    block_start = 0
    for val, block_end in zip(block_values, block_final_indices):
        if val is None or np.isnan(val):
            # block_start = block_end + 1
            continue
        try:
            color = cmap.get(val, None)
        except TypeError:
            logging.warning(f"Ignored behavior of value: {val}")
            # Just ignore
            continue
        # finally:
        #     block_start = block_end + 1

        if DEBUG:
            print(color, val, block_start, block_end)
        if color is not None:
            ax.axvspan(block_start, block_end, alpha=0.9, color=color, zorder=-10)

        block_start = block_end + 1


def calc_pairwise_corr_of_dataframes(df_traces, df_speed):
    """
    Columns are data, rows are time

    Do not need to be the same length. Can contain nans

    Parameters
    ----------
    df_traces
    df_speed

    Returns
    -------

    """
    neuron_names = get_names_from_df(df_traces)
    corr = {name: df_speed.corrwith(df_traces[name]) for name in neuron_names}
    return pd.DataFrame(corr)


def _smooth(dat, window):
    return pd.Series(dat).rolling(window, center=True).mean().to_numpy()


def smooth_mat(dat, window_vec):
    return pd.DataFrame(np.vstack([_smooth(dat, window) for window in window_vec]).T)


def plot_highest_correlations(df_traces, df_speed):
    df_corr = calc_pairwise_corr_of_dataframes(df_traces, df_speed)

    def _plot(max_vals, max_names):
        for max_val, (i, max_name) in zip(max_vals, max_names.iteritems()):
            plt.figure()
            # plt.plot(df_speed[i] / np.max(df_speed[i]), label='Normalized speed')
            plt.plot(df_speed[i], label='Speed')
            plt.plot(df_traces[max_name] / np.max(df_traces[max_name]), label='Normalized trace')
            plt.title(f"Corr = {max_val} for {max_name}")
            plt.ylabel("Speed (mm/s) or amplitude")
            plt.xlabel("Frames")
            plt.legend()

    # Positive then negative correlation
    max_names = df_corr.idxmax(axis=1)
    max_vals = df_corr.max(axis=1)
    _plot(max_vals, max_names)

    min_names = df_corr.idxmin(axis=1)
    min_vals = df_corr.min(axis=1)
    _plot(min_vals, min_names)

