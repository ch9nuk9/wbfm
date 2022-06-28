import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from skimage import transform
from sklearn.decomposition import PCA
from backports.cached_property import cached_property
from sklearn.neighbors import NearestNeighbors

from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig


@dataclass
class WormFullVideoPosture:
    """
    Class for everything to do with Behavior videos

    Specifically collects centerline, curvature, and behavioral annotation information.
    Implements basic pca visualization of the centerlines

    Also knows the frame-rate conversion between the behavioral and fluorescence videos
    """

    filename_curvature: str
    filename_x: str
    filename_y: str
    filename_beh_annotation: str

    # This will be true for old manual annotations
    beh_annotation_already_converted_to_fluorescence_fps: bool = False
    beh_annotation_is_stable_style: bool = False
    _beh_annotation: pd.Series = None

    pca_i_start: int = 10
    pca_i_end: int = -10

    fps: int = 32  # TODO: make sure this is synchronized with z_slices

    @cached_property
    def pca_projections(self):
        pca = PCA(n_components=3, whiten=True)
        curvature_nonan = self.curvature.replace(np.nan, 0.0)
        pca_proj = pca.fit_transform(curvature_nonan.iloc[:, self.pca_i_start:self.pca_i_end])

        return pca_proj

    @cached_property
    def centerlineX(self):
        return pd.read_csv(self.filename_x, header=None)

    @cached_property
    def centerlineY(self):
        return pd.read_csv(self.filename_y, header=None)

    @cached_property
    def curvature(self):
        return pd.read_csv(self.filename_curvature, header=None)

    @property
    def beh_annotation(self):
        """Name is shortened to avoid US-UK spelling confusion"""
        if self._beh_annotation is None:
            self._beh_annotation = get_manual_behavior_annotation(behavior_fname=self.filename_beh_annotation)
        return self._beh_annotation

    def __post_init__(self):
        self.fix_temporary_annotation_format()

    def plot_pca(self):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        c = np.arange(self.curvature.shape[0]) / 1e6
        plt.scatter(self.pca_projections[:, 0], self.pca_projections[:, 1], self.pca_projections[:, 2], c=c)
        plt.colorbar()

    def get_centerline_for_time(self, t):
        c_x = self.centerlineX.iloc[t * self.fps]
        c_y = self.centerlineY.iloc[t * self.fps]
        return np.vstack([c_x, c_y]).T

    @staticmethod
    def load_from_config(project_config: ModularProjectConfig):
        # Get the relevant foldernames from a config file
        # The exact files may not be in the config, so try to find them

        # Before anything, load metadata
        fps = get_behavior_fluorescence_fps_conversion(project_config)
        opt = dict(fps=fps)

        # First, get the folder
        behavior_fname = project_config.config.get('behavior_bigtiff_fname', None)
        if behavior_fname is None:
            project_config.logger.info("behavior_fname not found; searching")
            behavior_subfolder, flag = project_config.get_behavior_raw_parent_folder_from_red_fname()
            if not flag:
                project_config.logger.warning("behavior_fname search failed; aborting")
                raise FileNotFoundError
        else:
            behavior_subfolder = Path(behavior_fname).parent

        # Second get the centerline-specific files
        filename_curvature = None
        filename_x = None
        filename_y = None
        for file in Path(behavior_subfolder).iterdir():
            if not file.is_file():
                continue
            if file.name == 'skeleton_spline_K.csv':
                filename_curvature = str(file)
            elif file.name == 'skeleton_spline_X_coords.csv':
                filename_x = str(file)
            elif file.name == 'skeleton_spline_Y_coords.csv':
                filename_y = str(file)
        all_files = [filename_curvature, filename_x, filename_y]
        if None in all_files:
            project_config.logger.warning(f"Did not find at least one centerline related file: {all_files}")
            # raise FileNotFoundError

        # Third, get the automatic behavior annotations
        try:
            filename_beh_annotation, is_stable_style = get_manual_behavior_annotation_fname(project_config)
            all_files.append(filename_beh_annotation)
            opt.update(dict(beh_annotation_already_converted_to_fluorescence_fps=is_stable_style,
                       beh_annotation_is_stable_style=is_stable_style))
        except FileNotFoundError:
            # Many old projects won't have this
            project_config.logger.warning("Did not find behavioral annotations")
            pass

        return WormFullVideoPosture(*all_files, **opt)

    def shade_using_behavior(self, **kwargs):
        """Takes care of fps conversion and new vs. old annotation format"""
        bh = self.behavior_annotations_fluorescence_fps
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
            print("Annotations are already stable style")
            return self.beh_annotation

        # Define a lookup table from tmp to stable
        def lut(val):
            _lut = {-1: 0, 0: -1, 1: 1}
            if not np.isscalar(val):
                val = val[0]
            if np.isnan(val):
                return -1
            else:
                return _lut[val]

        vec_lut = np.vectorize(lut)

        self._beh_annotation = pd.Series(np.squeeze(vec_lut(self.beh_annotation.to_numpy())))
        self.beh_annotation_is_stable_style = True
        return self.beh_annotation

    @property
    def behavior_annotations_fluorescence_fps(self):
        if self.beh_annotation_already_converted_to_fluorescence_fps:
            return self.beh_annotation
        else:
            return self.beh_annotation.loc[self.subsample_indices]

    @property
    def curvature_fluorescence_fps(self):
        return self.curvature.iloc[self.subsample_indices, :]

    @property
    def subsample_indices(self):
        return range(0, len(self.beh_annotation), self.fps)

    def __repr__(self):
        return f"=======================================\n\
Posture class with the following files:\n\
============Centerline====================\n\
filename_x:                 {self.filename_x is not None}\n\
filename_y:                 {self.filename_y is not None}\n\
filename_curvature:         {self.filename_curvature is not None}\n\
============Annotations================\n\
filename_beh_annotation:    {self.filename_beh_annotation is not None}\n"


def get_behavior_fluorescence_fps_conversion(project_config):
    # TODO: in new config files, there should be a way to read this directly
    preprocessing_cfg = project_config.get_preprocessing_config()
    raw_number_of_planes = preprocessing_cfg.config['raw_number_of_planes']
    final_number_of_planes = project_config.config['dataset_params']['num_slices']
    # True for older datasets, i.e. I had to remove it in postprocessing
    was_flyback_saved = final_number_of_planes != raw_number_of_planes
    if not was_flyback_saved:
        # Example: 23 saved fluorescence planes correspond to 24 behavior frames
        raw_number_of_planes += 1
    return raw_number_of_planes


def get_manual_behavior_annotation_fname(cfg: ModularProjectConfig):
    """First tries to read from the config file, and if that fails, goes searching"""
    is_stable_style = False
    try:
        behavior_cfg = cfg.get_behavior_config()
        behavior_fname = behavior_cfg.config.get('manual_behavior_annotation', None)
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
        raise FileNotFoundError

    return behavior_fname, is_stable_style


def get_manual_behavior_annotation(cfg: ModularProjectConfig = None, behavior_fname: str = None):
    if behavior_fname is None:
        behavior_fname, is_old_style = get_manual_behavior_annotation_fname(cfg)
    if behavior_fname is not None:
        if str(behavior_fname).endswith('.csv'):
            behavior_annotations = pd.read_csv(behavior_fname, header=1, names=['annotation'], index_col=0)
            if behavior_annotations.shape[1] > 1:
                # Sometimes there is a messed up extra column
                behavior_annotations = pd.Series(behavior_annotations.iloc[:, 0])
        else:
            behavior_annotations = pd.read_excel(behavior_fname, sheet_name='behavior')['Annotation']
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
    neuron_zxy: np.ndarray
    centerline: np.ndarray

    centerline_neighbors: NearestNeighbors = None
    neuron_neighbors: NearestNeighbors = None

    def __post_init__(self):
        self.centerline_neighbors = NearestNeighbors(n_neighbors=2).fit(self.centerline)
        self.neuron_neighbors = NearestNeighbors(n_neighbors=5).fit(self.neuron_zxy)

    def get_closest_centerline_point(self, anchor_pt):
        n_neighbors = 1
        closest_centerline_dist, closest_centerline_ind = self.centerline_neighbors.kneighbors(
            anchor_pt[1:].reshape(1, -1), n_neighbors)
        closest_centerline_pt = self.centerline[closest_centerline_ind[0][0], :]

        return closest_centerline_pt, closest_centerline_ind

    def get_transformation_using_centerline_tangent(self, anchor_pt):
        closest_centerline_pt, closest_centerline_ind = self.get_closest_centerline_point(anchor_pt)

        centerline_tangent = self.centerline[closest_centerline_ind[0][0] + 1, :] - closest_centerline_pt
        angle = np.arctan2(centerline_tangent[0], centerline_tangent[1])
        # angle = np.angle(centerline_tangent[0] + 1j * centerline_tangent[1])
        # print(f"Rotation angle of {angle} with centerline index {closest_centerline_ind} and tangent {centerline_tangent} (pt={closest_centerline_pt})")
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
                2: 'red',
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
            ax.axvspan(block_start, block_end, alpha=0.9, color=color)

        block_start = block_end + 1
