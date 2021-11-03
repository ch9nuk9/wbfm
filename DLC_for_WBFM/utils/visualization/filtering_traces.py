import logging
import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.visualization.napari_from_config import napari_labels_from_frames
from DLC_for_WBFM.utils.visualization.napari_utils import napari_labels_from_traces_dataframe
from DLC_for_WBFM.utils.visualization.visualization_behavior import shade_using_behavior
from scipy.spatial.distance import cdist


def remove_outliers_via_rolling_mean(y: pd.DataFrame, window: int, outlier_threshold=None):
    y_filt = y.rolling(window, min_periods=1, center=True).mean()
    error = np.abs(y - y_filt)
    if outlier_threshold is None:
        # TODO: not working very well
        outlier_threshold = 10*error.var() + error.mean()
        print(f"Calculated error threshold at {outlier_threshold}")
        # logging.info(f"Calculated error threshold at {outlier_threshold}")
    is_outlier = error > outlier_threshold
    y[is_outlier] = np.nan

    return y


def remove_outliers_large_diff(y: pd.DataFrame, outlier_threshold=None):
    raise NotImplementedError
    diff = y.diff()
    if outlier_threshold is None:
        # TODO: not working very well
        outlier_threshold = 10*diff.var() + np.abs(diff.mean())
        print(f"Calculated error threshold at {outlier_threshold}")
    # Only remove the first jump frame, because often the outliers are only one frame
    is_outlier = diff > outlier_threshold
    y[is_outlier] = np.nan

    return y


def filter_rolling_mean(y: pd.DataFrame, window: int = 7):
    return y.rolling(window, min_periods=3).mean()


def filter_linear_interpolation(y: pd.DataFrame, window=15):
    return y.interpolate(method='linear', limit=window, limit_direction='both')


def trace_from_dataframe_factory(calculation_mode, background_per_pixel):
    # Way to process a single dataframe
    if calculation_mode == 'integration':
        def calc_single_trace(i, df_tmp):
            try:
                y_raw = df_tmp[i]['brightness']
                vol = df_tmp[i]['volume']
            except KeyError:
                y_raw = df_tmp[i]['intensity_image']
                vol = df_tmp[i]['area']
            return y_raw - background_per_pixel * vol
    elif calculation_mode == 'max':
        def calc_single_trace(i, df_tmp):
            y_raw = df_tmp[i]['all_values']
            f = lambda x: np.max(x, initial=np.nan)
            return y_raw.apply(f) - background_per_pixel
    elif calculation_mode == 'mean':
        def calc_single_trace(i, df_tmp):
            try:
                y_raw = df_tmp[i]['brightness']
                vol = df_tmp[i]['volume']
            except KeyError:
                y_raw = df_tmp[i]['intensity_image']
                vol = df_tmp[i]['area']
            return y_raw / vol - background_per_pixel
    # elif calculation_mode == 'quantile90':
    #     def calc_single_trace(i, df_tmp):
    #         y_raw = df_tmp[i]['all_values']
    #         return np.quantile(y_raw, 0.9) - self.background_per_pixel
    # elif calculation_mode == 'quantile50':
    #     def calc_single_trace(i, df_tmp):
    #         y_raw = df_tmp[i]['all_values']
    #         f = lambda x: np.quantile(x, initial=np.nan)
    #         return np.quantile(y_raw, 0.5) - self.background_per_pixel
    elif calculation_mode == 'volume':
        def calc_single_trace(i, df_tmp):
            try:
                y_raw = df_tmp[i]['volume']
            except KeyError:
                y_raw = df_tmp[i]['area']
            return y_raw
    elif calculation_mode == 'z':
        def calc_single_trace(i, df_tmp):
            try:
                y_raw = df_tmp[i]['z_dlc']
            except KeyError:
                y_raw = df_tmp[i]['z']
            return y_raw
    else:
        raise ValueError(f"Unknown calculation mode {calculation_mode}")

    return calc_single_trace

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

    def correct_relative_index(self, i):
        return self.which_training_frames[i]

    def napari_of_single_match(self, pair, which_matches='final_matches'):
        import napari
        from DLC_for_WBFM.utils.visualization.napari_from_config import napari_tracks_from_match_list

        raw_red_data = self.red_data[pair[0]:pair[1] + 1, ...]
        this_match = self.raw_matches[pair]
        n0_zxy_raw = this_match.frame0.neuron_locs
        n1_zxy_raw = this_match.frame1.neuron_locs

        list_of_matches = getattr(this_match, which_matches)
        all_tracks_list = napari_tracks_from_match_list(list_of_matches, n0_zxy_raw, n1_zxy_raw)

        v = napari.view_image(raw_red_data, ndisplay=3)
        v.add_points(n0_zxy_raw, size=3, face_color='green', symbol='x', n_dimensional=True)
        v.add_points(n1_zxy_raw, size=3, face_color='blue', symbol='o', n_dimensional=True)
        v.add_tracks(all_tracks_list, head_length=2, name=which_matches)

        # Add text overlay
        frames = {0: this_match.frame0, 1: this_match.frame1}
        options = napari_labels_from_frames(frames, num_frames=2, to_flip_zxy=False)
        v.add_points(**options)

        return v

    def add_layers_to_viewer(self, viewer, which_layers='all'):
        print("Finished loading data, starting napari...")
        viewer.add_image(self.red_data, name="Red data", opacity=0.5, colormap='red', visible=False)
        viewer.add_image(self.green_data, name="Green data", opacity=0.5, colormap='green')
        viewer.add_labels(self.raw_segmentation, name="Raw segmentation", opacity=0.4, visible=False)
        if self.segmentation is not None:
            viewer.add_labels(self.segmentation, name="Colored segmentation", opacity=0.4)

        # Add a text overlay
        df = self.red_traces
        options = napari_labels_from_traces_dataframe(df)
        viewer.add_points(**options)

    def __repr__(self):
        return f"=======================================\n\
Project data for directory:\n\
{self.project_dir} \n\
=======================================\n\
Found the following raw data files:\n\
red_data: {self.red_data is not None}\n\
green_data: {self.green_data is not None}\n\
============Segmentation===============\n\
raw_segmentation: {self.raw_segmentation is not None}\n\
segmentation: {self.segmentation is not None}\n\
============Tracklets==================\n\
df_training_tracklets: {self.df_training_tracklets is not None}\n\
reindexed_masks_training: {self.reindexed_masks_training is not None}\n\
============Traces=====================\n\
red_traces: {self.red_traces is not None}\n\
green_traces: {self.green_traces is not None}\n\
final_tracks: {self.final_tracks is not None}\n\
behavior_annotations: {self.behavior_annotations is not None}\n"