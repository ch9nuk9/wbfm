import copy
import glob
import logging
import os
from pathlib import Path
from typing import Union, Optional
import statsmodels.api as sm

import numpy as np
import pandas as pd
from dataclasses import dataclass
from matplotlib import pyplot as plt
from methodtools import lru_cache
from scipy.ndimage import gaussian_filter1d
from skimage import transform
from sklearn.decomposition import PCA
from backports.cached_property import cached_property
from sklearn.neighbors import NearestNeighbors

from wbfm.utils.external.utils_pandas import get_durations_from_column, get_contiguous_blocks_from_column
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_filenames import resolve_mounted_path_in_current_os, read_if_exists
from wbfm.utils.traces.triggered_averages import TriggeredAverageIndices, \
    assign_id_based_on_closest_onset_in_split_lists
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.filtering_traces import remove_outliers_using_std, remove_outliers_via_rolling_mean, \
    filter_gaussian_moving_average


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
    frames_per_volume: int = 32  # Enhancement: make sure this is synchronized with z_slices

    project_config: ModularProjectConfig = None
    num_frames: int = None

    # Postprocessing the time series
    tracking_failure_idx: np.ndarray = None

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
        curvature_nonan = self.curvature().replace(np.nan, 0.0)
        pca_proj = pca.fit_transform(curvature_nonan.iloc[:, self.pca_i_start:self.pca_i_end])

        return pca_proj

    def _validate_and_downsample(self, df: Optional[Union[pd.DataFrame, pd.Series]], fluorescence_fps: bool,
                                 reset_index=False) -> Union[pd.DataFrame, pd.Series]:
        if df is not None:
            df = self.remove_idx_of_tracking_failures(df)
            if fluorescence_fps:
                try:
                    if len(df.shape) == 2:
                        df = df.iloc[self.subsample_indices, :]
                    elif len(df.shape) == 1:
                        df = df.iloc[self.subsample_indices]
                    else:
                        raise NotImplementedError
                except IndexError as e:
                    print(df)
                    print(df.shape)
                    print(self.subsample_indices)
                    raise e
            if reset_index:
                df.reset_index(drop=True, inplace=True)
        return df

    ##
    ## Basic properties
    ##

    @lru_cache(maxsize=8)
    def centerlineX(self, fluorescence_fps=False) -> pd.DataFrame:
        df = self._raw_centerlineX
        df = self._validate_and_downsample(df, fluorescence_fps)
        return df

    @cached_property
    def _raw_centerlineX(self):
        return read_if_exists(self.filename_x, reader=pd.read_csv, header=None)

    @lru_cache(maxsize=8)
    def centerlineY(self, fluorescence_fps=False) -> pd.DataFrame:
        df = self._raw_centerlineY
        df = self._validate_and_downsample(df, fluorescence_fps)
        return df

    @cached_property
    def _raw_centerlineY(self):
        return read_if_exists(self.filename_y, reader=pd.read_csv, header=None)

    @lru_cache(maxsize=8)
    def curvature(self, fluorescence_fps=False) -> pd.DataFrame:
        df = self._raw_curvature
        df = self._validate_and_downsample(df, fluorescence_fps)
        return df

    @cached_property
    def _raw_curvature(self):
        return read_if_exists(self.filename_curvature, reader=pd.read_csv, header=None)

    @lru_cache(maxsize=8)
    def stage_position(self, fluorescence_fps=False) -> pd.DataFrame:
        df = self._raw_stage_position
        df = self._validate_and_downsample(df, fluorescence_fps)
        return df

    @cached_property
    def _raw_stage_position(self):
        df = pd.read_csv(self.filename_table_position, index_col='time')
        df.index = pd.DatetimeIndex(df.index)
        return df

    @lru_cache(maxsize=8)
    def centerline_absolute_coordinates(self, fluorescence_fps=False) -> pd.DataFrame:
        """Returns a multi-index dataframe, where each body segment looks like the stage_position dataframe"""
        # Depends on camera and magnification
        mm_per_pixel = 0.00245
        # Offset depends on camera and frame size
        x = (self.centerlineX(fluorescence_fps) - 340) * mm_per_pixel
        y = (self.centerlineY(fluorescence_fps) - 324) * mm_per_pixel

        # Rotation depends on Ulises' pipeline and camera
        x_abs = self.stage_position(fluorescence_fps).values[:, 0] - y.T
        y_abs = self.stage_position(fluorescence_fps).values[:, 1] + x.T

        df = pd.concat([x_abs, y_abs], keys=['X', 'Y']).swaplevel().T
        return df

    @cached_property
    def _raw_beh_annotation(self) -> pd.Series:
        if self._beh_annotation is None:
            self._beh_annotation = get_manual_behavior_annotation(behavior_fname=self.filename_beh_annotation)
        if isinstance(self._beh_annotation, pd.DataFrame):
            self._beh_annotation = self._beh_annotation.annotation
        return self._beh_annotation

    def calc_speed_from_alias(self, speed_alias: str) -> pd.Series:
        """
        Basically this calls worm_speed with defined key word arguments

        Note: always has fluorescence_fps=True

        Parameters
        ----------
        speed_alias
        kwargs

        Returns
        -------

        """

        possible_values = ['signed_stage_speed', 'abs_stage_speed', 'leifer_curvature', 'summed_curvature', 'pirouette',
                           'signed_stage_speed_smoothed', 'signed_speed_angular',
                           'signed_middle_body_speed', 'worm_speed_average_all_segments',
                           'worm_speed_average_all_segments']
        assert speed_alias in possible_values, f"Must be one of {possible_values}"

        if speed_alias == 'signed_stage_speed':
            y = self.worm_speed(fluorescence_fps=True, signed=True)
        elif speed_alias == 'abs_stage_speed':
            y = self.worm_speed(fluorescence_fps=True)
        elif speed_alias == 'signed_middle_body_speed':
            y = self.worm_speed(fluorescence_fps=True, use_stage_position=False, signed=True)
        elif speed_alias == 'leifer_curvature' or speed_alias == 'summed_curvature':
            assert self.has_full_kymograph, f"No kymograph found for project {self.project_config.project_dir}"
            y = self.summed_curvature_from_kymograph(fluorescence_fps=True)
        elif speed_alias == 'pirouette':
            y = self.calc_psuedo_pirouette_state()
        elif speed_alias == 'signed_stage_speed_smoothed':
            y = self.worm_speed(fluorescence_fps=True, signed=True, strong_smoothing=True)
        elif speed_alias == 'signed_speed_angular':
            y = self.worm_angular_velocity(fluorescence_fps=True)
        elif speed_alias == 'worm_speed_average_all_segments':
            y = self.worm_speed_average_all_segments(fluorescence_fps=True)
        elif speed_alias == 'worm_nose_residual_speed':
            y = self.worm_speed_average_all_segments(fluorescence_fps=True)
        else:
            raise NotImplementedError(speed_alias)

        return y

    @lru_cache(maxsize=8)
    def beh_annotation(self, fluorescence_fps=False) -> Optional[pd.Series]:
        """Name is shortened to avoid US-UK spelling confusion"""
        beh = self._raw_beh_annotation
        if fluorescence_fps:
            if beh is None or self.beh_annotation_already_converted_to_fluorescence_fps:
                return beh
            else:
                return self._validate_and_downsample(beh, fluorescence_fps=fluorescence_fps)
        else:
            if self.beh_annotation_already_converted_to_fluorescence_fps:
                raise ValueError("Full fps behavioral annotation requested, but only low resolution exists")
            return beh

    @lru_cache(maxsize=8)
    def summed_curvature_from_kymograph(self, fluorescence_fps=False) -> pd.Series:
        """Signed average over segments 15 to 80"""
        curvature = self.curvature().loc[:, 15:80].mean(axis=1)
        curvature = self._validate_and_downsample(curvature, fluorescence_fps=fluorescence_fps)
        return curvature

    ##
    ## Speed properties (derivatives)
    ##

    @cached_property
    def _raw_worm_angular_velocity(self):
        """Using angular velocity in 2d pca space. Note: interpolates by default"""

        xyz_pca = self.pca_projections
        window = 5
        x = remove_outliers_via_rolling_mean(pd.Series(xyz_pca[:, 0]), window)
        y = remove_outliers_via_rolling_mean(pd.Series(xyz_pca[:, 1]), window)

        x = pd.Series(x).interpolate()
        y = pd.Series(y).interpolate()
        # Note: arctan2 is required to give the proper sign
        # x = filter_gaussian_moving_average(pd.Series(x), std=12)
        # y = filter_gaussian_moving_average(pd.Series(y), std=12)
        angles = np.unwrap(np.arctan2(y, x))
        smoothed_angles = filter_gaussian_moving_average(pd.Series(angles), std=12)
        # smoothed_angles = filter_gaussian_moving_average(pd.Series(angles), std=12)

        velocity = np.gradient(smoothed_angles)
        velocity = remove_outliers_via_rolling_mean(pd.Series(velocity), window)
        velocity = pd.Series(velocity).interpolate()

        return velocity

    @lru_cache(maxsize=8)
    def worm_angular_velocity(self, fluorescence_fps=False, remove_outliers=True):
        """
        This is the angular velocity in PCA space (first two modes)

        Note: remove outliers by default"""
        velocity = self._raw_worm_angular_velocity
        velocity = self._validate_and_downsample(velocity, fluorescence_fps=fluorescence_fps)
        if fluorescence_fps:
            velocity.reset_index(drop=True, inplace=True)
        if remove_outliers:
            window = 10
            velocity = remove_outliers_via_rolling_mean(pd.Series(velocity), window)
            velocity = pd.Series(velocity).interpolate()
        return velocity

    @lru_cache(maxsize=256)
    def worm_speed(self, fluorescence_fps=False, subsample_before_derivative=True, signed=False,
                   strong_smoothing=False, use_stage_position=True, remove_outliers=True, body_segment=50) -> pd.Series:
        """
        Calculates derivative of position

        Parameters
        ----------
        fluorescence_fps - Whether to downsample
        subsample_before_derivative - Order of downsampling operation
        signed - whether to multiply by -1 when a reversal is annotated
        strong_smoothing - whether to apply a strong smoothing
        use_stage_position - whether to use the stage position (default) or body segment 50
        remove_outliers - whether to remove outliers (replace with nan and interpolate)
        body_segment - only used if use_stage_position=False

        Returns
        -------

        """
        if use_stage_position:
            get_positions = self.stage_position
        else:
            # Use segment 50 out of 100 by default
            get_positions = lambda fluorescence_fps: \
                self.centerline_absolute_coordinates(fluorescence_fps=fluorescence_fps)[body_segment]
        if subsample_before_derivative:
            df = get_positions(fluorescence_fps=fluorescence_fps)
        else:
            df = get_positions(fluorescence_fps=False)

        # Derivative, then convert to physical units (note that subsampling might not have happened yet)
        speed = np.sqrt(np.gradient(df['X']) ** 2 + np.gradient(df['Y']) ** 2)
        tdelta_s = self.get_time_delta_in_s(fluorescence_fps and subsample_before_derivative)
        speed_mm_per_s = pd.Series(speed / tdelta_s)
        # print(speed)
        # print(speed_mm_per_s)
        # print(tdelta_s)

        # Postprocessing
        if not subsample_before_derivative:
            speed_mm_per_s = self._validate_and_downsample(speed_mm_per_s, fluorescence_fps=fluorescence_fps,
                                                           reset_index=True)
        if strong_smoothing:
            window = 50
            speed_mm_per_s = pd.Series(speed_mm_per_s).rolling(window=window, center=True).mean()
        if remove_outliers:
            window = 10
            speed_mm_per_s = remove_outliers_via_rolling_mean(pd.Series(speed_mm_per_s), window)
            speed_mm_per_s = pd.Series(speed_mm_per_s).interpolate()
        if signed:
            speed_mm_per_s = self.flip_of_vector_during_state(speed_mm_per_s, fluorescence_fps=fluorescence_fps)

        return speed_mm_per_s

    def worm_speed_average_all_segments(self, **kwargs):
        """
        Computes the speed of each individual segment (absolute magnitude), then takes an average

        See worm_speed for options

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        single_segment_opt = kwargs.copy()
        single_segment_opt['use_stage_position'] = False
        sign_after_mean = single_segment_opt.get('signed', False)
        single_segment_opt['signed'] = False

        all_speeds = []
        for i in range(100):
            single_segment_opt['body_segment'] = i
            all_speeds.append(self.worm_speed(**single_segment_opt))
        mean_speed = pd.DataFrame(all_speeds).mean(axis=0)

        if sign_after_mean:
            fluorescence_fps = single_segment_opt.get('fluorescence_fps', False)
            mean_speed = self.flip_of_vector_during_state(mean_speed, fluorescence_fps)

        return mean_speed

    def worm_nose_residual_speed(self, **kwargs):
        """
        Computes the difference of the nose and the middle body segment

        See worm_speed for options

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        single_segment_opt = kwargs.copy()
        single_segment_opt['use_stage_position'] = False
        sign_after_mean = single_segment_opt.get('signed', False)
        single_segment_opt['signed'] = False

        nose_speed = self.worm_speed(body_segment=2, **single_segment_opt)
        middle_speed = self.worm_speed(body_segment=50, **single_segment_opt)
        residual_speed = nose_speed - middle_speed

        if sign_after_mean:
            fluorescence_fps = single_segment_opt.get('fluorescence_fps', False)
            residual_speed = self.flip_of_vector_during_state(residual_speed, fluorescence_fps)

        return residual_speed

    def get_time_delta_in_s(self, fluorescence_fps):
        df = self.stage_position(fluorescence_fps=fluorescence_fps)
        tdelta = pd.Series(df.index).diff().mean()
        tdelta_s = tdelta.delta / 1e9
        return tdelta_s

    def flip_of_vector_during_state(self, vector, fluorescence_fps=False, state=1) -> pd.Series:
        """By default changes sign during reversal"""
        rev_ind = pd.Series(self.beh_annotation(fluorescence_fps=fluorescence_fps) == state).reset_index(drop=True)
        velocity = copy.copy(vector)
        if len(velocity) == len(rev_ind):
            velocity[rev_ind] *= -1
        elif len(velocity) == len(rev_ind) + 1:
            velocity = velocity.iloc[:-1]
            velocity[rev_ind] *= -1
        else:
            raise ValueError(f"Velocity ({len(velocity)}) and reversal indices ({len(rev_ind)}) are desynchronized")

        return velocity

    ##
    ## Basic data validation
    ##

    @property
    def has_beh_annotation(self):
        return self.filename_beh_annotation is not None and os.path.exists(self.filename_beh_annotation)

    @property
    def has_full_kymograph(self):
        fnames = [self.filename_y, self.filename_x, self.filename_curvature]
        return all([f is not None for f in fnames]) and all([os.path.exists(f) for f in fnames])

    def validate_dataframes_of_correct_size(self):
        dfs = [self.centerlineX(), self.centerlineY(), self.curvature(), self.stage_position()]
        shapes = [df.shape for df in dfs]
        assert np.allclose(*shapes), "Found invalid shape for some dataframes"

    ##
    ## Other complex states
    ##

    def plot_pca_eigenworms(self):
        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111, projection='3d')
        c = np.arange(self.num_frames) / 1e6
        ax.scatter(self.pca_projections[:, 0], self.pca_projections[:, 1], self.pca_projections[:, 2], c=c)
        plt.colorbar()

    def get_centerline_for_time(self, t):
        c_x = self.centerlineX().iloc[t * self.frames_per_volume]
        c_y = self.centerlineY().iloc[t * self.frames_per_volume]
        return np.vstack([c_x, c_y]).T

    def calc_triggered_average_indices(self, state=0, min_duration=5, ind_preceding=20, **kwargs):
        """
        Calculates a list of indices that can be used to calculate triggered averages of 'state' ONSET

        By default, state=0 is forward, and 1 is reversal. Sometimes 2 is annotated (turn), but this will likely change


        Parameters
        ----------
        state
        min_duration
        trace_len
        kwargs

        Returns
        -------

        """
        ind_class = TriggeredAverageIndices(self.beh_annotation(fluorescence_fps=True) , state, min_duration,
                                            trace_len=self.num_frames, ind_preceding=ind_preceding,
                                            **kwargs)
        return ind_class

    def calc_triggered_average_indices_with_pirouette_split(self, duration_threshold=34, **kwargs):
        """
        Calculates triggered average reversals, with a dictionary classifying them based on the previous forward state

        See calc_triggered_average_indices

        Parameters
        ----------
        duration_threshold: based on a population 2-exponential fit of forward durations
        kwargs

        Returns
        -------

        """
        default_kwargs = dict(gap_size_to_remove=3)
        default_kwargs.update(kwargs)

        # Get the indices for each of the types of states: short/long fwd, and all reversals
        ind_short_fwd = self.calc_triggered_average_indices(state=0, max_duration=duration_threshold, **default_kwargs)
        ind_long_fwd = self.calc_triggered_average_indices(state=0, min_duration=duration_threshold, **default_kwargs)
        ind_rev = self.calc_triggered_average_indices(state=1, min_duration=3, **default_kwargs)

        # Classify the reversals
        short_onsets = np.array(ind_short_fwd.idx_onsets)
        long_onsets = np.array(ind_long_fwd.idx_onsets)
        rev_onsets = np.array(ind_rev.idx_onsets)
        # Assigns 1 for onset type 1, i.e. short
        dict_of_pirouette_rev = assign_id_based_on_closest_onset_in_split_lists(short_onsets, long_onsets, rev_onsets)
        dict_of_non_pirouette_rev = {k: int(1 - v) for k, v in dict_of_pirouette_rev.items()}

        # Build new rev_onset classes based on the classes, and a flipped version
        default_kwargs.update(state=1, min_duration=3)
        ind_rev_pirouette = self.calc_triggered_average_indices(dict_of_events_to_keep=dict_of_pirouette_rev,
                                                                **default_kwargs)
        ind_rev_non_pirouette = self.calc_triggered_average_indices(dict_of_events_to_keep=dict_of_non_pirouette_rev,
                                                                    **default_kwargs)

        return ind_rev_pirouette, ind_rev_non_pirouette

    # def plot_triggered_average(self, state, trace):
    #     ind_class = self.calc_triggered_average_indices(state=state, trace_len=len(trace))
    #     mat = ind_class.calc_triggered_average_matrix(trace)
    #     plot_triggered_average_from_matrix_with_histogram(mat)

    def calc_psuedo_roaming_state(self, thresh=80, only_onset=False, onset_blur_sigma=5):
        """
        Calculates a binary vector that is 1 when the worm is in a long forward bout (defined by thresh), and 0
        otherwise

        If only_onset is true, then the vector is only on at the first point

        Returns
        -------

        """
        binary_fwd = self.beh_annotation(fluorescence_fps=True)  == 0
        all_durations = get_durations_from_column(binary_fwd, already_boolean=True, remove_edges=False)
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_fwd, already_boolean=True)
        start2duration_and_end_dict = {}
        for duration, start, end in zip(all_durations, all_starts, all_ends):
            start2duration_and_end_dict[start] = [duration, end]

        # Turn into time series
        num_pts = len(self.subsample_indices)
        state_trace = np.zeros(num_pts)
        for start, (duration, end) in start2duration_and_end_dict.items():
            if duration < thresh:
                continue

            if not only_onset:
                state_trace[start:end] = 1
            else:
                state_trace[start] = 1
        if only_onset:
            state_trace = gaussian_filter1d(state_trace, onset_blur_sigma)

        return state_trace

    def calc_psuedo_pirouette_state(self, min_duration=3, window=600, std=50):
        """
        Calculates a state that is high when there are many reversal onsets, and low otherwise
            Note: is low even during reversals if they are isolated

        This time series may be entirely 0 if there are only isolated reversals

        Parameters
        ----------
        min_duration
        window
        std

        Returns
        -------

        """
        ind_class = self.calc_triggered_average_indices(state=1, ind_preceding=0,
                                                        min_duration=min_duration)

        onsets = np.array([vec[0] for vec in ind_class.triggered_average_indices() if vec[0] > 0])

        onset_vec = np.zeros(ind_class.trace_len)
        onset_vec[onsets] = 1
        pad_num = int(window / 2)
        onset_vec_pad = np.pad(onset_vec, pad_num, constant_values=0)
        x = np.arange(len(onset_vec_pad)) - pad_num
        # probability_to_reverse = pd.Series(onset_vec_pad).rolling(center=True, window=window, win_type=None,
        # min_periods=1).mean()
        probability_to_reverse = pd.Series(onset_vec_pad).rolling(center=True, window=window, win_type='gaussian',
                                                                  min_periods=1).mean(std=std)

        mod = sm.tsa.MarkovRegression(probability_to_reverse, k_regimes=2)
        res = mod.fit()
        binarized_probability_to_reverse = res.predict()
        predicted_pirouette_state = binarized_probability_to_reverse > 0.010
        # Remove padded indices
        predicted_pirouette_state = predicted_pirouette_state[pad_num:-pad_num].reset_index(drop=True)

        return predicted_pirouette_state

    def calc_fwd_counter_state(self):
        """
        Calculates an integer vector that counts the time since last reversal

        Returns
        -------

        """
        binary_fwd = self.beh_annotation(fluorescence_fps=True)  == 0
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_fwd, already_boolean=True)

        # Turn into time series
        num_pts = len(self.subsample_indices)
        state_trace = np.zeros(num_pts)
        for start, end in zip(all_starts, all_ends):
            state_trace[start:end] = np.arange(end - start)

        return state_trace

    def calc_exponential_chance_to_end_fwd_state(self):
        """        Using a double exponential fit from a population of forward durations, estimates the probability to terminate
        a forward state, assuming one exponential is active at once. Specifically:
            - For short forward periods (<34 volumes), use a sharp exponential of ~2 volume decay
            - For long forward periods, use a flat exponential of ~30 volume decay time

        For now, just use a flat prediction of the tau of these two states... this might be better than an increasing
        series that is very flat towards the end of the forward

        Returns
        -------

        """
        binary_fwd = self.beh_annotation(fluorescence_fps=True)  == 0
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_fwd, already_boolean=True)

        # Turn into time series
        num_pts = len(self.subsample_indices)
        state_trace = np.zeros(num_pts)
        for start, end in zip(all_starts, all_ends):
            state_trace[start:end] = np.arange(end - start)

        return state_trace

    def calc_rev_counter_state(self):
        """
        Calculates an integer vector that counts the time since last forward state

        Returns
        -------

        """
        binary_rev = self.beh_annotation(fluorescence_fps=True)  == 1
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_rev, already_boolean=True)

        # Turn into time series
        num_pts = len(self.subsample_indices)
        state_trace = np.zeros(num_pts)
        for start, end in zip(all_starts, all_ends):
            state_trace[start:end] = np.arange(end - start)

        return state_trace

    @staticmethod
    def load_from_config(project_config: ModularProjectConfig):
        # Get the relevant foldernames from a config file
        # The exact files may not be in the config, so try to find them

        # Before anything, load metadata
        frames_per_volume = get_behavior_fluorescence_fps_conversion(project_config)
        # Use the project data class to check for tracking failures
        from wbfm.utils.projects.finished_project_data import ProjectData
        proj = ProjectData.load_final_project_data_from_config(project_config, to_load_segmentation_metadata=True)
        invalid_idx = proj.estimate_tracking_failures_from_project()

        bigtiff_start_volume = project_config.config['dataset_params'].get('bigtiff_start_volume', 0)
        opt = dict(frames_per_volume=frames_per_volume,
                   bigtiff_start_volume=bigtiff_start_volume,
                   num_frames=proj.num_frames,
                   project_config=project_config,
                   tracking_failure_idx=invalid_idx)

        # Get the folder that contains all behavior information
        # Try 1: read from config file
        behavior_fname = project_config.config.get('behavior_bigtiff_fname', None)
        if behavior_fname is None:
            # Try 2: look in the parent folder of the red raw data
            project_config.logger.debug("behavior_fname not found; searching")
            behavior_subfolder, flag = project_config.get_behavior_raw_parent_folder_from_red_fname()
            if not flag:
                project_config.logger.warning("behavior_fname search failed; "
                                              "All calculations with curvature (kymograph) will fail")
                behavior_subfolder = None
        else:
            behavior_subfolder = Path(behavior_fname).parent

        if behavior_subfolder is not None:

            # Second get the centerline-specific files
            filename_curvature, filename_x, filename_y, filename_beh_annotation = None, None, None, None
            for file in Path(behavior_subfolder).iterdir():
                if not file.is_file() or file.name.startswith('.'):
                    # Skip hidden files and directories
                    continue
                if file.name.endswith('skeleton_spline_K.csv'):
                    filename_curvature = str(file)
                elif file.name.endswith('skeleton_spline_X_coords.csv'):
                    filename_x = str(file)
                elif file.name.endswith('skeleton_spline_Y_coords.csv'):
                    filename_y = str(file)
                elif file.name.endswith('_beh_annotation.csv'):
                    filename_beh_annotation = str(file)
            all_files = dict(filename_curvature=filename_curvature,
                             filename_x=filename_x,
                             filename_y=filename_y,
                             filename_beh_annotation=filename_beh_annotation)

            # Third, get the table stage position
            # Should always exist IF you have access to the raw data folder (which probably means a mounted drive)
            filename_table_position = None
            fnames = [fn for fn in glob.glob(os.path.join(behavior_subfolder.parent, '*TablePosRecord.txt'))]
            if len(fnames) != 1:
                logging.warning(f"Did not find stage position file in {behavior_subfolder}")
            else:
                filename_table_position = fnames[0]
            all_files['filename_table_position'] = filename_table_position

        else:
            all_files = dict()

        # Get the manual behavior annotations if automatic wasn't found
        if all_files.get('filename_beh_annotation', None) is None:
            try:
                filename_beh_annotation, is_stable_style = get_manual_behavior_annotation_fname(project_config)
                opt.update(dict(beh_annotation_already_converted_to_fluorescence_fps=is_stable_style,
                           beh_annotation_is_stable_style=is_stable_style))
            except FileNotFoundError:
                # Many projects won't have either annotation
                project_config.logger.warning("Did not find behavioral annotations")
                filename_beh_annotation = None
            all_files['filename_beh_annotation'] = filename_beh_annotation

        # Even if no files found, at least save the fps
        return WormFullVideoPosture(**all_files, **opt)

    def shade_using_behavior(self, **kwargs):
        """Takes care of fps conversion and new vs. old annotation format"""
        bh = self.beh_annotation(fluorescence_fps=True) 
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
            return self.beh_annotation()
        if self.beh_annotation() is None:
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

            self._beh_annotation = pd.Series(np.squeeze(vec_lut(self.beh_annotation() .to_numpy())))
            self.beh_annotation_is_stable_style = True
            return self.beh_annotation()
        except KeyError:
            logging.warning("Could not correct behavior annotations; returning them as they are")
            self.beh_annotation_is_stable_style = True
            return self.beh_annotation()

    @property
    def subsample_indices(self):
        # Note: sometimes the curvature and beh_annotations are different length, if one is manually created
        offset = self.frames_per_volume // 2  # Take the middle frame
        return range(self.bigtiff_start_volume*self.frames_per_volume + offset,
                     len(self._raw_stage_position),
                     self.frames_per_volume)

    def remove_idx_of_tracking_failures(self, vec: pd.Series) -> pd.Series:
        """
        Removes indices of known tracking failures, if any

        Assumes the high frame rate index
        """
        if self.tracking_failure_idx is not None:
            vec = vec.copy()
            logging.debug(f"Setting these indices as tracking failures: {self.tracking_failure_idx}")
            vec.iloc[self.tracking_failure_idx] = np.nan
        return vec

    def __repr__(self):
        return f"=======================================\n\
Posture class with the following files:\n\
============Centerline====================\n\
filename_x:                 {self.filename_x is not None}\n\
filename_y:                 {self.filename_y is not None}\n\
filename_curvature:         {self.filename_curvature is not None}\n\
============Annotations================\n\
filename_beh_annotation:    {self.has_beh_annotation}\n\
============Stage Position================\n\
filename_table_position:    {self.filename_table_position is not None}\n"


def get_behavior_fluorescence_fps_conversion(project_config):
    # Enhancement: In new config files, there should be a way to read this directly
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
        if behavior_fname is not None and not Path(behavior_fname).is_absolute():
            # Assume it is in this project's behavior folder
            behavior_fname = behavior_cfg.resolve_relative_path(behavior_fname, prepend_subfolder=True)
            if str(behavior_fname).endswith('.xlsx'):
                # This means the user probably did it by hand... but is a fragile check
                is_stable_style = True
            if not os.path.exists(behavior_fname):
                behavior_fname = None
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
            except PermissionError:
                logging.warning(f"Permission error when reading {behavior_fname} "
                                f"Do you have the excel sheet open elsewhere?")
                behavior_annotations = None
            except FileNotFoundError:
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
                         cmap=None, index_conversion=None,
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
            continue
        try:
            color = cmap.get(val, None)
        except TypeError:
            logging.warning(f"Ignored behavior of value: {val}")
            # Just ignore
            continue

        if DEBUG:
            print(color, val, block_start, block_end)
        if color is not None:
            if index_conversion is not None:
                ax_start = index_conversion[block_start]
                ax_end = index_conversion[block_end]
            else:
                ax_start = block_start
                ax_end = block_end

            ax.axvspan(ax_start, ax_end, alpha=0.9, color=color, zorder=-10)

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

