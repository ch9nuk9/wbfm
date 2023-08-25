import copy
import glob
import logging
import math
import os
from pathlib import Path
from typing import Union, Optional, List

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

from wbfm.utils.external.utils_breakpoints import plot_with_offset_x
from wbfm.utils.external.utils_self_collision import calculate_self_collision_using_pairwise_distances
from wbfm.utils.general.utils_behavior_annotation import BehaviorCodes, detect_peaks_and_interpolate, \
    shade_using_behavior, get_same_phase_segment_pairs, get_heading_vector_from_phase_pair_segments, \
    shade_using_behavior_plotly
from wbfm.utils.external.utils_pandas import get_durations_from_column, get_contiguous_blocks_from_column, \
    remove_short_state_changes, pad_events_in_binary_vector
from wbfm.utils.general.custom_errors import NoManualBehaviorAnnotationsError, NoBehaviorAnnotationsError, \
    MissingAnalysisError, DataSynchronizationError
from wbfm.utils.projects.physical_units import PhysicalUnitConversion
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.projects.utils_filenames import resolve_mounted_path_in_current_os, read_if_exists
from wbfm.utils.traces.triggered_averages import TriggeredAverageIndices, \
    assign_id_based_on_closest_onset_in_split_lists
from wbfm.utils.tracklets.high_performance_pandas import get_names_from_df
from wbfm.utils.visualization.filtering_traces import remove_outliers_via_rolling_mean, \
    filter_gaussian_moving_average, fill_nan_in_dataframe
from wbfm.utils.visualization.hardcoded_paths import forward_distribution_statistics, reverse_distribution_statistics


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
    # Does not always exist
    filename_manual_beh_annotation: str = None

    filename_hilbert_amplitude: str = None
    filename_hilbert_frequency: str = None
    filename_hilbert_phase: str = None
    filename_hilbert_carrier: str = None

    filename_self_collision: str = None
    filename_turn_annotation: str = None
    filename_head_cast: str = None

    # Exists without running Ulises' code, but not for immobilized worms
    filename_table_position: str = None

    # This will be true for old manual annotations
    beh_annotation_already_converted_to_fluorescence_fps: bool = False
    _beh_annotation: pd.Series = None

    pca_i_start: int = 10
    pca_i_end: int = -10

    bigtiff_start_volume: int = 0
    frames_per_volume: int = 32  # Enhancement: make sure this is synchronized with z_slices

    project_config: ModularProjectConfig = None
    num_trace_frames: int = None

    # Postprocessing the time series
    tracking_failure_idx: np.ndarray = None  # See estimate_tracking_failures_from_project
    physical_unit_conversion: PhysicalUnitConversion = None

    # If additional files are needed
    behavior_subfolder: str = None

    def __post_init__(self):
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
                                 reset_index=False, use_physical_time=False) -> Optional[Union[pd.DataFrame, pd.Series]]:
        if df is None:
            return df
        elif self.beh_annotation_already_converted_to_fluorescence_fps and not fluorescence_fps:
            raise MissingAnalysisError("Full fps annotation requested, but only low resolution exists")
        else:
            # Get cleaned and downsampled dataframe
            needs_subsampling = fluorescence_fps and not self.beh_annotation_already_converted_to_fluorescence_fps
            try:
                df = self.remove_idx_of_tracking_failures(df, fluorescence_fps=fluorescence_fps)
                if needs_subsampling:
                    if len(df.shape) == 2:
                        df = df.iloc[self.subsample_indices, :]
                    elif len(df.shape) == 1:
                        df = df.iloc[self.subsample_indices]
                    else:
                        raise NotImplementedError
            except IndexError as e:
                print(df)
                print(df.shape)
                print(self.tracking_failure_idx)
                print(self.subsample_indices)
                raise e
            # Optional postprocessing
            if reset_index:
                df.reset_index(drop=True, inplace=True)
            # Shorten to the correct length, if necessary. Note that we have to check for series or dataframe
            if needs_subsampling:
                df = self._shorten_to_trace_length(df)

        # Convert to physical time
        if use_physical_time:
            if fluorescence_fps:
                df.index = self._x_physical_time_volumes
            else:
                df.index = self._x_physical_time_frames

        return df

    @property
    def _x_physical_time_frames(self):
        """Helper for reindexing plots from frames to seconds"""
        x = np.arange(self.num_trace_frames)
        x = x / self.physical_unit_conversion.frames_per_second
        return x

    @property
    def _x_physical_time_volumes(self):
        """Helper for reindexing plots from frames to seconds"""
        x = np.arange(self.num_trace_frames)
        x = x / self.physical_unit_conversion.volumes_per_second
        return x

    def _shorten_to_trace_length(self, df: Union[pd.DataFrame, pd.Series]):
        if len(df.shape) == 2:
            df = df.iloc[:self.num_trace_frames, :]
        elif len(df.shape) == 1:
            df = df.iloc[:self.num_trace_frames]
        return df

    ##
    ## Basic properties
    ##

    @lru_cache(maxsize=8)
    def centerlineX(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        df = self._raw_centerlineX
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_centerlineX(self):
        return read_if_exists(self.filename_x, reader=pd.read_csv, header=None)

    @lru_cache(maxsize=8)
    def centerlineY(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        df = self._raw_centerlineY
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_centerlineY(self):
        return read_if_exists(self.filename_y, reader=pd.read_csv, header=None)

    @lru_cache(maxsize=8)
    def curvature(self, fluorescence_fps=False, rename_columns=False, **kwargs) -> pd.DataFrame:
        df = self._raw_curvature
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        if rename_columns:
            df.columns = [f"segment_{i+1:03d}" for i in df.columns]
        return df

    @lru_cache(maxsize=8)
    def head_smoothed_curvature(self, fluorescence_fps=False, start_segment=1, final_segment=20,
                                **kwargs) -> pd.DataFrame:
        df = self._raw_curvature
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        # Smooth the curvature by taking an expanding average
        # Start with segment 4, which is the first segment that is not too noisy
        # Then take the average of segments 4 and 5, then 4-6, then 4-7, etc.
        df_new = df.copy()

        for i in range(start_segment+1, final_segment):
            df_new.iloc[:, i] = df.iloc[:, i - start_segment:i + 1].mean(axis=1)
        # Remove the first and last few segments, which are not smoothed
        # df_new = df_new.iloc[:, start_segment:final_segment+1]

        return df_new

    @cached_property
    def _raw_curvature(self):
        df = read_if_exists(self.filename_curvature, reader=pd.read_csv, header=None)
        # Remove the first column, which is the frame number
        if df is None:
            raise NoBehaviorAnnotationsError("(curvature)")
        df = df.iloc[:, 1:]
        return df

    @lru_cache(maxsize=8)
    def hilbert_amplitude(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        df = self._raw_hilbert_amplitude
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_hilbert_amplitude(self):
        return read_if_exists(self.filename_hilbert_amplitude, reader=pd.read_csv, header=None)

    @lru_cache(maxsize=8)
    def hilbert_phase(self, fluorescence_fps=False, mod_2pi=True, **kwargs) -> pd.DataFrame:
        df = self._raw_hilbert_phase
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        if mod_2pi:
            df = (df % (2 * math.pi))
        return df

    @lru_cache(maxsize=8)
    def hilbert_phase_derivative(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        df = self.hilbert_phase(fluorescence_fps, mod_2pi=False, **kwargs)
        df = df.diff(axis=0)
        # Sometimes the derivative has some singularities
        return df

    @cached_property
    def _raw_hilbert_phase(self):
        return read_if_exists(self.filename_hilbert_phase, reader=pd.read_csv, header=None)

    @lru_cache(maxsize=8)
    def hilbert_frequency(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        df = self._raw_hilbert_frequency
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_hilbert_frequency(self):
        return read_if_exists(self.filename_hilbert_frequency, reader=pd.read_csv, header=None)

    @lru_cache(maxsize=8)
    def hilbert_carrier(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        df = self._raw_hilbert_carrier
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_hilbert_carrier(self):
        return read_if_exists(self.filename_hilbert_carrier, reader=pd.read_csv, header=None)

    # @lru_cache(maxsize=8)
    def _self_collision(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        """This is intended to be summed with the main behavioral vector"""
        df = self._raw_self_collision
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_self_collision(self) -> Optional[pd.Series]:
        # Ulises' file is not really working right now, so calculate one ourselves
        # This one has a header
        # _raw_vector = read_if_exists(self.filename_self_collision, reader=pd.read_csv, index_col=0)['self_touch']
        if self.centerlineY() is None:
            return None
        _raw_vector, _ = calculate_self_collision_using_pairwise_distances(self.centerlineX(), self.centerlineY())

        # Convert 1's to BehaviorCodes.SELF_COLLISION and 0's to BehaviorCodes.NOT_ANNOTATED
        _raw_vector = _raw_vector.replace(True, BehaviorCodes.SELF_COLLISION)
        _raw_vector = _raw_vector.replace(False, BehaviorCodes.NOT_ANNOTATED)
        _raw_vector = _raw_vector.replace(np.nan, BehaviorCodes.NOT_ANNOTATED)
        BehaviorCodes.assert_all_are_valid(_raw_vector)
        return _raw_vector

    # @lru_cache(maxsize=8)
    def _pause(self, fluorescence_fps=False, **kwargs) -> Optional[pd.DataFrame]:
        """This is intended to be summed with the main behavioral vector"""
        try:
            df = self._raw_pause
            df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
            return df
        except NoBehaviorAnnotationsError:
            return None

    @cached_property
    def _raw_pause(self) -> Optional[pd.Series]:
        # Ulises does not really believe in this one
        if self.curvature() is None:
            return None

        # Hardcoded thresholds for what "slow" body curvature is
        # df_freq = self.hilbert_frequency(fluorescence_fps=False)
        # df_pause = df_freq.T.copy()
        # df_pause[df_pause.abs() > 0.05] = 0
        # df_pause[df_pause.abs() > 0] = 1
        # _raw_vector = df_pause.iloc[5:10].mean()
        # _raw_vector = _raw_vector > 0.25

        # Simpler: just a threshold on the speed
        _raw_vector = self.worm_speed(fluorescence_fps=False, signed=False, strong_smoothing=True) < 0.01

        # Remove any pauses that are too short (less than 0.5 seconds)
        _raw_vector = remove_short_state_changes(_raw_vector, min_length=30)

        # Convert 1's to BehaviorCodes.PAUSE and 0's to BehaviorCodes.NOT_ANNOTATED
        _raw_vector = _raw_vector.replace(True, BehaviorCodes.PAUSE)
        _raw_vector = _raw_vector.replace(False, BehaviorCodes.NOT_ANNOTATED)
        _raw_vector = _raw_vector.replace(np.nan, BehaviorCodes.NOT_ANNOTATED)
        BehaviorCodes.assert_all_are_valid(_raw_vector)
        return _raw_vector
    
    # @lru_cache(maxsize=8)
    def _hesitation(self, fluorescence_fps=False, **kwargs) -> Optional[pd.DataFrame]:
        """This is intended to be summed with the main behavioral vector"""
        try:
            df = self._raw_hesitation
            df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
            return df
        except NoBehaviorAnnotationsError:
            return None

    @cached_property
    def _raw_hesitation(self) -> Optional[pd.Series]:
        # Ulises does not really believe in this one
        if self.curvature() is None:
            return None

        # Threshold on the speed, set at about 50% of the average
        _raw_vector = self.worm_speed(fluorescence_fps=False, signed=False, strong_smoothing=True) < 0.04

        # Remove any hesitations that are too short (less than ~0.5 seconds)
        _raw_vector = remove_short_state_changes(_raw_vector, min_length=30)

        # Convert 1's to BehaviorCodes.HESITATION and 0's to BehaviorCodes.NOT_ANNOTATED
        _raw_vector = _raw_vector.replace(True, BehaviorCodes.HESITATION)
        _raw_vector = _raw_vector.replace(False, BehaviorCodes.NOT_ANNOTATED)
        _raw_vector = _raw_vector.replace(np.nan, BehaviorCodes.NOT_ANNOTATED)
        BehaviorCodes.assert_all_are_valid(_raw_vector)
        return _raw_vector

    @lru_cache(maxsize=8)
    def worm_length(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        """"""
        df = self._raw_worm_length
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_worm_length(self) -> Optional[pd.Series]:
        # Ulises does not really believe in this one
        if self.centerlineX() is None:
            return None

        # Just calculate the summed distance between all points
        x = self.centerlineX()
        y = self.centerlineY()
        _raw_vector = np.sum(np.sqrt(np.diff(x, axis=1) ** 2 + np.diff(y, axis=1) ** 2), axis=1)
        _raw_vector = pd.Series(_raw_vector, index=x.index)
        return _raw_vector

    # @lru_cache(maxsize=8)
    def _turn_annotation(self, fluorescence_fps=False, **kwargs) -> Optional[pd.DataFrame]:
        """This is intended to be summed with the main behavioral vector"""
        try:
            df = self._raw_turn_annotation
            df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
            return df
        except NoBehaviorAnnotationsError:
            return None

    @property
    def _raw_turn_annotation(self) -> Optional[pd.Series]:
        # Ulises' file uses the whole body, which gives very long turns... recalculate using the head
        # This one has a header
        # _raw_vector = read_if_exists(self.filename_turn_annotation, reader=pd.read_csv, index_col=0)
        # if _raw_vector is None:
        #     return None
        # else:
        #     _raw_vector = _raw_vector['turn']

        # See alias: ventral_only_head_curvature
        opt = dict(fluorescence_fps=False, start_segment=2, end_segment=10, do_abs=False)
        thresh = 0.035  # Threshold from looking at histograms of peaks
        _raw_ventral = (self.summed_curvature_from_kymograph(only_positive=True, **opt) > thresh)
        _raw_dorsal = (self.summed_curvature_from_kymograph(only_negative=True, **opt) > thresh)

        # Remove any turns that are too short (less than about 0.5 seconds)
        _raw_ventral = remove_short_state_changes(_raw_ventral, min_length=30)
        _raw_dorsal = remove_short_state_changes(_raw_dorsal, min_length=30)

        # Pad the edges of the surviving states
        _raw_ventral = pad_events_in_binary_vector(_raw_ventral, pad_length=30)
        _raw_dorsal = pad_events_in_binary_vector(_raw_dorsal, pad_length=30)

        # Combine
        _raw_vector = pd.Series(_raw_ventral.astype(int) - _raw_dorsal.astype(int))

        # Harcoded conversion for these files :(
        _raw_vector = _raw_vector.replace(1, BehaviorCodes.VENTRAL_TURN)
        _raw_vector = _raw_vector.replace(0, BehaviorCodes.NOT_ANNOTATED)
        _raw_vector = _raw_vector.replace(-1, BehaviorCodes.DORSAL_TURN)
        _raw_vector = _raw_vector.replace(np.nan, BehaviorCodes.NOT_ANNOTATED)
        BehaviorCodes.assert_all_are_valid(_raw_vector)
        return _raw_vector

    @lru_cache(maxsize=8)
    def _head_cast_annotation(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        """This is intended to be summed with the main behavioral vector"""
        df = self._raw_head_cast_annotation
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_head_cast_annotation(self) -> Optional[pd.Series]:
        # This one has a header
        _raw_vector = read_if_exists(self.filename_head_cast, reader=pd.read_csv, index_col=0)

        if _raw_vector is None:
            return None
        else:
            _raw_vector = _raw_vector['head_cast']
        # Harcoded conversion for these files :(
        _raw_vector = _raw_vector.replace(1, BehaviorCodes.HEAD_CAST)
        _raw_vector = _raw_vector.replace(0, BehaviorCodes.NOT_ANNOTATED)
        _raw_vector = _raw_vector.replace(np.nan, BehaviorCodes.NOT_ANNOTATED)
        BehaviorCodes.assert_all_are_valid(_raw_vector)
        return _raw_vector

    @lru_cache(maxsize=8)
    def stage_position(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        """Units of mm"""
        df = self._raw_stage_position
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_stage_position(self):
        if self.filename_table_position is None:
            raise NoBehaviorAnnotationsError("stage_position; may be missing in immobilized recordings")
        df = pd.read_csv(self.filename_table_position, index_col='time')
        df.index = pd.DatetimeIndex(df.index)
        return df

    @lru_cache(maxsize=8)
    def centerline_absolute_coordinates(self, fluorescence_fps=False, **kwargs) -> pd.DataFrame:
        """Returns a multi-index dataframe, where each body segment looks like the stage_position dataframe"""
        if self.centerlineX() is None:
            raise NoBehaviorAnnotationsError("(centerline)")
        # Depends on camera and magnification
        mm_per_pixel = 0.00245
        # Offset depends on camera and frame size
        x = (self.centerlineX(fluorescence_fps, **kwargs) - 340) * mm_per_pixel
        y = (self.centerlineY(fluorescence_fps, **kwargs) - 324) * mm_per_pixel

        # Rotation depends on Ulises' pipeline and camera
        x_abs = self.stage_position(fluorescence_fps, **kwargs).values[:, 0] - y.T
        y_abs = self.stage_position(fluorescence_fps, **kwargs).values[:, 1] + x.T

        df = pd.concat([x_abs, y_abs], keys=['X', 'Y']).swaplevel().T
        return df

    # @lru_cache(maxsize=8)
    def heading_vector(self, fluorescence_fps=False, return_radians=False, signed=True, **kwargs) -> pd.DataFrame:
        """
        Returns a series of the heading vector, as a vector of XY or radians across time

        Parameters
        ----------
        fluorescence_fps
        kwargs

        Returns
        -------

        """
        kwargs['reset_index'] = kwargs.get('reset_index', fluorescence_fps)

        df_pos = self.centerline_absolute_coordinates(fluorescence_fps=fluorescence_fps, **kwargs)
        df_phase = self.hilbert_phase(fluorescence_fps=fluorescence_fps, **kwargs)

        all_vec_means = np.zeros((len(df_pos), 2))
        for t in df_pos.index:
            seg_pairs = get_same_phase_segment_pairs(t, df_phase)
            vec_mean = get_heading_vector_from_phase_pair_segments(t, seg_pairs, df_pos)
            all_vec_means[t, :] = vec_mean

        if return_radians:
            # Requires no nan in the data
            vec_filtered = fill_nan_in_dataframe(pd.DataFrame(all_vec_means)).values
            all_vec_means = np.unwrap(np.arctan2(vec_filtered[:, 1], vec_filtered[:, 0]))
            df = pd.DataFrame(all_vec_means, index=df_pos.index, columns=['Radians'])
        else:
            if signed:
                all_vec_means = self.flip_of_vector_during_state(all_vec_means, fluorescence_fps=fluorescence_fps)
            df = pd.DataFrame(all_vec_means, index=df_pos.index, columns=['X', 'Y'])
        return df

    @cached_property
    def _raw_beh_annotation(self) -> pd.Series:
        """
        Reads behavior from the annotation file, and converts it to the BehaviorCodes enum

        Raises NoBehaviorAnnotationsError if the annotation file is not found

        Returns
        -------

        """
        if self._beh_annotation is None:
            self._beh_annotation = get_manual_behavior_annotation(behavior_fname=self.filename_beh_annotation)
        if isinstance(self._beh_annotation, pd.DataFrame):
            self._beh_annotation = self._beh_annotation.annotation
        if self._beh_annotation is not None:
            self._beh_annotation = BehaviorCodes.load_using_dict_mapping(self._beh_annotation)
        return self._beh_annotation

    @lru_cache(maxsize=8)
    def manual_beh_annotation(self, fluorescence_fps=False, keep_reversal_turns=True, **kwargs) -> \
            Optional[pd.DataFrame]:
        """Ulises' manual annotations of behavior"""
        df = self._raw_manual_beh_annotation
        if not keep_reversal_turns:
            # Map reversal turns to regular reversal state
            # Can't use regular pandas replace, because we have an enum
            # So I will find the indices myself and replace them
            ind = BehaviorCodes.vector_equality(df, BehaviorCodes.REV | BehaviorCodes.DORSAL_TURN)
            df.iloc[ind] = BehaviorCodes.REV
            ind = BehaviorCodes.vector_equality(df, BehaviorCodes.REV | BehaviorCodes.VENTRAL_TURN)
            df.iloc[ind] = BehaviorCodes.REV
        df = self._validate_and_downsample(df, fluorescence_fps, **kwargs)
        return df

    @cached_property
    def _raw_manual_beh_annotation(self) -> pd.Series:
        """
        Reads behavior from the annotation file, and converts it to the BehaviorCodes enum

        Raises NoManualBehaviorAnnotationsError if the annotation file is not found

        Returns
        -------

        """
        df = read_if_exists(self.filename_manual_beh_annotation, reader=pd.read_csv)
        if df is None:
            raise NoManualBehaviorAnnotationsError(self.filename_manual_beh_annotation)
        # Assume we only want the Annotation column
        df = df['Annotation']
        # Convert to BehaviorCodes
        df = BehaviorCodes.load_using_dict_mapping(df)
        return df

    @classmethod
    def beh_aliases_stable(cls) -> List[str]:
        """A list of behavior aliases that are stable, i.e. not currently an experimental function"""
        behaviors = ['signed_stage_speed', 'rev', 'fwd', 'abs_stage_speed',
                     'middle_body_speed', 'signed_middle_body_speed', 'signed_speed_angular',
                     'worm_speed_average_all_segments',
                     'summed_curvature',
                     'summed_signed_curvature', 'head_curvature', 'head_signed_curvature',
                     'quantile_curvature', 'quantile_head_curvature',
                     'interpolated_ventral_midbody_curvature', 'interpolated_ventral_head_curvature',
                     'interpolated_dorsal_midbody_curvature', 'interpolated_dorsal_head_curvature',
                     'speed_plateau_piecewise_constant',
                     'speed_plateau_piecewise_linear_onset', 'speed_plateau_piecewise_linear_offset',
                     'fwd_empirical_distribution', 'rev_empirical_distribution']
        behaviors.extend(BehaviorCodes.possible_behavior_aliases())
        return behaviors

    def calc_behavior_from_alias(self, behavior_alias: str, **kwargs) -> pd.Series:
        """
        This calls worm_speed or summed_curvature_from_kymograph with defined key word arguments

        Some strings call specific other functions:
            'leifer_curvature' -> summed_curvature_from_kymograph
            'pirouette' -> calc_psuedo_pirouette_state
            'plateau' -> calc_plateau_state

        Note: always has fluorescence_fps=True

        Parameters
        ----------
        behavior_alias
        kwargs

        Returns
        -------

        """

        # Default arguments
        kwargs['fluorescence_fps'] = kwargs.get('fluorescence_fps', True)
        kwargs['reset_index'] = kwargs.get('reset_index', True)

        if behavior_alias == 'raw_annotations':
            y = self.beh_annotation(**kwargs)
        elif behavior_alias == 'signed_stage_speed':
            y = self.worm_speed(**kwargs, signed=True)
        elif behavior_alias == 'rev':
            y = BehaviorCodes.vector_equality(self.beh_annotation(**kwargs), BehaviorCodes.REV)
        elif behavior_alias == 'fwd':
            y = BehaviorCodes.vector_equality(self.beh_annotation(**kwargs), BehaviorCodes.FWD)
        elif behavior_alias == 'abs_stage_speed':
            y = self.worm_speed(**kwargs)
        elif behavior_alias == 'middle_body_speed':
            y = self.worm_speed(**kwargs, use_stage_position=False, signed=False)
        elif behavior_alias == 'signed_middle_body_speed':
            y = self.worm_speed(**kwargs, use_stage_position=False, signed=True)
        elif behavior_alias == 'signed_middle_body_speed_smoothed':
            y = self.worm_speed(**kwargs, use_stage_position=False, signed=True, strong_smoothing_before_derivative=True)
        elif behavior_alias == 'summed_curvature':
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=30, **kwargs)
        elif behavior_alias == 'leifer_curvature' or behavior_alias == 'summed_signed_curvature':
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(do_abs=False, **kwargs)
        elif behavior_alias == 'head_curvature':
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph( start_segment=5, end_segment=30, **kwargs)
        elif behavior_alias == 'head_signed_curvature':
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(do_abs=False, 
                                                     start_segment=5, end_segment=30, **kwargs)
        elif behavior_alias == 'quantile_curvature':
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=10, end_segment=90,
                                                     do_quantile=True, which_quantile=0.9, **kwargs)
        elif behavior_alias == 'quantile_head_curvature':
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=5, end_segment=30,
                                                     do_quantile=True, which_quantile=0.75, **kwargs)
        elif behavior_alias == 'ventral_quantile_curvature':
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=10, end_segment=90,
                                                     do_abs=False, do_quantile=True, which_quantile=0.9, **kwargs)
        elif behavior_alias == 'dorsal_quantile_curvature':
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=10, end_segment=90,
                                                     do_abs=False, do_quantile=True, which_quantile=0.1, **kwargs)
        elif behavior_alias == 'ventral_only_curvature':
            # Same as Ulises curvature annotation
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=10, end_segment=90,
                                                     do_abs=False, only_positive=True, **kwargs)
        elif behavior_alias == 'dorsal_only_curvature':
            # Same as Ulises curvature annotation
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=10, end_segment=90,
                                                     do_abs=False, only_negative=True, **kwargs)
        elif behavior_alias == 'ventral_only_head_curvature':
            # Same as Ulises curvature annotation
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=2, end_segment=10,
                                                     do_abs=False, only_positive=True, **kwargs)
        elif behavior_alias == 'dorsal_only_head_curvature':
            # Same as Ulises curvature annotation
            self.check_has_full_kymograph()
            y = self.summed_curvature_from_kymograph(start_segment=2, end_segment=10,
                                                     do_abs=False, only_negative=True, **kwargs)
        elif behavior_alias == 'pirouette':
            y = self.calc_pseudo_pirouette_state(**kwargs)
        elif behavior_alias == 'speed_plateau_piecewise_linear_onset':
            y, _ = self.calc_piecewise_linear_plateau_state(n_breakpoints=2, return_last_breakpoint=False, **kwargs)
        elif behavior_alias == 'speed_plateau_piecewise_linear_offset':
            y, _ = self.calc_piecewise_linear_plateau_state(n_breakpoints=2, return_last_breakpoint=True, **kwargs)
        elif behavior_alias == 'speed_plateau_piecewise_constant':
            y = self.calc_constant_offset_plateau_state(**kwargs)
        elif behavior_alias == 'signed_stage_speed_strongly_smoothed':
            y = self.worm_speed(signed=True, strong_smoothing=True, **kwargs)
        elif behavior_alias == 'signed_speed_angular':
            y = self.worm_angular_velocity(**kwargs)
        elif behavior_alias == 'worm_speed_average_all_segments':
            y = self.worm_speed_average_all_segments(**kwargs)
        # elif behavior_alias == 'worm_nose_residual_speed':
        #     y = self.worm_speed_average_all_segments(fluorescence_fps=True)
        elif behavior_alias == 'fwd_counter':
            y = self.calc_counter_state(state=BehaviorCodes.FWD, **kwargs)
        elif behavior_alias == 'fwd_phase_counter':
            y = self.calc_counter_state(state=BehaviorCodes.FWD, phase_not_real_time=True, **kwargs)
        elif behavior_alias == 'rev_counter':
            y = self.calc_counter_state(state=BehaviorCodes.REV, **kwargs)
        elif behavior_alias == 'rev_phase_counter':
            y = self.calc_counter_state(state=BehaviorCodes.REV, phase_not_real_time=True, **kwargs)
        elif behavior_alias == 'fwd_empirical_distribution':
            y = self.calc_empirical_probability_to_end_state(state=BehaviorCodes.FWD, **kwargs)
        elif behavior_alias == 'rev_empirical_distribution':
            y = self.calc_empirical_probability_to_end_state(state=BehaviorCodes.REV, **kwargs)
        # elif behavior_alias == 'interpolated_ventral_midbody_curvature':
        #     y = self.calc_interpolated_curvature_using_peak_detection(i_segment=41, flip=False)
        # elif behavior_alias == 'interpolated_dorsal_midbody_curvature':
        #     y = self.calc_interpolated_curvature_using_peak_detection(i_segment=41, flip=True)
        elif behavior_alias == 'interpolated_ventral_head_curvature':
            y = self.calc_interpolated_curvature_using_peak_detection(flip=False)
        elif behavior_alias == 'interpolated_dorsal_head_curvature':
            y = self.calc_interpolated_curvature_using_peak_detection(flip=True)
        elif behavior_alias == 'interpolated_ventral_minus_dorsal_head_curvature':
            y1 = self.calc_behavior_from_alias('interpolated_ventral_head_curvature', **kwargs)
            y0 = self.calc_behavior_from_alias('interpolated_dorsal_head_curvature', **kwargs)
            y = y1 + y0
        elif behavior_alias == 'interpolated_ventral_minus_dorsal_midbody_curvature':
            y1 = self.calc_behavior_from_alias('interpolated_ventral_midbody_curvature', **kwargs)
            y0 = self.calc_behavior_from_alias('interpolated_dorsal_midbody_curvature', **kwargs)
            y = y1 + y0
        else:
            # Check if there is a BehaviorCodes enum with this name
            try:
                beh = BehaviorCodes[behavior_alias.upper()]
                y = BehaviorCodes.vector_equality(self.beh_annotation(**kwargs), beh).astype(int)
            except KeyError:
                raise NotImplementedError(behavior_alias)

        return y

    # @lru_cache(maxsize=8)
    def beh_annotation(self, fluorescence_fps=False, reset_index=False, use_manual_annotation=False,
                       include_collision=True, include_turns=True, include_head_cast=True, include_pause=True,
                       include_hesitation=True) -> \
            Optional[pd.Series]:
        """
        Name is shortened to avoid US-UK spelling confusion

        Note that _raw_beh_annotation raises NoBehaviorAnnotationsError if no behavior annotation is found
        """
        if not use_manual_annotation:
            beh = self._raw_beh_annotation
        else:
            try:
                beh = self._raw_manual_beh_annotation
            except NoManualBehaviorAnnotationsError:
                logging.warning("Requested manual annotation, but none exists")
                logging.warning("Using automatic annotation instead")
                beh = self._raw_beh_annotation

        # Add additional annotations from other files
        # Note that these other annotations are one frame shorter than the behavior annotation
        beh = beh.iloc[:-1]
        if include_collision and self._self_collision() is not None:
            beh = beh + self._self_collision(fluorescence_fps=False, reset_index=False)
        if include_pause and self._pause() is not None:
            beh = beh + self._pause(fluorescence_fps=False, reset_index=False)
        if include_hesitation and self._hesitation() is not None:
            beh = beh + self._hesitation(fluorescence_fps=False, reset_index=False)
        if include_turns and self._turn_annotation() is not None:
            # Note that the turn annotation is one frame shorter than the behavior annotation
            beh = beh + self._turn_annotation(fluorescence_fps=False, reset_index=False)
        if include_head_cast and self._head_cast_annotation() is not None:
            beh = beh + self._head_cast_annotation(fluorescence_fps=False, reset_index=False)

        # Make sure there are no nan values.
        # Necessary because sometimes removing tracking failures adds nan, even when they should be recognized
        beh_vec = self._validate_and_downsample(beh, fluorescence_fps=fluorescence_fps, reset_index=reset_index)
        beh_vec.replace(np.nan, BehaviorCodes.UNKNOWN, inplace=True)
        BehaviorCodes.assert_all_are_valid(beh_vec)
        return beh_vec

    def all_found_behaviors(self, convert_to_strings=False, **kwargs):
        beh = self.beh_annotation(**kwargs)
        beh_unique = beh.unique()
        if convert_to_strings:
            beh_unique = [behavior.individual_names for behavior in beh_unique]
            # Flatten the nested list, and only keep unique values
            beh_unique = list({item for sublist in beh_unique for item in sublist})
        return beh_unique

    @lru_cache(maxsize=64)
    def summed_curvature_from_kymograph(self, fluorescence_fps=False, start_segment=30, end_segment=80,
                                        do_abs=True, do_quantile=False, which_quantile=0.9,
                                        only_positive=False, only_negative=False, reset_index=False) -> pd.Series:
        """
        Average over value of curvature of segments
            default segments: 30 to 80
            optional: absolute value, not signed
            optional: quantile, not mean
            optional: only take positive (ventral) or negative (dorsal) values
        """
        assert sum([only_positive, only_negative, do_abs]) <= 1, \
            "Can only have one of only_positive, only_negative, or do_abs"

        mat = self.curvature().loc[:, start_segment:end_segment]
        if only_positive:
            mat = mat[mat > 0]
        if only_negative:
            # Ulises defines this as positive in the end, so flip the sign
            mat = -mat[mat < 0]
        if do_abs:
            mat = mat.abs()
        if not do_quantile:
            mat = mat.mean(axis=1)
        else:
            mat = mat.quantile(axis=1, q=which_quantile)
        # Sometimes there may be no values that are positive or negative, so the mean will be nan
        mat.fillna(0, inplace=True)
        curvature = self._validate_and_downsample(mat, fluorescence_fps=fluorescence_fps, reset_index=reset_index)
        return curvature

    ##
    ## Speed properties (derivatives)
    ##

    @cached_property
    def _raw_worm_angular_velocity(self):
        """Using angular velocity in 2d pca space"""

        xyz_pca = self.pca_projections
        window = 5
        x = remove_outliers_via_rolling_mean(pd.Series(xyz_pca[:, 0]), window)
        y = remove_outliers_via_rolling_mean(pd.Series(xyz_pca[:, 1]), window)

        # Second interpolation to get rid of nan at position 0
        x = pd.Series(x).interpolate().interpolate(method='bfill')
        y = pd.Series(y).interpolate().interpolate(method='bfill')
        # Note: arctan2 is required to give the proper sign
        angles = np.unwrap(np.arctan2(y, x))
        smoothed_angles = filter_gaussian_moving_average(pd.Series(angles), std=12)

        velocity = np.gradient(smoothed_angles)
        velocity = remove_outliers_via_rolling_mean(pd.Series(velocity), window)
        # velocity = pd.Series(velocity).interpolate()

        return velocity

    @lru_cache(maxsize=8)
    def worm_angular_velocity(self, fluorescence_fps=False, remove_outliers=True, **kwargs):
        """
        This is the angular velocity in PCA space (first two modes)

        Note: remove outliers by default
        """
        velocity = self._raw_worm_angular_velocity
        velocity = self._validate_and_downsample(velocity, fluorescence_fps=fluorescence_fps, **kwargs)
        if fluorescence_fps:
            velocity.reset_index(drop=True, inplace=True)
        if remove_outliers:
            window = 10
            velocity = remove_outliers_via_rolling_mean(pd.Series(velocity), window)
            velocity = pd.Series(velocity).interpolate()
        # Sign to be consistent with the regular speed
        stage_speed = self.worm_speed(fluorescence_fps=fluorescence_fps, signed=True, strong_smoothing=True)
        # Flip if the correlation is negative
        if np.corrcoef(stage_speed, velocity)[0, 1] < 0:
            velocity *= -1

        return velocity

    # @lru_cache(maxsize=256)
    def worm_speed(self, fluorescence_fps=False, subsample_before_derivative=True, signed=False,
                   strong_smoothing=False, use_stage_position=True, remove_outliers=True, body_segment=50,
                   clip_unrealistic_values=True, strong_smoothing_before_derivative=False, reset_index=True) -> pd.Series:
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
        reset_index - Used for compatibility, but is not used. Always True.

        Returns
        -------

        """
        if use_stage_position:
            get_positions = self.stage_position
        else:
            # Use segment 50 out of 100 by default
            get_positions = lambda fluorescence_fps: self.centerline_absolute_coordinates(
                fluorescence_fps=fluorescence_fps)[body_segment]
        if subsample_before_derivative:
            df = get_positions(fluorescence_fps=fluorescence_fps)
        else:
            df = get_positions(fluorescence_fps=False)
        if strong_smoothing_before_derivative:
            df = filter_gaussian_moving_average(df, std=5)
        # Derivative, then convert to physical units (note that subsampling might not have happened yet)
        speed = np.sqrt(np.gradient(df['X']) ** 2 + np.gradient(df['Y']) ** 2)
        tdelta_s = self.get_time_delta_in_s(fluorescence_fps and subsample_before_derivative)
        speed_mm_per_s = pd.Series(speed / tdelta_s)

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
        if clip_unrealistic_values:
            thresh = 0.5
            speed_mm_per_s = speed_mm_per_s.clip(lower=-thresh, upper=thresh)

        return speed_mm_per_s

    def worm_acceleration(self, fluorescence_fps=False, **kwargs):
        """
        Calculates derivative of speed

        Parameters
        ----------
        fluorescence_fps - Whether to downsample

        Returns
        -------

        """
        speed = self.worm_speed(fluorescence_fps=False, strong_smoothing=True, signed=True, **kwargs)
        acceleration = pd.Series(np.gradient(speed), index=speed.index)

        # Downsample after gradient
        acceleration = self._validate_and_downsample(acceleration, fluorescence_fps=fluorescence_fps, **kwargs)
        return acceleration

    def worm_speed_average_all_segments(self, start_segment=10, end_segment=90, **kwargs):
        """
        Computes the speed of each individual segment (absolute magnitude), then takes an average

        See worm_speed for options

        Parameters
        ----------
        kwargs

        Returns
        -------

        """
        # Do not sign this initial speed calculation
        sign_after_mean = kwargs.get('signed', False)
        kwargs['signed'] = False

        all_speeds = self.calc_speed_kymograph(start_segment, end_segment, **kwargs)
        mean_speed = all_speeds.mean(axis=0)

        if sign_after_mean:
            fluorescence_fps = kwargs.get('fluorescence_fps', False)
            mean_speed = self.flip_of_vector_during_state(mean_speed, fluorescence_fps)

        return mean_speed

    def calc_speed_kymograph(self, start_segment=10, end_segment=90, **kwargs):
        single_segment_opt = kwargs.copy()
        single_segment_opt['use_stage_position'] = False
        fluorescence_fps = single_segment_opt.get('fluorescence_fps', False)
        all_speeds = np.zeros(
            (end_segment - start_segment, len(self.stage_position(fluorescence_fps=fluorescence_fps))))
        for i, i_seg in enumerate(range(start_segment, end_segment)):
            single_segment_opt['body_segment'] = i_seg
            all_speeds[i, :] = self.worm_speed(**single_segment_opt)
        all_speeds = pd.DataFrame(all_speeds)
        return all_speeds

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
        all_diffs = pd.Series(df.index).diff()
        # If the recording crossed a day or daylight saving boundary, then it will have a large jump
        half_hour = pd.to_timedelta(30 * 60 * 1e9)
        invalid_ind = np.where(np.abs(all_diffs) > half_hour)[0]
        if len(invalid_ind) > 0:
            all_diffs[invalid_ind[0]-1:invalid_ind[-1]+1] = pd.to_timedelta(0)
        tdelta = all_diffs.mean()
        # To replicate the behavior of tdelta.delta
        tdelta_s = (1000*tdelta.microseconds + tdelta.nanoseconds) / 1e9
        assert tdelta_s > 0, f"Calculated negative delta time ({tdelta_s}); was there a power outage or something?"
        return tdelta_s

    def flip_of_vector_during_state(self, vector, fluorescence_fps=False, state=BehaviorCodes.REV) -> pd.Series:
        """By default changes sign during reversal"""
        BehaviorCodes.assert_is_valid(state)
        rev_ind = BehaviorCodes.vector_equality(
            self.beh_annotation(fluorescence_fps=fluorescence_fps, reset_index=True), state)
        velocity = copy.copy(vector)
        if len(velocity) == len(rev_ind):
            velocity[rev_ind] *= -1
        elif len(velocity) == len(rev_ind) + 1:
            velocity = velocity.iloc[:-1]
            try:
                velocity[rev_ind] *= -1
            except Exception as e:
                print(velocity, rev_ind)
                raise e
        elif len(velocity) == len(rev_ind) - 1:
            try:
                velocity[rev_ind.iloc[:-1]] *= -1
            except Exception as e:
                print(velocity, rev_ind)
                raise e
        else:
            raise DataSynchronizationError(f"velocity ({len(velocity)})", f"reversal indices ({len(rev_ind)})")

        return velocity

    ##
    ## Basic data validation
    ##

    @property
    def has_beh_annotation(self):
        return self.filename_beh_annotation is not None and os.path.exists(self.filename_beh_annotation)

    @property
    def has_manual_beh_annotation(self):
        return self.filename_manual_beh_annotation is not None and os.path.exists(self.filename_manual_beh_annotation)

    @property
    def has_full_kymograph(self):
        fnames = [self.filename_y, self.filename_x, self.filename_curvature]
        return all([f is not None for f in fnames]) and all([os.path.exists(f) for f in fnames])

    def check_has_full_kymograph(self):
        if not self.has_full_kymograph:
            raise NoBehaviorAnnotationsError(self.project_config.project_dir)

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
        c = np.arange(self.num_trace_frames) / 1e6
        ax.scatter(self.pca_projections[:, 0], self.pca_projections[:, 1], self.pca_projections[:, 2], c=c)
        plt.colorbar()

    def get_centerline_for_time(self, t):
        c_x = self.centerlineX().iloc[t * self.frames_per_volume]
        c_y = self.centerlineY().iloc[t * self.frames_per_volume]
        return np.vstack([c_x, c_y]).T

    def calc_triggered_average_indices(self, state=BehaviorCodes.FWD, min_duration=5, ind_preceding=20,
                                       behavior_name=None,
                                       use_manual_annotation=None,
                                       use_hilbert_phase=False,
                                       **kwargs):
        """
        Calculates a list of indices that can be used to calculate triggered averages of 'state' ONSET

        Default uses the behavior annotation, binarized via comparing to state
            See BehaviorCodes for state indices
        Alternatively, can pass a behavior_name, which will be used to look up the behavior in this class
            Note: this overrides the state parameter
            Note: unless the behavior returned has values understood by BehaviorCodes, this should be set as continuous
            using the kwargs:
                behavioral_annotation_is_continuous = True
                behavioral_annotation_threshold = threshold [OPTIONAL; see TriggeredAverageIndices]
        Alternatively, can pass a behavioral_annotation, which will be used directly
            See TriggeredAverageIndices for more details
            Note: this overrides the state and behavior_name parameters
        Similarly, use_hilbert_phase will create and use a commonly used continuous signal

        Parameters
        ----------
        state
        min_duration
        trace_len
        kwargs

        Returns
        -------

        """
        # First: Check if manual annotation can be calculated
        if use_manual_annotation is None:
            if BehaviorCodes.must_be_manually_annotated(state):
                use_manual_annotation = True
                if not self.has_manual_beh_annotation:
                    raise NoManualBehaviorAnnotationsError()
            else:
                use_manual_annotation = False
        # If the behavior is passed directly, use that
        behavioral_annotation = kwargs.get('behavioral_annotation', None)
        behavioral_annotation_for_rectification = kwargs.get('behavioral_annotation_for_rectification',
                                                             behavioral_annotation)
        # Calculate the behavioral annotation, either from an alias or directly from the behavioral annotation
        if behavioral_annotation is None:
            if use_hilbert_phase:
                df_behavioral_annotation = self.hilbert_phase(fluorescence_fps=True, reset_index=True)
                # Choose one body segment
                behavioral_annotation = df_behavioral_annotation.loc[:, 41]
                behavioral_annotation = behavioral_annotation - behavioral_annotation.mean()
                kwargs['behavioral_annotation_is_continuous'] = True
            elif behavior_name is None:
                behavioral_annotation = self.beh_annotation(fluorescence_fps=True,
                                                            use_manual_annotation=use_manual_annotation)
            else:
                behavioral_annotation = self.calc_behavior_from_alias(behavior_name)
        # This one is always from the raw annotation
        if behavioral_annotation_for_rectification is None:
            behavioral_annotation_for_rectification = self.beh_annotation(fluorescence_fps=True,
                                                                          use_manual_annotation=use_manual_annotation,
                                                                          reset_index=True)
        # Build the class
        opt = dict(behavioral_annotation=behavioral_annotation,
                   behavioral_annotation_for_rectification=behavioral_annotation_for_rectification,
                   min_duration=min_duration,
                   ind_preceding=ind_preceding,
                   trace_len=self.num_trace_frames,
                   behavioral_state=state)
        opt.update(kwargs)
        ind_class = TriggeredAverageIndices(**opt)
        return ind_class

    def calc_triggered_average_indices_with_pirouette_split(self, duration_threshold=34, **kwargs):
        """
        Calculates triggered average reversals, with a dictionary classifying them based on the previous forward state

        Specifically, if the previous forward state was longer than duration_threshold, it is an event in the
        ind_rev_pirouette return class, and if the forward was short it is in ind_rev_non_pirouette

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
        ind_short_fwd = self.calc_triggered_average_indices(state=BehaviorCodes.FWD, max_duration=duration_threshold,
                                                            **default_kwargs)
        ind_long_fwd = self.calc_triggered_average_indices(state=BehaviorCodes.FWD, min_duration=duration_threshold,
                                                           **default_kwargs)
        ind_rev = self.calc_triggered_average_indices(state=BehaviorCodes.REV, min_duration=3,
                                                      **default_kwargs)

        # Classify the reversals
        short_onsets = np.array(ind_short_fwd.idx_onsets)
        long_onsets = np.array(ind_long_fwd.idx_onsets)
        rev_onsets = np.array(ind_rev.idx_onsets)
        # Assigns 1 for onset type 1, i.e. short
        dict_of_pirouette_rev = assign_id_based_on_closest_onset_in_split_lists(short_onsets, long_onsets, rev_onsets)
        dict_of_non_pirouette_rev = {k: int(1 - v) for k, v in dict_of_pirouette_rev.items()}

        # Build new rev_onset classes based on the classes, and a flipped version
        default_kwargs.update(state=BehaviorCodes.REV, min_duration=3)
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
        binary_fwd = BehaviorCodes.vector_equality(self.beh_annotation(fluorescence_fps=True), BehaviorCodes.FWD)
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

    def calc_pseudo_pirouette_state(self, min_duration=3, window=600, std=50):
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
        ind_class = self.calc_triggered_average_indices(state=BehaviorCodes.REV, ind_preceding=0,
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

        import statsmodels.api as sm
        mod = sm.tsa.MarkovRegression(probability_to_reverse, k_regimes=2)
        res = mod.fit()
        binarized_probability_to_reverse = res.predict()
        predicted_pirouette_state = binarized_probability_to_reverse > 0.010
        # Remove padded indices
        predicted_pirouette_state = predicted_pirouette_state[pad_num:-pad_num].reset_index(drop=True)

        return predicted_pirouette_state

    def calc_constant_offset_plateau_state(self, frames_to_remove=5, DEBUG=False):
        """
        Calculates a state that is high when the worm is in a "plateau", and low otherwise
        Plateau is defined in two steps:
            1. Find all reversals that are longer than 2 * frames_to_remove
            2. Determine a single break point, and keep all points after

        Parameters
        ----------
        frames_to_remove

        Returns
        -------

        """
        from wbfm.utils.traces.triggered_averages import calc_time_series_from_starts_and_ends
        import ruptures as rpt
        from ruptures.exceptions import BadSegmentationParameters

        # Get the binary state
        beh_vec = self.beh_annotation(fluorescence_fps=True)
        rev_ind = BehaviorCodes.vector_equality(beh_vec, BehaviorCodes.REV)
        all_starts, all_ends = get_contiguous_blocks_from_column(rev_ind, already_boolean=True)
        # Also get the speed
        speed = self.worm_speed(fluorescence_fps=True, strong_smoothing_before_derivative=True)
        # Loop through all the reversals, shorten them, and calculate a break point in the middle as the new onset
        new_starts = []
        new_ends = []
        for start, end in zip(all_starts, all_ends):
            # The breakpoint algorithm needs at least 3 points
            if end - start - 2 * frames_to_remove < 3:
                continue
            dat = speed.loc[start+frames_to_remove:end-frames_to_remove].to_numpy()
            algo = rpt.Dynp(model="l2").fit(dat)
            try:
                result = algo.predict(n_bkps=1)
            except BadSegmentationParameters:
                continue
            breakpoint_absolute_coords = result[0] + start + frames_to_remove
            new_starts.append(breakpoint_absolute_coords)
            new_ends.append(end)

            if DEBUG:
                fig, ax = plt.subplots()
                plt.plot(dat)
                for r in result:
                    ax.axvline(x=r, color='black')
                plt.title(f"Start: {start}, bkps: {breakpoint_absolute_coords}, End: {end}")
                plt.show()
        if DEBUG:
            print(f"Original starts: {all_starts}")
            print(f"New starts: {new_starts}")

        num_pts = len(beh_vec)
        plateau_state = calc_time_series_from_starts_and_ends(new_starts, new_ends, num_pts, only_onset=False)
        return pd.Series(plateau_state)

    def calc_piecewise_linear_plateau_state(self, **plateau_kwargs):
        """
        Calculates a state that is high when the worm speed is in a "semi-plateau", and low otherwise
        A semi-plateau is defined in two steps:
            1. Find all reversals that are longer than min_length
            2. Fit a piecewise regression (default 3 breaks) to speed, and keep all points between the first and last breakpoints

        (this is not called a plateau, because none of the segments actually have to be flat)
        (see calc_constant_offset_plateau_state if that is needed)

        Alternatively, if return_last_breakpoint is true, return the last breakpoint as the onset and the index of the
        reversal end + end_padding as the end

        See fit_3_break_piecewise_regression and calc_plateau_state_from_trace for more details

        Parameters
        ----------
        min_length
        end_padding
        return_last_breakpoint
        DEBUG

        Returns
        -------

        """

        # Get the speed; use angular speed because sometimes the reversal annotations are wrong
        speed = self.worm_angular_velocity(fluorescence_fps=True)
        plateau_state, working_pw_fits = self.calc_plateau_state_from_trace(speed, **plateau_kwargs)
        return plateau_state, working_pw_fits

    def calc_plateau_state_from_trace(self, plateau_trace, min_length=10, start_padding=3, end_padding=3,
                                      n_breakpoints=3, return_last_breakpoint=False, replace_nan=True, DEBUG=False):
        from wbfm.utils.traces.triggered_averages import calc_time_series_from_starts_and_ends

        # Get the binary state
        beh_vec = self.beh_annotation(fluorescence_fps=True)
        rev_ind = BehaviorCodes.vector_equality(beh_vec, BehaviorCodes.REV)
        all_starts, all_ends = get_contiguous_blocks_from_column(rev_ind, already_boolean=True)
        # Loop through all the reversals, shorten them, and calculate a break point in the middle as the new onset
        new_starts_with_nan, new_ends_with_nan, new_times_series_starts, new_times_series_ends, all_pw_fits = \
            fit_piecewise_regression(plateau_trace, all_ends, all_starts, min_length,
                                     start_padding=start_padding, end_padding=end_padding,
                                     n_breakpoints=n_breakpoints,
                                     DEBUG=DEBUG)
        if return_last_breakpoint:
            new_starts_with_nan = new_ends_with_nan
            new_ends_with_nan = new_times_series_ends  # Not a fit point, but the end of the reversal with padding
        # Remove values that were nan in either the start or end
        new_starts = [s for s, e in zip(new_starts_with_nan, new_ends_with_nan) if not np.isnan(s) and not np.isnan(e)]
        new_ends = [e for s, e in zip(new_starts_with_nan, new_ends_with_nan) if not np.isnan(s) and not np.isnan(e)]
        time_series_starts = [s for s, e in zip(new_times_series_starts, new_times_series_ends) if
                              not np.isnan(s) and not np.isnan(e)]
        working_pw_fits = [fit for fit, s, e in zip(all_pw_fits, new_starts_with_nan, new_ends_with_nan) if
                           not np.isnan(s) and not np.isnan(e)]
        num_pts = len(beh_vec)
        plateau_state = calc_time_series_from_starts_and_ends(new_starts, new_ends, num_pts, only_onset=False)
        plateau_state = pd.Series(plateau_state)
        if replace_nan:
            plateau_state = plateau_state.fillna(False)
        return plateau_state, (working_pw_fits, time_series_starts)

    def plot_plateau_state(self, ax=None, **kwargs):
        # Assume there is already a plot present, and we are plotting on top
        plateau_state, (working_pw_fits, new_starts) = self.calc_plateau_state_from_trace(**kwargs)
        for fit, start in zip(working_pw_fits, new_starts):
            # The fit object has a plotting function, but the internal xx variable must be changed to be absolute
            # coordinates (they all start at 0 by default)
            plot_with_offset_x(start, fit, color="red", linewidth=2)
            # fit.plot_fit(color="red", linewidth=4)
            # fit.plot_breakpoints()
            # fit.plot_breakpoint_confidence_intervals()

    def calc_counter_state(self, state=BehaviorCodes.FWD,
                           fluorescence_fps=True, phase_not_real_time=False):
        """
        Calculates an integer vector that counts the time since last reversal

        Parameters
        ----------
        state: Which state to count (0 outside this state)
        fluorescence_fps
        phase_not_real_time: If true, will be in fraction of a forward state, rather than real time

        Returns
        -------

        """
        BehaviorCodes.assert_is_valid(state)
        # TODO: Use vector_equality
        binary_state = self.beh_annotation(fluorescence_fps=fluorescence_fps) == state
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_state, already_boolean=True)

        # Turn into time series
        num_pts = len(self.subsample_indices)
        state_trace = np.zeros(num_pts)
        for start, end in zip(all_starts, all_ends):
            time_counter = np.arange(end - start)
            if phase_not_real_time:
                time_counter = time_counter / (end - start)
            state_trace[start:end] = time_counter

        return self._shorten_to_trace_length(pd.Series(state_trace))

    def calc_empirical_probability_to_end_state(self, fluorescence_fps=True, state=BehaviorCodes.FWD):
        """
        Using an observed set of forward durations from worms with coverslip, estimates the probability to terminate
        a forward state, assuming one exponential is active at once.

        Note that this is loaded assuming a frame rate of 3.5 volumes per second

        Original was a two exponential fit, but I actually think a weibull distribution is much better

        Returns
        -------

        """
        if not fluorescence_fps:
            raise NotImplementedError("Empirical distribution is only implemented for fluorescence fps")
        if not state in (BehaviorCodes.FWD, BehaviorCodes.REV):
            raise NotImplementedError("Only fwd and rev are implemented")
        # Load the hardcoded empirical distribution
        if BehaviorCodes.FWD == state:
            duration_dict = forward_distribution_statistics()
        elif BehaviorCodes.REV == state:
            duration_dict = reverse_distribution_statistics()
        else:
            raise NotImplementedError("Only fwd and rev are implemented")
        y_dat = duration_dict['y_dat']

        # Load this dataset

        # TODO: Use vector_equality
        binary_vec = self.beh_annotation(fluorescence_fps=True) == state
        all_starts, all_ends = get_contiguous_blocks_from_column(binary_vec, already_boolean=True)

        # Turn into time series
        num_pts = len(self.subsample_indices)
        state_trace = np.zeros(num_pts)
        for start, end in zip(all_starts, all_ends):
            duration = end - start
            if duration >= len(y_dat):
                raise NotImplementedError(f"Duration {duration} is too long for the empirical distribution"
                                 f"It could be padded with 1s, but this probably means it needs to be recalculated")
            state_trace[start:end] = y_dat[:duration].copy()

        return self._shorten_to_trace_length(pd.Series(state_trace))

    def calc_interpolated_curvature_using_peak_detection(self, i_segment=41, fluorescence_fps=True, flip=False,
                                                         to_plot=False):
        # Use the curvature calculation from Ulises, not just a single segment
        if flip:
            dat = self.calc_behavior_from_alias('dorsal_only_head_curvature')
        else:
            dat = self.calc_behavior_from_alias('ventral_only_head_curvature')

        # kymo = self.curvature(fluorescence_fps=fluorescence_fps, reset_index=True).T
        # dat = kymo.iloc[i_segment, :]
        # if flip:
        #     # Ventral should be unflipped
        #     dat = -dat

        x, y_interp, interp_obj = detect_peaks_and_interpolate(dat, to_plot=to_plot, width=2)
        # if flip:
        #     y_interp = -y_interp
        return self._shorten_to_trace_length(pd.Series(y_interp))

    def calc_full_matrix_interpolated_curvature_using_peak_detection(self, fluorescence_fps=True, flip=False):

        kymo = self.curvature(fluorescence_fps=fluorescence_fps, reset_index=True)
        kymo_envelope = np.zeros_like(kymo)
        for i_seg in range(kymo.shape[1]):
            kymo_envelope[:, i_seg] = self.calc_interpolated_curvature_using_peak_detection(i_seg,
                                                                                            fluorescence_fps=fluorescence_fps,
                                                                                            flip=flip).values

        return kymo_envelope

    @staticmethod
    def load_from_project(project_data):
        # Get the relevant foldernames from the project
        # The exact files may not be in the config, so try to find them
        project_config = project_data.project_config

        # Before anything, load metadata
        frames_per_volume = get_behavior_fluorescence_fps_conversion(project_config)
        # Use the project data class to check for tracking failures
        invalid_idx = project_data.estimate_tracking_failures_from_project()

        bigtiff_start_volume = project_config.config['dataset_params'].get('bigtiff_start_volume', 0)
        opt = dict(frames_per_volume=frames_per_volume,
                   bigtiff_start_volume=bigtiff_start_volume,
                   num_trace_frames=project_data.num_frames,
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
            all_files = dict(filename_curvature=None, filename_x=None, filename_y=None, filename_beh_annotation=None,
                             filename_hilbert_amplitude=None, filename_hilbert_phase=None,
                             filename_hilbert_frequency=None, filename_hilbert_carrier=None)
            for file in Path(behavior_subfolder).iterdir():
                if not file.is_file() or file.name.startswith('.'):
                    # Skip hidden files and directories
                    continue
                if file.name.endswith('skeleton_spline_K_signed_avg.csv'):
                    all_files['filename_curvature'] = str(file)
                elif file.name.endswith('skeleton_spline_X_coords_avg.csv'):
                    all_files['filename_x'] = str(file)
                elif file.name.endswith('skeleton_spline_Y_coords_avg.csv'):
                    all_files['filename_y'] = str(file)
                elif file.name.endswith('hilbert_inst_amplitude.csv'):
                    all_files['filename_hilbert_amplitude'] = str(file)
                elif file.name.endswith('hilbert_inst_freq.csv'):
                    all_files['filename_hilbert_frequency'] = str(file)
                elif file.name.endswith('hilbert_inst_phase.csv'):
                    all_files['filename_hilbert_phase'] = str(file)
                elif file.name.endswith('hilbert_regenerated_carrier.csv'):
                    all_files['filename_hilbert_carrier'] = str(file)
                elif file.name.endswith('self_touch.csv'):
                    all_files['filename_self_collision'] = str(file)
                elif file.name.endswith('turns_annotation.csv'):
                    all_files['filename_turn_annotation'] = str(file)
                elif file.name.endswith('head_cast_ground_truth_timeseries.csv'):
                    all_files['filename_head_cast'] = str(file)

            # Third, get the table stage position
            # Should always exist IF you have access to the raw data folder (which probably means a mounted drive)
            filename_table_position = None
            fnames = [fn for fn in glob.glob(os.path.join(behavior_subfolder.parent, '*TablePosRecord.txt'))]
            if len(fnames) != 1:
                logging.warning(f"Did not find stage position file in {behavior_subfolder}")
            else:
                filename_table_position = fnames[0]
            all_files['filename_table_position'] = filename_table_position

            # Fourth, get manually annotated behavior (if it exists)
            # Note that these may have additional behaviors annotated that are not in the automatic annotation
            filename_manual_beh_annotation = None
            manual_annotation_subfolder = Path(behavior_subfolder).joinpath('ground_truth_beh_annotation')
            if manual_annotation_subfolder.exists():
                fnames = [fn for fn in glob.glob(os.path.join(manual_annotation_subfolder,
                                                              '*beh_annotation_timeseries.csv'))]
                if len(fnames) != 1:
                    logging.warning(f"Did not find manual behavior annotation file in {manual_annotation_subfolder}")
                else:
                    filename_manual_beh_annotation = fnames[0]
            all_files['filename_manual_beh_annotation'] = filename_manual_beh_annotation

        else:
            all_files = dict()

        # Get other manual behavior annotations if automatic wasn't found
        if all_files.get('filename_beh_annotation', None) is None:
            try:
                filename_beh_annotation, is_manual_style = get_manual_behavior_annotation_fname(project_config)
                opt.update(dict(beh_annotation_already_converted_to_fluorescence_fps=is_manual_style))
            except FileNotFoundError:
                # Many projects won't have either annotation
                project_config.logger.warning("Did not find behavioral annotations")
                filename_beh_annotation = None
            all_files['filename_beh_annotation'] = filename_beh_annotation
        all_files['behavior_subfolder'] = behavior_subfolder

        # Add class for converting physical units
        opt['physical_unit_conversion'] = project_data.physical_unit_conversion

        # Even if no files found, at least save the fps
        return WormFullVideoPosture(**all_files, **opt)

    def shade_using_behavior(self, **kwargs):
        """Takes care of fps conversion and new vs. old annotation format"""
        try:
            if kwargs.get('plotly_fig', None) is not None:
                # For now only works with fluorescence fps
                bh = self.beh_annotation(fluorescence_fps=True, reset_index=True)
                # Rename plotly_fig to fig
                kwargs['fig'] = kwargs.pop('plotly_fig')
                # Remove matplotlib specific kwargs
                kwargs.pop('ax', None)
                kwargs.pop('index_conversion', None)
                kwargs.pop('plot_fig', None)
                shade_using_behavior_plotly(bh, **kwargs)
            else:
                bh = self.beh_annotation(fluorescence_fps=True)
                shade_using_behavior(bh, **kwargs)
        except NoBehaviorAnnotationsError:
            pass

    @property
    def subsample_indices(self):
        # Note: sometimes the curvature and beh_annotations are different length, if one is manually created
        offset = self.frames_per_volume // 2  # Take the middle frame
        return range(self.bigtiff_start_volume*self.frames_per_volume + offset,
                     len(self._raw_stage_position),
                     self.frames_per_volume)

    def remove_idx_of_tracking_failures(self, vec: Union[pd.Series, pd.DataFrame],
                                        estimate_failures_from_kymograph=True,
                                        fluorescence_fps=True) -> pd.Series:
        """
        Removes indices of known tracking failures, if any

        Assumes the high frame rate index
        """

        # Get value to use for tracking failures
        # TODO: sometimes this check improperly fails to recognize BehaviorCodes
        if isinstance(vec, pd.DataFrame):
            invalid_value = BehaviorCodes.TRACKING_FAILURE if isinstance(vec.iat[0, 0], BehaviorCodes) else np.nan
        elif isinstance(vec, pd.Series):
            invalid_value = BehaviorCodes.TRACKING_FAILURE if isinstance(vec.iat[0], BehaviorCodes) else np.nan
        else:
            raise ValueError(f"Unknown type {type(vec)}")

        tracking_failure_idx = self.tracking_failure_idx
        if tracking_failure_idx is None and estimate_failures_from_kymograph:
            tracking_failure_idx = self.estimate_tracking_failures_from_kymo(fluorescence_fps)
        if tracking_failure_idx is not None and len(tracking_failure_idx) > 0 and vec is not None:
            vec = vec.copy()
            logging.debug(f"Setting these indices as tracking failures: {tracking_failure_idx}")
            if isinstance(vec, pd.DataFrame):
                vec.iloc[tracking_failure_idx, :] = invalid_value
            elif isinstance(vec, pd.Series):
                vec.iloc[tracking_failure_idx] = invalid_value
        return vec

    def estimate_tracking_failures_from_kymo(self, fluorescence_fps):
        kymo = self.curvature(fluorescence_fps=fluorescence_fps, reset_index=True)
        tracking_failure_idx = np.where(kymo.isnull())[0]
        return tracking_failure_idx

    def get_peaks_post_reversal(self, y: pd.Series, num_points_after_reversal=50,
                                allow_reversal_before_peak=False,
                                use_idx_of_absolute_max=False):
        """
        Calculates the peaks of a trace after each reversal period

        Parameters
        ----------
        y

        Returns
        -------

        """

        beh_annotation = self.beh_annotation(fluorescence_fps=True, reset_index=True)
        y_rev = BehaviorCodes.vector_equality(beh_annotation, BehaviorCodes.REV)
        rev_starts, rev_ends = get_contiguous_blocks_from_column(y_rev, already_boolean=True)

        peaks = []
        peak_times = []
        all_rev_ends = []
        for i, rev_end in enumerate(rev_ends):
            if rev_end == len(y):
                break
            # Check to see if there was an intervening reversal
            end_of_check_period = rev_end + num_points_after_reversal
            if not allow_reversal_before_peak and i+1 < len(rev_starts):
                # Set next reversal start to be end if it is within the check period
                end_of_check_period = rev_starts[i+1] if rev_starts[i+1] < end_of_check_period else end_of_check_period
            if not use_idx_of_absolute_max:
                idx = y.iloc[rev_end:end_of_check_period].idxmax()
            else:
                idx = y.iloc[rev_end:end_of_check_period].abs().idxmax()
            if np.isnan(idx):
                continue
            this_peak = y.iloc[idx]
            peaks.append(this_peak)
            peak_times.append(idx)
            all_rev_ends.append(rev_end)
        return peaks, peak_times, all_rev_ends

    # Raw videos
    def behavior_video_avi_fname(self):
        """Loops through the behavior subfolder and returns the first AVI file found"""
        if self.behavior_subfolder is None:
            return None
        for file in Path(self.behavior_subfolder).iterdir():
            if file.is_dir():
                continue
            if file.name.endswith('Ch0-BHbigtiff_AVG_background_subtracted.avi'):
                return file
            elif file.name.endswith('raw_stack_AVG_background_subtracted_normalised.avi'):
                # Newer naming convention
                return file
        return None

    def behavior_video_btf_fname(self, raw=False):
        """
        Note that the newer naming convention for the btf files is raw_stack_AVG_background_subtracted.btf,
        and not the file with _normalised in the name

        See behavior_video_avi_fname
        """
        if self.behavior_subfolder is None:
            return None
        for file in Path(self.behavior_subfolder).iterdir():
            if file.is_dir():
                continue
            if raw and file.name.endswith('raw_stack.btf'):
                return file
            elif file.name.endswith('Ch0-BHbigtiff_AVG_background_subtracted.btf'):
                return file
            elif file.name.endswith('raw_stack_AVG_background_subtracted.btf'):
                # Newer naming convention
                return file
        return None

    def __repr__(self):
        return \
f"=========================================\n\
Posture class with the following files:\n\
============Centerline=====================\n\
filename_x:                 {self.filename_x is not None}\n\
filename_y:                 {self.filename_y is not None}\n\
filename_curvature:         {self.filename_curvature is not None}\n\
============Annotations===================\n\
filename_beh_annotation:    {self.has_beh_annotation}\n\
============Stage Position================\n\
filename_table_position:    {self.filename_table_position is not None}\n\
=========Raw Behavior Videos==============\n\
behavior_video_avi:         {self.behavior_video_avi_fname() is not None}\n\
behavior_video_btf:         {self.behavior_video_btf_fname() is not None}\n"


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


def get_manual_behavior_annotation_fname(cfg: ModularProjectConfig, verbose=0):
    """First tries to read from the config file, and if that fails, goes searching"""

    # Initial checks are all in project local folders
    is_likely_manually_annotated = False
    try:
        behavior_cfg = cfg.get_behavior_config()
        behavior_fname = behavior_cfg.config.get('manual_behavior_annotation', None)
        if behavior_fname is not None:
            if Path(behavior_fname).exists():
                # Unclear if it is manually annotated or not
                return behavior_fname, is_likely_manually_annotated
            if not Path(behavior_fname).is_absolute():
                # Assume it is in this project's behavior folder
                behavior_fname = behavior_cfg.resolve_relative_path(behavior_fname, prepend_subfolder=True)
                if str(behavior_fname).endswith('.xlsx'):
                    # This means the user probably did it by hand... but is a fragile check
                    is_likely_manually_annotated = True
                if not os.path.exists(behavior_fname):
                    behavior_fname = None
    except FileNotFoundError:
        # Old style project
        behavior_fname = None

    if behavior_fname is not None:
        logging.warning("Note: all annotation should be in the Ulises format")
        return behavior_fname, is_likely_manually_annotated

    # Otherwise, check for other local places I used to put it
    is_likely_manually_annotated = True
    behavior_fname = "3-tracking/manual_annotation/manual_behavior_annotation.xlsx"
    behavior_fname = cfg.resolve_relative_path(behavior_fname)
    if not os.path.exists(behavior_fname):
        behavior_fname = "3-tracking/postprocessing/manual_behavior_annotation.xlsx"
        behavior_fname = cfg.resolve_relative_path(behavior_fname)
    if not os.path.exists(behavior_fname):
        behavior_fname = None
    if behavior_fname is not None:
        logging.warning("Note: all annotation should be in the Ulises format")
        return behavior_fname, is_likely_manually_annotated

    # Final checks are all in raw behavior data folders, implying they are not the stable style
    is_likely_manually_annotated = False
    raw_behavior_folder, flag = cfg.get_behavior_raw_parent_folder_from_red_fname()
    if not flag:
        return behavior_fname, is_likely_manually_annotated

    # Check if there is a manually corrected version
    manually_corrected_suffix = "beh_annotation_manual_corrected_timeseries.csv"
    behavior_fname = Path(raw_behavior_folder).joinpath(manually_corrected_suffix)
    if not behavior_fname.exists():
        # Could be named this, or have this as a suffix
        behavior_suffix = "beh_annotation.csv"
        behavior_fname = Path(raw_behavior_folder).joinpath(behavior_suffix)
        # Check if that exact file exists
        if not behavior_fname.exists():
            behavior_fname = [f for f in raw_behavior_folder.iterdir() if f.name.endswith(behavior_suffix) and
                              not f.name.startswith('.')]
            if len(behavior_fname) == 0:
                behavior_fname = None
            elif len(behavior_fname) == 1:
                behavior_fname = behavior_fname[0]
            else:
                logging.warning(f"Found multiple possible behavior annotations {behavior_fname}; taking the first one")
                behavior_fname = behavior_fname[0]

    return behavior_fname, is_likely_manually_annotated


def get_manual_behavior_annotation(cfg: ModularProjectConfig = None, behavior_fname: str = None):
    """
    Reads from a directly passed filename, or from the config file if that fails

    Parameters
    ----------
    cfg
    behavior_fname

    Returns
    -------

    """
    if behavior_fname is None:
        if cfg is not None:
            behavior_fname, is_old_style = get_manual_behavior_annotation_fname(cfg)
        else:
            # Only None was passed
            raise NoBehaviorAnnotationsError("Filename not passed")
    if behavior_fname is not None:
        if str(behavior_fname).endswith('.csv'):
            # Old style had two columns with no header, manually corrected style has a header
            if 'manual_corrected_timeseries' in str(behavior_fname):
                df_behavior_annotations = pd.read_csv(behavior_fname)
                behavior_annotations = df_behavior_annotations['Annotation']
            else:
                behavior_annotations = pd.read_csv(behavior_fname, header=1, names=['annotation'], index_col=0)
                if behavior_annotations.shape[1] > 1:
                    # Sometimes there is a messed up extra column
                    behavior_annotations = pd.Series(behavior_annotations.iloc[:, 0])
            behavior_annotations.fillna(BehaviorCodes.UNKNOWN, inplace=True)

        else:
            try:
                behavior_annotations = pd.read_excel(behavior_fname, sheet_name='behavior')['Annotation']
                behavior_annotations.fillna(BehaviorCodes.UNKNOWN, inplace=True)
            except PermissionError:
                logging.warning(f"Permission error when reading {behavior_fname} "
                                f"Do you have the excel sheet open elsewhere?")
                behavior_annotations = None
            except FileNotFoundError:
                behavior_annotations = None
    else:
        behavior_annotations = None

    if behavior_annotations is None:
        raise NoBehaviorAnnotationsError()

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


def fit_piecewise_regression(dat, all_ends, all_starts, min_length=10,
                             end_padding=0, start_padding=0,
                             n_breakpoints=3,
                             DEBUG=False,
                             DEBUG_base_fname=None):
    """
    Within a time series (dat), fit a piecewise regression model to each segment between start and end points
    given by zip(all_starts, all_ends)

    The final output is a "score" of the plateau of the piecewise regression model, which is defined differently
    depending on n_breakpoints:
        n_breakpoints=2: The score is the distance between the breaks, with no checks
        n_breakpoints=3: The score is the distance between the 1/2nd or 2/3rd break, decided in this way:
            If the absolute amplitude at the first breakpoint is much lower than the second, use the second breakpoint
            If the last breakpoint is not in the second half of the data, the fit failed and return np.nan
            Otherwise, use the first and last breakpoints

    Parameters
    ----------
    dat
    all_ends
    all_starts
    min_length
    end_padding
    start_padding
    n_breakpoints
    DEBUG

    Returns
    -------

    """
    import piecewise_regression

    new_starts = []
    new_ends = []
    new_times_series_starts = []
    new_times_series_ends = []
    all_pw_fits = []
    for i_event, (start, end) in enumerate(zip(all_starts, all_ends)):
        if end - start < min_length:
            continue
        time_series_start = start - start_padding
        time_series_end = end + end_padding
        new_times_series_starts.append(time_series_start)
        new_times_series_ends.append(time_series_end)
        y = dat.loc[time_series_start:time_series_end].to_numpy()
        x = np.arange(len(y))

        pw_fit = piecewise_regression.Fit(x, y, n_breakpoints=n_breakpoints, n_boot=25)
        all_pw_fits.append(pw_fit)
        if DEBUG:
            # Plot even if it isn't kept
            pw_fit.summary()
            # import json
            # print(json.dumps(results, indent=2, default=str))
            pw_fit.plot_data(color="grey", s=20)
            # Pass in standard matplotlib keywords to control any of the plots
            pw_fit.plot_fit(color="red", linewidth=4)
            pw_fit.plot_breakpoints()
            pw_fit.plot_breakpoint_confidence_intervals()

            if DEBUG_base_fname is None:
                plt.show()

            # Test multiple breakpoints
            ms = piecewise_regression.ModelSelection(x, y, max_breakpoints=5, n_boot=100)

        results = pw_fit.get_results()['estimates']
        if results is None:
            new_starts.append(np.nan)
            new_ends.append(np.nan)
            continue
        # Only use first and last breakpoints, but do a quality check
        if n_breakpoints >= 3:
            breakpoint_start = results['breakpoint1']['estimate']
            breakpoint2 = results['breakpoint2']['estimate']
            breakpoint_end = results['breakpoint3']['estimate']
            # If the last breakpoint is not in the second half of the data, the fit failed
            if breakpoint_end < len(x) / 2:
                new_starts.append(np.nan)
                new_ends.append(np.nan)
                continue
            # If the absolute amplitude at the first breakpoint is much lower than the second, use the second
            if np.abs(y[int(breakpoint_start)]) < np.abs(y[int(breakpoint2)]) / 2:
                breakpoint_start = breakpoint2
        else:
            # Assume that the first breakpoint is the start of the plateau, and the second is the end
            # Thus if the slope of the 3rd (final) segment is the same as the 1st segment, we don't have a plateau
            is_first_segment_slope_positive = results['alpha1']['estimate'] > 0.0
            is_last_segment_slope_positive = results['alpha3']['estimate'] > 0.0
            if is_first_segment_slope_positive == is_last_segment_slope_positive:
                new_starts.append(np.nan)
                new_ends.append(np.nan)
                continue

            # Also, the last segment should be steeper than the middle; otherwise we might have just forced a breakpoint
            # in the middle of a plateau or slow decline
            # In the same way, the first segment should be steeper than the middle
            abs_slope1 = np.abs(results['alpha1']['estimate'])
            abs_slope2 = np.abs(results['alpha2']['estimate'])
            abs_slope3 = np.abs(results['alpha3']['estimate'])
            if abs_slope2 > abs_slope1 or abs_slope2 > abs_slope3:
                new_starts.append(np.nan)
                new_ends.append(np.nan)
                continue

            breakpoint_start = results['breakpoint1']['estimate']
            breakpoint_end = results['breakpoint2']['estimate']

        start_absolute_coords = int(np.round(breakpoint_start + time_series_start))
        end_absolute_coords = int(np.round(breakpoint_end + time_series_start))
        new_starts.append(start_absolute_coords)
        new_ends.append(end_absolute_coords)

        if DEBUG:
            # Add final chosen points to the plot (if any are successfully chosen)
            plt.plot(breakpoint_start, y[int(breakpoint_start)], 'o', color='black')
            plt.plot(breakpoint_end, y[int(breakpoint_end)], 'o', color='black')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title(f"Difference: {end - start} vs {end_absolute_coords - start_absolute_coords}")

            if DEBUG_base_fname is not None:
                if not os.path.exists(DEBUG_base_fname):
                    os.makedirs(DEBUG_base_fname, exist_ok=True)
                fname = os.path.join(DEBUG_base_fname, f"time_series_with_breakpoints-{i_event}.png")
                plt.savefig(fname)

            plt.show()
    if DEBUG:
        print(f"Original starts: {all_starts}")
        print(f"New starts: {new_starts}")
    return new_starts, new_ends, new_times_series_starts, new_times_series_ends, all_pw_fits


