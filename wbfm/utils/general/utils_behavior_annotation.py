from enum import IntEnum, Flag, auto
from pathlib import Path
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wbfm.utils.external.utils_pandas import get_contiguous_blocks_from_column, make_binary_vector_from_starts_and_ends
from wbfm.utils.general.custom_errors import InvalidBehaviorAnnotationsError


class BehaviorCodes(Flag):
    """
    Top-level behaviors that are discretely annotated.
    Designed to work with Ulises' automatic annotations via a hardcoded mapping. See also: from_ulises_int

    Note that this should always be loaded using this mapping, not directly from integers!
    """
    # Basic automatically annotated behaviors
    FWD = auto()
    REV = auto()

    VENTRAL_TURN = auto()
    DORSAL_TURN = auto()
    SUPERCOIL = auto()  # Manually annotated
    QUIESCENCE = auto()  # Manually annotated
    SELF_COLLISION = auto()  # Annotated using a different pipeline
    HEAD_CAST = auto()  # Manually annotated

    NOT_ANNOTATED = auto()
    UNKNOWN = auto()
    TRACKING_FAILURE = auto()

    @classmethod
    def _ulises_int_2_flag(cls, flip: bool = False):
        original_mapping = {
            -1: cls.FWD,
            1: cls.REV,
            2: cls.FWD | cls.VENTRAL_TURN,
            3: cls.FWD | cls.DORSAL_TURN,
            4: cls.REV | cls.VENTRAL_TURN,
            5: cls.REV | cls.DORSAL_TURN,
            6: cls.SUPERCOIL,
            7: cls.QUIESCENCE,
            0: cls.NOT_ANNOTATED,
            -99: cls.UNKNOWN,  # Should not be in any files that Ulises produces
        }
        if flip:
            original_mapping = {v: k for k, v in original_mapping.items()}
        return original_mapping

    @classmethod
    def ulises_int_to_enum(cls, value: int) -> 'BehaviorCodes':
        """
        Convert from Ulises' integer value to the corresponding BehaviorCodes value

        HARDCODED!

        Parameters
        ----------
        value

        Returns
        -------

        """
        original_mapping = cls._ulises_int_2_flag()
        return original_mapping.get(value, cls.UNKNOWN)

    @classmethod
    def enum_to_ulises_int(cls, value: 'BehaviorCodes') -> int:
        """
        Convert from BehaviorCodes to the corresponding Ulises' integer value

        HARDCODED!

        Parameters
        ----------
        value

        Returns
        -------

        """
        original_mapping = cls._ulises_int_2_flag(flip=True)
        return original_mapping[value]

    def __add__(self, other):
        # Allows adding vectors as well
        if other in (BehaviorCodes.NOT_ANNOTATED, BehaviorCodes.UNKNOWN):
            return self
        elif self in (BehaviorCodes.NOT_ANNOTATED, BehaviorCodes.UNKNOWN):
            return other
        else:
            return self | other

    def __radd__(self, other):
        # Required for sum to work
        # https://stackoverflow.com/questions/5082190/typeerror-after-overriding-the-add-method
        return self.__add__(other)

    def __eq__(self, other):
        # Allows equality comparisons, but only between this enum
        if isinstance(other, BehaviorCodes):
            return self.value == other.value
        else:
            return False

    def __hash__(self):
        # Allows this enum to be used as a key in a dictionary
        return hash(self.value)

    @classmethod
    def _load_from_list(cls, vec: List[int]) -> pd.Series:
        """
        Load from a list of int; DO NOT USE DIRECTLY!

        Returns
        -------

        """
        return pd.Series([cls(i) for i in vec])

    @classmethod
    def load_using_dict_mapping(cls, vec: Union[pd.Series, List[int]]) -> pd.Series:
        """
        Load using the hardcoded mapping between Ulises' integers and BehaviorCodes

        See also: from_ulises_int

        Returns
        -------

        """
        beh_vec = pd.Series([cls.ulises_int_to_enum(i) for i in vec])
        cls.assert_all_are_valid(beh_vec)
        return beh_vec

    @classmethod
    def vector_equality(cls, enum_list: Union[list, pd.Series], query_enum, exact=False):
        """
        Compares a query enum to a list of enums, returning a binary vector

        By default allows complex comparisons, e.g. FWD_VENTRAL_TURN == FWD because FWD_VENTRAL_TURN is a subset of FWD

        Parameters
        ----------
        enum_list
        query_enum
        exact

        Returns
        -------

        """
        if isinstance(enum_list, pd.DataFrame):
            # Check that there is only one column, then convert to series
            assert len(enum_list.columns) == 1, "Can only compare to one column at a time"
            enum_list = enum_list.iloc[:, 0]

        if exact:
            binary_vector = [query_enum == e for e in enum_list]
        else:
            binary_vector = [query_enum in e for e in enum_list]
        if isinstance(enum_list, pd.Series):
            return pd.Series(binary_vector, index=enum_list.index)
        else:
            return pd.Series(binary_vector)

    @classmethod
    def vector_diff(cls, enum_list: Union[list, pd.Series], exact=False) -> pd.Series:
        """
        Calculates the vector np.diff, which can't be done directly because these aren't integers

        Returns
        -------

        """
        if exact:
            return pd.Series([e2 != e1 for e1, e2 in zip(enum_list.iloc[:-1], enum_list.iloc[1:])])
        else:
            return pd.Series([e2 not in e1 for e1, e2 in zip(enum_list.iloc[:-1], enum_list.iloc[1:])])

    @classmethod
    def possible_colors(cls):
        # Because I'm refactoring the colormaps to functions, I want a list to loop over
        return [cls.FWD, cls.REV, cls.SELF_COLLISION,
                cls.FWD | cls.VENTRAL_TURN, cls.FWD | cls.DORSAL_TURN,
                cls.REV | cls.VENTRAL_TURN, cls.REV | cls.DORSAL_TURN]

    @classmethod
    def shading_cmap_func(cls, query_state: 'BehaviorCodes', include_collision=False,
                          custom_shaded_state: Optional['BehaviorCodes']=None):
        """Colormap for shading on top of traces, but using 'in' logic instead of '==' logic"""
        if custom_shaded_state is not None:
            if custom_shaded_state in query_state:
                return 'lightgray'
            else:
                return None
        # Otherwise use a hardcoded colormap
        if cls.FWD in query_state:
            return None
        elif cls.REV in query_state:
            return 'lightgray'
        elif cls.SELF_COLLISION in query_state and include_collision:
            return 'red'
        else:
            return None
        # cmap = {cls.UNKNOWN: None,
        #         cls.FWD: None,
        #         cls.REV: 'lightgray',
        #         cls.SELF_COLLISION: 'red'}

    @classmethod
    def base_colormap(cls):
        # See: https://plotly.com/python/discrete-color/
        return px.colors.qualitative.Set1_r

    @classmethod
    def ethogram_cmap(cls, include_turns=True, include_reversal_turns=False):
        """Colormap for shading as a stand-alone ethogram"""
        base_cmap = cls.base_colormap()
        cmap = {cls.UNKNOWN: None,
                cls.FWD: base_cmap[0],
                cls.REV: base_cmap[1],
                # Same as FWD by default
                cls.FWD | cls.VENTRAL_TURN: base_cmap[0],
                cls.FWD | cls.DORSAL_TURN: base_cmap[0],
                cls.FWD | cls.SELF_COLLISION: base_cmap[0],
                cls.FWD | cls.SELF_COLLISION | cls.DORSAL_TURN: base_cmap[0],
                cls.FWD | cls.SELF_COLLISION | cls.VENTRAL_TURN: base_cmap[0],
                # Same as REV by default
                cls.REV | cls.VENTRAL_TURN: base_cmap[1],
                cls.REV | cls.DORSAL_TURN: base_cmap[1],
                cls.REV | cls.SELF_COLLISION: base_cmap[1],
                cls.REV | cls.SELF_COLLISION | cls.DORSAL_TURN: base_cmap[1],
                cls.REV | cls.SELF_COLLISION | cls.VENTRAL_TURN: base_cmap[1],
                # Unclear
                cls.QUIESCENCE: base_cmap[6],
                cls.QUIESCENCE | cls.VENTRAL_TURN: base_cmap[6],
                cls.QUIESCENCE | cls.DORSAL_TURN: base_cmap[6],
                }
        if include_turns:
            cmap[cls.FWD | cls.VENTRAL_TURN] = base_cmap[2]
            cmap[cls.FWD | cls.DORSAL_TURN] = base_cmap[3]
            cmap[cls.REV | cls.VENTRAL_TURN] = base_cmap[1]
            cmap[cls.REV | cls.DORSAL_TURN] = base_cmap[1]
        if include_reversal_turns:
            cmap[cls.REV | cls.VENTRAL_TURN] = base_cmap[4]
            cmap[cls.REV | cls.DORSAL_TURN] = base_cmap[5]
            cmap[cls.QUIESCENCE] = base_cmap[6]
        return cmap

    # @classmethod
    # def __contains__(cls, value):
    #     # NOTE: I would have to do a metaclass instead of a normal override to make this work
    #     # Backport the python 3.12 feature of allowing "in" to work with non-integers
    #     try:
    #         super().__contains__(value)
    #         return True
    #     except TypeError:
    #         return False

    @classmethod
    def assert_is_valid(cls, value):
        if not isinstance(value, BehaviorCodes):
            raise InvalidBehaviorAnnotationsError(f"Value {value} is not a valid behavioral code "
                                                  f"({cls._value2member_map_})")

    @classmethod
    def assert_all_are_valid(cls, vec):
        for v in vec:
            cls.assert_is_valid(v)

    @classmethod
    def is_successful_behavior(cls, value):
        """Returns True if the behavior is a successful behavior, i.e. not a tracking or other pipeline failure"""
        return value not in (cls.NOT_ANNOTATED, cls.UNKNOWN, cls.TRACKING_FAILURE)

    @classmethod
    def must_be_manually_annotated(cls, value):
        """As of 23-03-2023, everything except FWD and REV must be manually annotated"""
        if value is None:
            return False
        return value in (cls.SUPERCOIL, cls.QUIESCENCE)

    @property
    def full_name(self):
        """
        Simple states will properly return a name, but if it is a compound state it will be None by default
        ... unfortunately the enum class relies on certain names being None, so I have to have a separate
        property for this

        Returns
        -------

        """
        if self._name_ is not None:
            return self._name_
        else:
            # Convert a string like 'BehaviorCodes.DORSAL_TURN|REV' to 'DORSAL_TURN and REV'
            split_name = self.__str__().split('.')[-1]
            full_name = split_name.replace('|', ' and ')
            return full_name


def options_for_ethogram(beh_vec, shading=False, include_reversal_turns=False, include_collision=False,
                         custom_shaded_state: Optional['BehaviorCodes']=None):
    """
    Returns a list of dictionaries that can be passed to plotly to draw an ethogram

    if shading is True, then the ethogram will be partially transparent, to be drawn on top of a trace

    Parameters
    ----------
    beh_vec
    shading
    include_reversal_turns
    include_collision
    custom_shaded_state

    Returns
    -------

    """
    all_shape_opt = []
    if shading:
        cmap_func = lambda state: BehaviorCodes.shading_cmap_func(state, include_collision=include_collision,
                                                                  custom_shaded_state=custom_shaded_state)
    else:
        cmap_func = lambda state: \
            BehaviorCodes.ethogram_cmap(include_reversal_turns=include_reversal_turns).get(state, None)

    # Loop over all behaviors in the colormap (some may not be present in the vector)
    for behavior_code in BehaviorCodes.possible_colors():
        binary_behavior = BehaviorCodes.vector_equality(beh_vec, behavior_code)
        if cmap_func(behavior_code) is None:
            # Do not draw anything for this behavior
            continue
        starts, ends = get_contiguous_blocks_from_column(binary_behavior, already_boolean=True)
        color = cmap_func(behavior_code)
        for s, e in zip(starts, ends):
            # If there is an index in the behavior vector, convert the starts and ends
            # to the corresponding time
            s = beh_vec.index[s]
            if e < len(beh_vec):
                e = beh_vec.index[e]
            else:
                # If the last behavior is the same as the one we are plotting, then we need to
                # extend the end of the last block to the end of the vector
                e = beh_vec.index[-1]
            # Note that yref is ignored if this is a subplot. If yref is manually set, then it refers to the entire plot
            shape_opt = dict(type="rect", x0=s, x1=e, yref='paper', y0=0, y1=1,
                             fillcolor=color, line_width=0, layer="below")
            all_shape_opt.append(shape_opt)

    return all_shape_opt


def detect_peaks_and_interpolate(dat, to_plot=False, fig=None,
                                 height="mean", width=5):
    """
    Builds a time series approximating the highest peaks of an oscillating signal

    Returns the interpolation class, which has the location and value of the peaks themselves

    Parameters
    ----------
    dat
    to_plot

    Returns
    -------

    """

    # Get peaks
    if height == "mean":
        height = np.mean(dat)
    peaks, properties = find_peaks(dat, height=height, width=width)
    y_peaks = dat[peaks]

    # Interpolate
    interp_obj = interp1d(peaks, y_peaks, kind='cubic', bounds_error=False, fill_value="extrapolate")
    x = np.arange(len(dat))
    y_interp = interp_obj(x)

    if to_plot:
        if fig is None:
            plt.figure(dpi=200, figsize=(10, 5))
        plt.plot(dat, label="Raw data")
        plt.scatter(peaks, y_peaks, c='r', label="Detected peaks")
        plt.plot(x, y_interp, label="Interpolated envelope")#, c='tab:purple')
        plt.title("Envelope signal interpolated between peaks")
        plt.legend()

    return x, y_interp, interp_obj


def approximate_behavioral_annotation_using_pc1(project_cfg):
    """
    Uses the first principal component of the traces to approximate annotations for forward and reversal
    IMPORTANT: Although pc0 should correspond to rev/fwd, the sign of the PC is arbitrary, so we need to check
    that the sign is correct. Currently there's no way to do that without ID'ing a neuron that should correlate to fwd
    or rev, and checking that the sign is correct
    TODO: Add a check for the sign of the PC

    Saves an excel file within the project's behavior folder, and updates the behavioral config

    This file should be found by get_manual_behavior_annotation_fname

    Parameters
    ----------
    project_cfg

    Returns
    -------

    """
    # Load project
    from wbfm.utils.projects.finished_project_data import ProjectData
    project_data = ProjectData.load_final_project_data_from_config(project_cfg)

    # Calculate traces of the project
    opt = dict(interpolate_nan=True,
               filter_mode='rolling_mean',
               min_nonnan=0.9,
               nan_tracking_failure_points=True,
               nan_using_ppca_manifold=True,
               channel_mode='dr_over_r_50')
    df_traces = project_data.calc_default_traces(**opt)
    from wbfm.utils.visualization.filtering_traces import fill_nan_in_dataframe
    df_traces_no_nan = fill_nan_in_dataframe(df_traces, do_filtering=True)
    # Then PCA
    pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
    pipe.fit(df_traces_no_nan.T)

    # Using a threshold of 0, assign forward and reversal
    pc0 = pipe.steps[1][1].components_[0, :]
    starts, ends = get_contiguous_blocks_from_column(pd.Series(pc0) > 0, already_boolean=True)
    beh_vec = pd.DataFrame(make_binary_vector_from_starts_and_ends(starts, ends, pc0, pad_nan_points=(5, 0)),
                           columns=['Annotation'])
    # Should save using Ulises' convention, because that's what all other files are using
    beh_vec[beh_vec == 1] = BehaviorCodes.enum_to_ulises_int(BehaviorCodes.REV)
    beh_vec[beh_vec == 0] = BehaviorCodes.enum_to_ulises_int(BehaviorCodes.FWD)

    # Save within the behavior folder
    beh_cfg = project_data.project_config.get_behavior_config()
    fname = 'immobilized_beh_annotation'
    beh_cfg.save_data_in_local_project(beh_vec, fname,
                                       prepend_subfolder=True, suffix='.xlsx', sheet_name='behavior')
    beh_cfg.config['manual_behavior_annotation'] = str(Path(fname).with_suffix('.xlsx'))
    beh_cfg.update_self_on_disk()

    return beh_vec


def shade_using_behavior(beh_vector, ax=None, behaviors_to_ignore=(BehaviorCodes.SELF_COLLISION, ),
                         cmap=None, index_conversion=None, custom_shaded_state: Optional['BehaviorCodes']=None,
                         include_collision=False, DEBUG=False):
    """
    Shades current plot using a 3-code behavioral annotation:
        Invalid data (no shade)
        FWD (no shade)
        REV (gray)

    See BehaviorCodes for valid codes

    See options_for_ethogram for a plotly-compatible version

    Parameters
    ----------
    beh_vector - vector of behavioral codes
    ax - axis to plot on
    behaviors_to_ignore - list of behaviors to ignore. See BehaviorCodes for valid codes
    cmap - colormap to use. See BehaviorCodes for default
    index_conversion - function to convert indices from the beh_vector to the plot indices
    DEBUG

    Returns
    -------

    """
    if cmap is None:
        cmap = lambda state: BehaviorCodes.shading_cmap_func(state, include_collision=include_collision,
                                                             custom_shaded_state=custom_shaded_state)
    if ax is None:
        ax = plt.gca()

    # Get all behaviors that exist in the data and the cmap
    beh_vector = pd.Series(beh_vector)
    data_behaviors = beh_vector.unique()
    cmap_behaviors = pd.Series(BehaviorCodes.possible_colors())
    # Note that this returns a numpy array in the end
    all_behaviors = pd.concat([pd.Series(data_behaviors), pd.Series(cmap_behaviors)]).unique()

    # Remove behaviors to ignore
    if behaviors_to_ignore is not None:
        for b in behaviors_to_ignore:
            all_behaviors = all_behaviors[all_behaviors != b]
    for b in [BehaviorCodes.UNKNOWN, BehaviorCodes.NOT_ANNOTATED, BehaviorCodes.TRACKING_FAILURE]:
        all_behaviors = all_behaviors[all_behaviors != b]

    # Loop through the remaining behaviors, and use the binary vector to shade per behavior
    beh_vector = pd.Series(beh_vector)
    for b in all_behaviors:
        binary_vec = BehaviorCodes.vector_equality(beh_vector, b)
        color = cmap(b)
        if color is None:
            continue

        # Get the start and end indices of the binary vector
        starts, ends = get_contiguous_blocks_from_column(binary_vec, already_boolean=True)
        for start, end in zip(starts, ends):
            if index_conversion is not None:
                ax_start = index_conversion[start]
                if end >= len(index_conversion):
                    # Often have an off by one error
                    ax_end = index_conversion[-1]
                else:
                    ax_end = index_conversion[end]
            else:
                ax_start = start
                ax_end = end

            ax.axvspan(ax_start, ax_end, alpha=0.9, color=color, zorder=-10)


def shade_triggered_average(ind_preceding, xlim=None, behavior_shading_type='fwd', ax=None):
    if xlim is None:
        if ax is None:
            xlim = plt.xlim()
        else:
            xlim = ax.get_xlim()
        # xlim has to be int
        xlim = (int(xlim[0]), int(xlim[1]))
    # Shade using behavior either before or after the ind_preceding line
    if behavior_shading_type is not None:
        # Initialize empty
        beh_vec = np.array([BehaviorCodes.FWD for _ in range(xlim[1] - xlim[0])])
        # beh_vec = np.array([BehaviorCodes.FWD for _ in range(int(np.ceil(xlim[1])))])
        if behavior_shading_type == 'fwd':
            # If 'fwd' triggered, the shading should go BEFORE the line
            beh_vec[:ind_preceding] = BehaviorCodes.REV
            # beh_vec[:xlim[0] + ind_preceding] = BehaviorCodes.REV
        elif behavior_shading_type == 'rev':
            # If 'rev' triggered, the shading should go AFTER the line
            beh_vec[ind_preceding:] = BehaviorCodes.REV
        else:
            raise ValueError(f"behavior_shading must be 'rev' or 'fwd', not {behavior_shading_type}")
        # Set up index conversion: ind_preceding should be set to 0
        # index_conversion = np.arange(xlim[0] - ind_preceding, xlim[1] - ind_preceding)

        # Shade
        shade_using_behavior(beh_vec, ax=ax)#, index_conversion=index_conversion)
