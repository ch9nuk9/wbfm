import os
from collections import defaultdict
from functools import partial
from typing import Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import plotly.express as px

from wbfm.gui.utils.utils_dash import save_folder_for_two_dataframe_dashboard
from wbfm.utils.external.utils_behavior_annotation import BehaviorCodes
from wbfm.utils.external.utils_pandas import cast_int_or_nan
from wbfm.utils.general.custom_errors import NoBehaviorAnnotationsError
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.traces.gui_kymograph_correlations import get_manual_annotation_from_project, \
    build_new_column_from_function
from wbfm.utils.traces.triggered_averages import FullDatasetTriggeredAverages


def build_all_gui_dfs_triggered_averages(all_projects_gcamp: Dict[str, ProjectData],
                                         all_projects_gfp: Dict[str, ProjectData],
                                         output_folder: str = None,
                                         trace_options=None,
                                         trigger_options=None,
                                         include_raw_behavioral_annotation=False,
                                         include_speed=False,
                                         **kwargs):
    """
    Builds all the dataframes needed for the GUI for triggered average style plots.

    Parameters
    ----------
    all_projects_gcamp
    all_projects_gfp
    output_folder
    trace_options
    posture_attribute
    kwargs

    Returns
    -------

    """
    # Use same trace options for all plots
    if trace_options is None:
        trace_options = {}
    if trigger_options is None:
        trigger_options = dict(state=BehaviorCodes.REV)
    opt = dict(interpolate_nan=False,
               filter_mode='rolling_mean',
               min_nonnan=0.9,
               nan_tracking_failure_points=True,
               nan_using_ppca_manifold=False,  # This takes a long time!
               channel_mode='dr_over_r_50')
    opt.update(trace_options)

    # Main dataframes
    output_dict_rev, output_dict_fwd, output_dict_traces = calc_all_triggered_average_dictionaries(
        all_projects_gcamp, opt, trigger_options)
    output_dict_rev_gfp, output_dict_fwd_gfp, output_dict_traces_gfp = calc_all_triggered_average_dictionaries(
        all_projects_gfp, opt, trigger_options)

    # Dict with all ind_classes
    ind_class_dict_gcamp = output_dict_rev['triggered_averages_class']
    ind_class_dict_gfp = output_dict_rev_gfp['triggered_averages_class']
    
    # Make dictionary with fwd and rev to loop over
    # If the user specified a specific state, then only use that one
    if 'state' in trigger_options:
        all_dict_rev_fwd = dict(custom=[output_dict_rev, output_dict_rev_gfp])
    else:
        all_dict_rev_fwd = dict(reversal_triggered=[output_dict_rev, output_dict_rev_gfp],
                                forward_triggered=[output_dict_fwd, output_dict_fwd_gfp])

    # Loop over all the dictionaries and save the dataframes
    for fname_suffix, value in all_dict_rev_fwd.items():
        
        # Unpack the dictionaries
        output_dict, output_dict_gfp = value

        # Combine all dataframes to one
        df_gcamp = calc_summary_df(output_dict, 'gcamp')
        df_gfp = calc_summary_df(output_dict_gfp, 'gfp')
        df_gcamp_gfp = pd.concat([df_gcamp, df_gfp])
    
        # Determine effect size "significance" via a 5% cutoff of the gfp
        # i.e. find the %5 quantile of the absolute effect sizes
        x_effect_line = df_gfp['effect size'][df_gfp['p value'] < 0.05].abs().quantile(0.95)
    
        # Create additional complex columns
        df_gcamp_gfp['-log(p value)'] = -np.log(df_gcamp_gfp['p value'])
        hline_height = - np.log(0.05)

        # Save dataframes
        # TODO: the display of the triggered trace doesn't work for the forward triggered dataframes
        df_summary, raw_dfs = reformat_dataframes(all_projects_gcamp, all_projects_gfp, df_gcamp_gfp,
                                                  output_dict_traces, output_dict_traces_gfp,
                                                  ind_class_dict_gcamp, ind_class_dict_gfp,
                                                  include_raw_behavioral_annotation=include_raw_behavioral_annotation,
                                                  include_speed=include_speed, state=trigger_options['state'])

        # Optionally add a raw dataframe with the exact behavioral variable

        # Additional columns: manually id'ed neuron name
        all_projects = {**all_projects_gcamp, **all_projects_gfp}
        func = partial(get_manual_annotation_from_project, all_projects)
        df_index = list(df_summary['index'])
        df_new_col = build_new_column_from_function(df_index, func)
        df_summary['manual_id'] = df_new_col.values

        if output_folder is not None:
            this_output_folder = f"{output_folder}-{fname_suffix}"
            opt = dict(output_foldername=this_output_folder, suffix=fname_suffix)

            # Actually save the dataframes
            save_folder_for_two_dataframe_dashboard(this_output_folder, df_summary, raw_dfs)

            # Summary png plots
            plot_triggered_average_with_lines(df_gcamp_gfp, x_effect_line, hline_height, **opt)
            plot_triggered_average_with_gray(df_gcamp_gfp, x_effect_line, **opt)
            plot_triggered_average_boxplot(df_gcamp_gfp, x_effect_line, **opt)


def reformat_dataframes(all_projects_gcamp, all_projects_gfp, df_gcamp_gfp_rev, output_dict_traces,
                        output_dict_traces_gfp, ind_class_dict_gcamp, ind_class_dict_gfp,
                        include_raw_behavioral_annotation, include_speed, state):
    # Index of summary df must be the same as the columns of the raw df (traces)
    df_summary = df_gcamp_gfp_rev.copy()
    df_summary['index'] = df_summary['dataset_name'] + '_' + df_summary['neuron_name']
    # Traces
    df_all_traces_gcamp = calc_raw_traces_df(output_dict_traces)
    df_all_traces_gfp = calc_raw_traces_df(output_dict_traces_gfp)
    df_all_traces = pd.concat([df_all_traces_gcamp, df_all_traces_gfp], axis=1)
    df_all_traces.columns = ['_'.join(col).strip() for col in df_all_traces.columns.values]
    raw_dfs = {'trace': df_all_traces}
    # Reversal (behavior)
    opt = dict(beh_mode='reversal')
    df_beh = build_beh_df(df_all_traces_gcamp, all_projects=all_projects_gcamp, **opt)
    df_beh_gfp = build_beh_df(df_all_traces_gfp, all_projects=all_projects_gfp, **opt)
    df_beh = pd.concat([df_beh, df_beh_gfp], axis=1)
    df_beh.columns = ['_'.join(col).strip() for col in df_beh.columns.values]
    raw_dfs.update({'reversal annotation': df_beh})
    if include_raw_behavioral_annotation:
        # Behavior from ind class directly
        opt = dict(beh_mode='individual class')
        df_beh = build_beh_df(df_all_traces_gcamp, ind_class_dict=ind_class_dict_gcamp, **opt)
        df_beh_gfp = build_beh_df(df_all_traces_gfp, ind_class_dict=ind_class_dict_gfp, **opt)
        df_beh = pd.concat([df_beh, df_beh_gfp], axis=1)
        df_beh.columns = ['_'.join(col).strip() for col in df_beh.columns.values]

        state_name = BehaviorCodes(state).name
        raw_dfs.update({state_name: df_beh})
    if include_speed:
        # Behavior from ind class directly
        opt = dict(beh_mode='speed')
        df_beh = build_beh_df(df_all_traces_gcamp, all_projects=all_projects_gcamp, **opt)
        df_beh_gfp = build_beh_df(df_all_traces_gfp, all_projects=all_projects_gfp, **opt)
        df_beh = pd.concat([df_beh, df_beh_gfp], axis=1)
        df_beh.columns = ['_'.join(col).strip() for col in df_beh.columns.values]
        raw_dfs.update({'speed': df_beh})

    return df_summary, raw_dfs


def calc_all_triggered_average_dictionaries(all_projects, trace_opt, trigger_opt=None):
    output_dict_rev = defaultdict(dict)
    output_dict_fwd = defaultdict(dict)
    output_dict_traces = dict()
    if trigger_opt is None:
        trigger_opt = dict()

    kwargs = dict(significance_calculation_method='ttest')
    default_trigger_opt = dict(ind_preceding=30)
    default_trigger_opt.update(trigger_opt)

    for proj_name, proj in tqdm(all_projects.items()):
        # First, reversal triggered
        try:
            triggered_averages_class = FullDatasetTriggeredAverages.load_from_project(proj,
                                                                                      trigger_opt=default_trigger_opt,
                                                                                      trace_opt=trace_opt,
                                                                                      **kwargs)
        except NoBehaviorAnnotationsError:
            continue

        significant_neurons, p_values, effect_sizes = triggered_averages_class.which_neurons_are_significant(
            num_baseline_lines=1000)

        output_dict_rev['significant_neurons'][proj_name] = significant_neurons
        output_dict_rev['num_significant_neurons'][proj_name] = len(significant_neurons)
        output_dict_rev['p_values'][proj_name] = p_values
        output_dict_rev['effect_sizes'][proj_name] = effect_sizes
        # Note: this is the same for both fwd and rev, and the behavioral state is overwritten below
        output_dict_rev['triggered_averages_class'][proj_name] = triggered_averages_class

        # Second, forward triggered, but only if the user didn't specify a state
        if 'state' not in trigger_opt:

            triggered_averages_class_fwd = FullDatasetTriggeredAverages.load_from_project(proj,
                                                                                          trigger_opt=default_trigger_opt,
                                                                                          trace_opt=trace_opt,
                                                                                          **kwargs)
            triggered_averages_class_fwd.ind_class.behavioral_state = BehaviorCodes.FWD
            significant_neurons, p_values, effect_sizes = triggered_averages_class_fwd.which_neurons_are_significant(
                num_baseline_lines=1000)

            output_dict_fwd['significant_neurons'][proj_name] = significant_neurons
            output_dict_fwd['num_significant_neurons'][proj_name] = len(significant_neurons)
            output_dict_fwd['p_values'][proj_name] = p_values
            output_dict_fwd['effect_sizes'][proj_name] = effect_sizes
            output_dict_fwd['triggered_averages_class'][proj_name] = triggered_averages_class_fwd

        # Final things
        num = len(proj.well_tracked_neuron_names(0.9))
        output_dict_rev['num_well_tracked_neurons'][proj_name] = num
        output_dict_fwd['num_well_tracked_neurons'][proj_name] = num

        output_dict_traces[proj_name] = triggered_averages_class.df_traces

    return output_dict_rev, output_dict_fwd, output_dict_traces


def calc_raw_traces_df(output_dict_traces, datatype_str='gcamp'):
    d = output_dict_traces
    df_effect_p = pd.concat(d.values(), keys=d.keys(), axis=1)
    return df_effect_p


def calc_summary_df(all_p_values, datatype_str='gcamp'):
    df_list = []

    for dataset_name in all_p_values['effect_sizes'].keys():

        p_values = all_p_values['p_values'][dataset_name]
        effect_size = all_p_values['effect_sizes'][dataset_name]

        df_tmp = pd.DataFrame.from_dict([p_values, effect_size]).T
        df_tmp.reset_index(inplace=True)
        df_tmp.columns = ['neuron_name', 'p value', 'effect size']
        df_tmp['dataset_name'] = dataset_name
        df_tmp['significant'] = df_tmp['p value'] < 0.05

        df_list.append(df_tmp)

    df_effect_p = pd.concat(df_list)
    df_effect_p['genotype'] = datatype_str
    return df_effect_p


def plot_triggered_average_with_lines(df_gcamp_gfp_rev, x_effect_line, hline_height, output_foldername,
                                      suffix='reversal_triggered'):
    # Set control to be cool color
    col_sequence = px.colors.qualitative.G10.copy()
    col_sequence = [col_sequence[1], col_sequence[0]]

    fig = px.scatter(df_gcamp_gfp_rev, y='-log(p value)', x='effect size', color='genotype', marginal_y='histogram',
                     title=f'{suffix} neurons', hover_data=['neuron_name', 'dataset_name'],
                     color_discrete_sequence=col_sequence)

    opt = dict(line=dict(dash='dot'), col=1)
    fig.add_vline(x=x_effect_line, **opt)
    fig.add_vline(x=-x_effect_line, **opt)
    fig.add_hline(y=hline_height, **opt)
    fig.show()

    fname = os.path.join(output_foldername, f'statistics_volcano_num_significant_{suffix}.png')
    fig.write_image(fname)


def plot_triggered_average_with_gray(df_gcamp_gfp_rev, x_effect_line, output_foldername,
                                     suffix='reversal_triggered'):
    # Additional volcano plot: uninteresting in gray
    col_sequence = px.colors.qualitative.G10.copy()
    col_sequence = ['#7F7F7F', col_sequence[1], col_sequence[0]]

    fwd_ind = (df_gcamp_gfp_rev['p value'] < 0.05) & (df_gcamp_gfp_rev['effect size'] > x_effect_line)
    rev_ind = (df_gcamp_gfp_rev['p value'] < 0.05) & (df_gcamp_gfp_rev['effect size'] < -x_effect_line)
    df_gcamp_gfp_rev['activity type'] = 'insignificant'
    df_gcamp_gfp_rev.loc[fwd_ind, 'activity type'] = 'reverse'
    df_gcamp_gfp_rev.loc[rev_ind, 'activity type'] = 'forward'

    fig = px.scatter(df_gcamp_gfp_rev, y='-log(p value)', x='effect size', color='activity type',
                     title='Reversal-triggered neurons', hover_data=['neuron_name', 'dataset_name'],
                     color_discrete_sequence=col_sequence)
    fig.show()

    fname = os.path.join(output_foldername, f'statistics_volcano_{suffix}_insignificant_gray.png')
    fig.write_image(fname)


def plot_triggered_average_boxplot(df_gcamp_gfp_rev, x_effect_line, output_foldername,
                                      suffix='reversal_triggered'):
    # Column based on final evaluation of 'interesting'
    # i.e. low p value and high effect size
    df_gcamp_gfp_rev['number of interesting neurons'] = (df_gcamp_gfp_rev['p value'] < 0.05) & (
                df_gcamp_gfp_rev['effect size'].abs() > x_effect_line)

    # New dataframe that summarizes how many 'interesting' per dataset and genotype there are
    dataset_names = df_gcamp_gfp_rev['dataset_name'].unique()
    new_df_dict = {}
    for name in dataset_names:
        num_interesting = df_gcamp_gfp_rev[df_gcamp_gfp_rev['dataset_name'] == name][
            'number of interesting neurons'].sum()
        # Get genotype in hacky way
        genotype = df_gcamp_gfp_rev[df_gcamp_gfp_rev['dataset_name'] == name]['genotype'].iat[0]
        new_df_dict[name] = [num_interesting, genotype]

    df_dataset_summary = pd.DataFrame(new_df_dict).T
    df_dataset_summary.columns = ['number of interesting neurons', 'genotype']

    # Control in cool color
    col_sequence = px.colors.qualitative.G10.copy()
    col_sequence = [col_sequence[1], col_sequence[0]]

    fig = px.box(df_dataset_summary, color='genotype', y='number of interesting neurons',
                 title='Reversal-triggered neurons',
                 color_discrete_sequence=col_sequence)
    fig.show()

    fname = os.path.join(output_foldername, f'statistics_boxplot_num_significant_{suffix}.png')
    fig.write_image(fname)


def build_beh_df(df_all_traces, beh_mode='reversal',
                 all_projects=None, beh_code=BehaviorCodes.REV, ind_class_dict=None):
    """
    Builds a dataframe with the same columns as the output of calc_raw_traces_df, but with the behavioral annotation
    instead of the raw traces.

    This will have repeated columns for each neuron within a single dataset

    The exact behavioral time series is determined by beh_mode:
    - 'reversal' (default): uses the reversal annotation from the posture class
    - 'individual class': uses the individual class annotation from the posture class
    - 'speed': uses the speed annotation from the posture class

    Parameters
    ----------
    df_all_traces
    beh_mode
    all_projects
    beh_code
    ind_class_dict

    Returns
    -------

    """
    assert all_projects is not None or ind_class_dict is not None, "Must provide either all_projects or ind_class_dict"
    top_cols = list(df_all_traces.columns.levels[0])
    df_beh = df_all_traces.copy()

    for dataset_name in top_cols:
        if beh_mode == 'reversal':
            p = all_projects[dataset_name]
            beh = p.worm_posture_class.beh_annotation(fluorescence_fps=True, reset_index=True) == beh_code
            beh = beh.apply(cast_int_or_nan)
        elif beh_mode == 'individual class':
            triggered_average_class = ind_class_dict[dataset_name]
            beh = triggered_average_class.ind_class.cleaned_binary_state.reset_index(drop=True)
            beh = beh.apply(cast_int_or_nan)
        elif beh_mode == 'speed':
            p = all_projects[dataset_name]
            beh = p.worm_posture_class.worm_speed(fluorescence_fps=True,
                                                  strong_smoothing_before_derivative=True,
                                                  signed=True)
            beh = beh / beh.std()
        else:
            raise NotImplementedError(f"Behavior mode {beh_mode} not implemented, "
                                      f"must be 'reversal' or 'individual class' or 'speed'")

        # Make dataframe of all beh, then overwrite
        df_one_dat = pd.DataFrame({k: beh for k in list(df_beh[dataset_name].columns)})
        df_beh[dataset_name] = df_one_dat

    return df_beh
