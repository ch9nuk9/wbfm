"""
The top level function for getting final traces from 3d tracks and neuron masks
"""

# Experiment tracking
import os
import sacred
from wbfm.utils.external.utils_zarr import zip_raw_data_zarr
from wbfm.utils.visualization.plot_traces import make_default_summary_plots_using_config
from sacred import Experiment
from sacred import SETTINGS
# main function
from wbfm.utils.external.monkeypatch_json import using_monkeypatch
from wbfm.utils.projects.utils_project_status import check_all_needed_data_for_step
from wbfm.pipeline.traces import match_segmentation_and_tracks_using_config, extract_traces_using_config, \
    reindex_segmentation_using_config
from wbfm.utils.projects.project_config_classes import ModularProjectConfig, SubfolderConfigFile
from wbfm.utils.projects.utils_project import safe_cd

import cgitb
cgitb.enable(format='text')

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
# Add single variable so that the cfg() function works
ex.add_config(project_path=None, DEBUG=False)


@ex.config
def cfg(project_path, DEBUG):
    # Manually load yaml files
    cfg = ModularProjectConfig(project_path)
    cfg.setup_logger('step_4.log')
    project_dir = cfg.project_dir

    seg_cfg = cfg.get_segmentation_config()
    tracking_cfg = cfg.get_tracking_config()
    traces_cfg = cfg.get_traces_config()

    check_all_needed_data_for_step(cfg, 4, training_data_required=False)

    if not DEBUG:
        using_monkeypatch()
        # log_dir = cfg.get_log_dir()
        # ex.observers.append(TinyDbObserver(log_dir))


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    DEBUG = _config['DEBUG']
    seg_cfg = _config['seg_cfg']
    track_cfg = _config['tracking_cfg']
    traces_cfg: SubfolderConfigFile = _config['traces_cfg']
    project_cfg = _config['cfg']

    # Set environment variables to (try to) deal with rare blosc decompression errors
    os.environ["BLOSC_NOLOCK"] = "1"
    os.environ["BLOSC_NTHREADS"] = "1"

    with safe_cd(_config['project_dir']):
        # Overwrites matching pickle object; nothing needs to be reloaded
        match_segmentation_and_tracks_using_config(seg_cfg,
                                                   track_cfg,
                                                   traces_cfg,
                                                   project_cfg,
                                                   DEBUG=DEBUG)

        # Creates segmentations indexed to tracking
        new_mask_fname = reindex_segmentation_using_config(traces_cfg, seg_cfg, project_cfg)

        # Zips the reindexed segmentations to shrink requirements
        out_fname_zip = zip_raw_data_zarr(new_mask_fname)
        relative_fname = traces_cfg.unresolve_absolute_path(out_fname_zip)
        traces_cfg.config['reindexed_masks'] = relative_fname
        traces_cfg.update_self_on_disk()

        # Reads masks from disk, and writes traces
        extract_traces_using_config(project_cfg, traces_cfg, name_mode='neuron', DEBUG=DEBUG)

        # By default make some visualizations
        make_default_summary_plots_using_config(project_cfg)
