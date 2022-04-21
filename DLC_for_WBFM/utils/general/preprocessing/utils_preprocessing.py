from DLC_for_WBFM.utils.external.utils_zarr import zip_raw_data_zarr
from DLC_for_WBFM.utils.projects.project_config_classes import ModularProjectConfig


def zip_zarr_using_config(project_cfg: ModularProjectConfig):
    out_fname_red_7z = zip_raw_data_zarr(project_cfg.config['preprocessed_red'])
    out_fname_green_7z = zip_raw_data_zarr(project_cfg.config['preprocessed_green'])

    project_cfg.config['preprocessed_red'] = out_fname_red_7z
    project_cfg.config['preprocessed_green'] = out_fname_green_7z
    project_cfg.update_self_on_disk()
