import os
import zarr
import pandas as pd
from pathlib import Path
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import ModularProjectConfig
from wbfm.utils.external.utils_zarr import zip_raw_data_zarr


def unpack_nwb_to_project_structure(project_dir, nwb_path=None):
    """
    Unpack an NWB file into the expected on-disk project structure for the pipeline.
    
    Usage:
        unpack_nwb_to_project_structure("/path/to/project", "/path/to/project/nwb/yourfile.nwb")

    Parameters
    ----------
    project_dir : str or Path
        Path to the root of the project (should contain nwb/ with the .nwb file).
    nwb_path : str or Path
        Path to the .nwb file.
    """
    # 1. Load project
    if nwb_path is None:
        # Load the project directory and try find the NWB file
        project_data = ProjectData.load_final_project_data(project_dir, allow_hybrid_loading=True)
        cfg = project_data.project_config
        nwb_cfg = cfg.get_nwb_config()
        nwb_path = nwb_cfg.resolve_relative_path_from_config("nwb_fname")
        if not nwb_path.exists():
            raise FileNotFoundError(f"Expected NWB file at {nwb_path}; otherwise, please provide the nwb_path argument.")
    else:
        # This should be within a project already
        project_data = ProjectData.load_final_project_data_from_nwb(nwb_path)

    # Write preprocessed videos as zarr
    if project_data.red_data is not None and project_data.green_data is not None:
        project_data.logger.info("Writing preprocessed videos as zarr")
        # Ensure the preprocessed directory exists
        preproc_cfg = cfg.get_preprocessing_config()
        preproc_dir = preproc_cfg.absolute_self_path
        if not preproc_dir.exists():
            raise FileNotFoundError(f"Expected preprocessed directory at {preproc_dir}")
        # These don't have a default name, so make one
        red_zarr_path = preproc_dir / "preprocessed_red.zarr"
        green_zarr_path = preproc_dir / "preprocessed_green.zarr"
        zarr.save_array(red_zarr_path, project_data.red_data, chunks=(1,)+project_data.red_data.shape[1:])
        zarr.save_array(green_zarr_path, project_data.green_data, chunks=(1,)+project_data.green_data.shape[1:])
        # Then zip these folders
        red_zarr_zip_path = zip_raw_data_zarr(red_zarr_path)
        green_zarr_zip_path = zip_raw_data_zarr(green_zarr_path)
        # Update the config file with these paths; this is actually the main config
        cfg.config['red_fname'] = str(red_zarr_zip_path)
        cfg.config['green_fname'] = str(green_zarr_zip_path)
        cfg.update_self_on_disk()
    else:
        project_data.logger.info("No preprocessed video data found in the NWB file.")

    # Write segmentation as zarr
    if project_data.raw_segmentation is not None:
        segment_cfg = cfg.get_segmentation_config()
        seg_dir = segment_cfg.absolute_self_path
        if not seg_dir.exists():
            raise FileNotFoundError(f"Expected segmentation directory at {seg_dir}")
        
        seg_zarr_path = segment_cfg.resolve_relative_path_from_config("output_masks")
        zarr.save_array(seg_zarr_path, project_data.raw_segmentation, chunks=(1,)+project_data.segmentation.shape[1:])

        # Update the config with the segmentation path
        segment_cfg.config['segmentation_fname'] = str(seg_zarr_path)
        segment_cfg.update_self_on_disk()
    else:
        project_data.logger.info("No raw segmentation data found in the NWB file.")

    # Write final segmentation, if available
    if project_data.segmentation is not None:
        traces_cfg = cfg.get_traces_config()
        final_seg_dir = traces_cfg.absolute_self_path
        if not final_seg_dir.exists():
            raise FileNotFoundError(f"Expected final segmentation directory at {final_seg_dir}")
        reindexed_masks_path = traces_cfg.resolve_relative_path_from_config("reindexed_masks")
        zarr.save_array(reindexed_masks_path, project_data.segmentation, chunks=(1,)+project_data.segmentation.shape[1:])
        reindexed_masks_path_zip = zip_raw_data_zarr(reindexed_masks_path)
        # Update the config with the reindexed masks path
        traces_cfg.config['reindexed_masks'] = str(reindexed_masks_path_zip)
        traces_cfg.update_self_on_disk()
    else:
        project_data.logger.info("No final segmentation data found in the NWB file.")

    # # Write traces as h5
    # traces_dir = Path(project_dir) / "3-traces"
    # traces_dir.mkdir(exist_ok=True)
    # red_traces_path = traces_dir / "red_traces.h5"
    # green_traces_path = traces_dir / "green_traces.h5"
    # project_data.red_traces.to_hdf(red_traces_path, key="traces")
    # project_data.green_traces.to_hdf(green_traces_path, key="traces")

    # # Write segmentation metadata if available
    # if project_data.segmentation_metadata is not None:
    #     meta_path = seg_dir / "segmentation_metadata.pkl"
    #     project_data.segmentation_metadata.save(meta_path)

    print(f"Project structure populated from NWB at {project_dir}")
