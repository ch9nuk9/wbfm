import logging
import os
import shutil
from pathlib import Path
import zarr


def zip_raw_data_zarr(raw_fname, delete_original=True, verbose=1):
    out_fname_zip = Path(raw_fname).with_suffix('.zarr.zip')
    assert os.path.exists(raw_fname), f"Did not find original zarr at {raw_fname}"
    if verbose >= 1:
        print(f"Zipping zarr file {raw_fname} to {out_fname_zip}")

    with zarr.ZipStore(str(out_fname_zip), mode='w') as target_data_store:
        with zarr.open(str(raw_fname), mode='r') as raw_data_store:
            # Copy the data to the zip store
            zarr.copy_store(raw_data_store, target_data_store)

    if delete_original:
        if verbose >= 1:
            print(f"Deleting original: {raw_fname}")
        shutil.rmtree(raw_fname)

    return out_fname_zip


def zarr_reader_folder_or_zipstore(fname: str, depth: int = 0):
    """Enforces readonly access"""
    try:
        if Path(fname).is_dir():
            dat = zarr.open(fname, mode='r')
        elif fname.endswith('.zarr.zip'):
            store = zarr.ZipStore(fname, mode='r')
            dat = zarr.open(store)
        else:
            raise NotImplementedError(f"Not a zarr directory or zip store: {fname}")
    except OSError as e:
        if depth > 0:
            raise e
        # On Windows, if the path contains a special character, zarr.open() will fail. Retry with raw string
        dat = zarr_reader_folder_or_zipstore(rf"{fname}", depth=depth + 1)
    except zarr.errors.PathNotFoundError as e:
        logging.error(f"Corrupted zarr file found at {fname}; most likely the file or folder needs to be deleted")
        raise e

    return dat
