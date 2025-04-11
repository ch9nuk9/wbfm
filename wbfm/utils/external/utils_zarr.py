import logging
import os
import shutil
import subprocess
from pathlib import Path
import zarr


def zip_raw_data_zarr(raw_fname, delete_original=True, verbose=1):
    out_fname_zip = Path(raw_fname).with_suffix('.zarr.zip')
    assert os.path.exists(raw_fname), f"Did not find original zarr at {raw_fname}"
    if verbose >= 1:
        print(f"Zipping zarr file {raw_fname} to {out_fname_zip}")

    cmd = ['7z', 'a', '-tzip']
    cmd.extend([str(out_fname_zip), os.path.join(str(raw_fname), '.')])

    subprocess.run(cmd)

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
