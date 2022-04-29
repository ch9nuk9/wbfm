import shutil
import subprocess
from pathlib import Path

import zarr


def zip_raw_data_zarr(out_fname, delete_original=True, verbose=1):
    out_fname_7z = Path(out_fname).with_suffix('.7zarr')
    if verbose >= 1:
        print(f"Zipping zarr file {out_fname} to {out_fname_7z}")

    cmd = ['7z', 'a', '-tzip']
    cmd.extend([out_fname_7z, out_fname])

    subprocess.run(cmd)

    if delete_original:
        if verbose >= 1:
            print(f"Deleting original: {out_fname}")
        shutil.rmtree(out_fname)

    return out_fname_7z


def zarr_reader_folder_or_zipstore(fname: str):
    """Enforces readonly access"""
    if Path(fname).is_dir():
        dat = zarr.open(fname, mode='r')
    elif fname.endswith('.7zarr'):
        store = zarr.ZipStore(fname, mode='r')
        dat = zarr.open(store)
    else:
        raise NotImplementedError(f"Not a zarr directory or zip store: {fname}")

    return dat
