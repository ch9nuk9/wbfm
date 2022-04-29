import os
import shutil
import subprocess
from pathlib import Path
import zarr


def zip_raw_data_zarr(out_fname, delete_original=True, verbose=1):
    out_fname_zip = Path(out_fname).with_suffix('.zarr.zip')
    if verbose >= 1:
        print(f"Zipping zarr file {out_fname} to {out_fname_zip}")

    cmd = ['7z', 'a', '-tzip']
    cmd.extend([os.path.join(out_fname_zip, '.'), out_fname])

    subprocess.run(cmd)

    if delete_original:
        if verbose >= 1:
            print(f"Deleting original: {out_fname}")
        shutil.rmtree(out_fname)

    return out_fname_zip


def zarr_reader_folder_or_zipstore(fname: str):
    """Enforces readonly access"""
    if Path(fname).is_dir():
        dat = zarr.open(fname, mode='r')
    elif fname.endswith('.zarr.zip'):
        store = zarr.ZipStore(fname, mode='r')
        dat = zarr.open(store)
    else:
        raise NotImplementedError(f"Not a zarr directory or zip store: {fname}")

    return dat
