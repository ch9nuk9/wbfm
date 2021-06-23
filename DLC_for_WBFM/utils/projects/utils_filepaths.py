import os
from pathlib import Path, PurePosixPath, PureWindowsPath


def resolve_mounted_path_in_current_os(path: str, verbose=1):
    """
    Removes windows-specific mounted drive names (Y:, D:, etc.) and replaces them with the networked system equivalent

    Does nothing if the path is relative

    Note: This is specific to the Zimmer lab, as of 23.06.2021 (at the IMP)
    """
    is_abs = PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()
    if not is_abs:
        return path

    if verbose >= 1:
        print(f"Checking path {path} on os {os.name}...")

    # Swap mounted drive locations
    # UPDATE REGULARLY
    mounted_drive_dict = {
        'Y:': "/groups/zimmer"
    }

    for win_drive, linux_drive in mounted_drive_dict.items():
        if os.name.lower() == "linux" and path.startswith(win_drive):
            path = path.replace(win_drive, linux_drive)
        if os.name.lower() == "windows" and path.startswith(linux_drive):
            path = path.replace(linux_drive, win_drive)

    # Check for unreachable local drives
    local_drives = ['C:', 'D:']
    if os.name.lower() == "linux":
        for drive in local_drives:
            if path.startswith(drive):
                raise FileNotFoundError("File mounted to local drive; network system can't find it")

    path = str(Path(path).resolve())
    if verbose >= 1:
        print(f"Resolved path to {path}")
    return path
