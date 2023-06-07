import numpy as np
import pandas as pd
import tifffile
from tqdm.auto import tqdm


def apply_function_to_pages(video_fname,
                            func,
                            num_pages=None):
    """
    Applies 'func' to each xy plane of an ome tiff, looping over pages

    This loop goes over pages, and makes sense when the metadata is corrupted

    """
    dat = []

    with tifffile.TiffFile(video_fname, multifile=False) as tif:
        if num_pages is None:
            num_pages = len(tif.pages)
        for i, page in enumerate(tif.pages):

            if i >= num_pages:
                break

            if i % 100 == 0:
                print(f'Page {i}/{num_pages}')
            dat.append(func(page.asarray()))

    return np.array(dat)


##
## Behavior summary statistics
##

from collections import defaultdict


def calc_net_displacement(p):
    # Units: mm
    i_seg = 50

    df = p.worm_posture_class.centerline_absolute_coordinates()
    xy0 = df.loc[0, :][i_seg]
    xy1 = df.iloc[-1, :][i_seg]

    return np.linalg.norm(xy0 - xy1)


def calc_cumulative_displacement(p):
    # Units: mm
    i_seg = 50

    df = p.worm_posture_class.centerline_absolute_coordinates()[i_seg]
    dist = np.sqrt((df['X'] - df['X'].shift()) ** 2 + (df['Y'] - df['Y'].shift()) ** 2)
    line_integral = np.nansum(dist)

    return line_integral


def calc_displacement_dataframes(all_projects):
    all_displacements = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        all_displacements['net'][name] = calc_net_displacement(p)
        all_displacements['cumulative'][name] = calc_cumulative_displacement(p)
    df_displacement_gcamp = pd.DataFrame(all_displacements)

    return df_displacement_gcamp


def calc_speed_vector(p, speed_type):
    # Units: mm/2
    worm = p.worm_posture_class
    return worm.calc_behavior_from_alias(speed_type)


def calc_speed_dataframe(all_projects):
    speed_types = ['abs_stage_speed', 'middle_body_speed', 'signed_middle_body_speed',
                   'worm_speed_average_all_segments', 'signed_speed_angular']

    all_speeds = defaultdict(dict)
    for name, p in tqdm(all_projects.items()):
        try:
            for speed_type in speed_types:
                all_speeds[speed_type][name] = calc_speed_vector(p, speed_type)
        except ValueError:
            continue
    df_speed = pd.DataFrame(all_speeds)

    return df_speed
