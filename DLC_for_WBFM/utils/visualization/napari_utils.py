import numpy as np


def napari_labels_from_traces_dataframe(df, neuron_name_dict=None, DEBUG=False):
    """
    Expected napari-ready format:
        A dict of options, with a nested dict 'properties' and a list 'data'
        'properties' has one entry, 'labels' = long list with all points at all time
        'dat' is a list of equal length with all the dimensions (tzxy)

    Parameters
    ----------
    df
    neuron_name_dict
    DEBUG

    Returns
    -------

    """
    df.replace(0, np.NaN, inplace=True)  # DLC uses all zeros as failed tracks

    if neuron_name_dict is None:
        neuron_name_dict = {}
    all_neurons = list(df.columns.levels[0])
    t_vec = np.expand_dims(np.array(list(df.index), dtype=int), axis=1)
    # label_vec = np.ones(len(df.index), dtype=int)
    all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    properties = {'label': []}
    for n in all_neurons:
        try:
            coords = ['z_dlc', 'x_dlc', 'y_dlc']
            zxy = np.array(df[n][coords])
        except KeyError:
            coords = ['z', 'x', 'y']
            zxy = np.array(df[n][coords])
        # zxy = df[n][zxy_names].to_numpy(dtype=int)
        t_zxy = np.hstack([t_vec, zxy])
        if n in neuron_name_dict:
            # label_vec[:] = this_name
            label_vec = [neuron_name_dict[n]] * len(df.index)
            if DEBUG:
                print(f"Found named neuron: {n} = {label_vec[0]}")
        else:
            try:
                i_name = 'i_reindexed_segmentation'
                label_vec = list(df[n][i_name])
            except KeyError:
                i_name = 'label'
                label_vec = [cast_int_or_nan(i) for i in df[n][i_name]]

        all_t_zxy = np.vstack([all_t_zxy, t_zxy])
        properties['label'].extend(label_vec)
    all_t_zxy = np.where(all_t_zxy < 0, np.nan, all_t_zxy)  # Some points are negative instead of nan
    to_keep = ~np.isnan(all_t_zxy).any(axis=1)
    all_t_zxy = all_t_zxy[to_keep, :]
    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    properties['label'] = [p for p, good in zip(properties['label'], to_keep[1:]) if good]

    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent', 'text': 'label',
               'properties': properties, 'name': 'Neuron IDs'}

    return options


def cast_int_or_nan(i):
    if np.isnan(i):
        return i
    else:
        return int(i)