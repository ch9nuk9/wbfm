import numpy as np

from DLC_for_WBFM.utils.projects.utils_neuron_names import name2int_neuron


def napari_labels_from_traces_dataframe(df, neuron_name_dict=None, DEBUG=False):
    """
    Expects dataframe with positions, with column names either:
        legacy format: ['z_dlc', 'x_dlc', 'y_dlc']
        current format: ['z', 'x', 'y']

        And optionally: 'i_reindexed_segmentation' or 'label'
        (note: additional columns do not matter)

    Returns napari-ready format:
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
    properties = dict(label=[])
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
            # Get the index from the dataframe, or try to convert the column name into a label
            if 'i_reindexed_segmentation' in df[n]:
                label_vec = list(df[n]['i_reindexed_segmentation'])
            elif 'label' in df[n]:
                label_vec = [cast_int_or_nan(i) for i in df[n]['label']]
            else:
                label_vec = [name2int_neuron(n) for _ in range(t_vec.shape[0])]

        all_t_zxy = np.vstack([all_t_zxy, t_zxy])
        properties['label'].extend(label_vec)
    all_t_zxy = np.where(all_t_zxy < 0, np.nan, all_t_zxy)  # Some points are negative instead of nan
    to_keep = ~np.isnan(all_t_zxy).any(axis=1)
    all_t_zxy = all_t_zxy[to_keep, :]
    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    properties['label'] = [p for p, good in zip(properties['label'], to_keep[1:]) if good]

    # More info on text: https://github.com/napari/napari/blob/main/examples/add_points_with_text.py
    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent',
               'text': {'text': 'label'}, # Can add color or size here
               'properties': properties, 'name': 'Neuron IDs', 'blending': 'additive'}

    return options


def cast_int_or_nan(i):
    if np.isnan(i):
        return i
    else:
        return int(i)
