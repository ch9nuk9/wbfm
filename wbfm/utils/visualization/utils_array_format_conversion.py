import numpy as np


def dlc_to_napari_tracks(df, likelihood_thresh=0.4):
    """
    Convert a deeplabcut-style track to an array that can be visualized using:
        napari.view_tracks(dat)
    """

    # Convert tracks to napari style
    neuron_names = df.columns.remove_unused_levels().levels[0]
    # 5 columns:
    # track_id, t, z, y, x
    coords = ['z', 'y', 'x']
    all_tracks_list = []
    for i, name in enumerate(neuron_names):
        zxy_array = np.array(df[name][coords])
        t_array = np.expand_dims(np.arange(zxy_array.shape[0]), axis=1)

        # Remove low likelihood
        if 'likelihood' in df[name]:
            to_keep = df[name]['likelihood'] > likelihood_thresh
            zxy_array = zxy_array[to_keep, :]
            t_array = t_array[to_keep, :]
        id_array = np.ones_like(t_array) * i

        all_tracks_list.append(np.hstack([id_array, t_array, zxy_array]))

    return np.vstack(all_tracks_list)


def napari_tracks_from_match_list(list_of_matches, n0_zxy_raw, n1_zxy_raw, null_value=-1, t0=0):
    """
    Create a list of lists to be used with napari.add_tracks (or viewer.add_tracks)

    Parameters
    ----------
    list_of_matches
    n0_zxy_raw
    n1_zxy_raw
    null_value
    t0

    Returns
    -------

    """
    all_tracks_list = []
    for i_track, m in enumerate(list_of_matches):
        if null_value in m:
            continue

        track_m0 = [i_track, t0]
        track_m0.extend(n0_zxy_raw[m[0]])
        track_m1 = [i_track, t0 + 1]
        track_m1.extend(n1_zxy_raw[m[1]])

        all_tracks_list.append(track_m0)
        all_tracks_list.append(track_m1)
    return all_tracks_list


def napari_labels_from_frames(all_frames: dict, num_frames=1, to_flip_zxy=True) -> dict:

    all_t_zxy = np.array([[0, 0, 0, 0]], dtype=int)
    properties = {'label': []}
    for i_frame, frame in all_frames.items():
        if i_frame >= num_frames:
            break
        zxy = frame.neuron_locs
        if to_flip_zxy:
            zxy = zxy[:, [0, 2, 1]]
        num_neurons = zxy.shape[0]
        t_vec = np.ones((num_neurons, 1)) * i_frame
        t_zxy = np.hstack([t_vec, zxy])

        label_vec = list(range(num_neurons))

        all_t_zxy = np.vstack([all_t_zxy, t_zxy])
        properties['label'].extend(label_vec)

    all_t_zxy = all_t_zxy[1:, :]  # Remove dummy starter point
    options = {'data': all_t_zxy, 'face_color': 'transparent', 'edge_color': 'transparent', 'text': 'label',
               'properties': properties, 'name': 'Raw IDs'}

    return options
