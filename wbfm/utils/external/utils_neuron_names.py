

def int2name_neuron(i_neuron: int, ignore_error=False) -> str:
    if 0 < i_neuron < 1000:
        return f"neuron_{i_neuron:03d}"
    elif ignore_error:
        # No way to know how big the number is
        return f"neuron_{i_neuron}"
    else:
        raise ValueError(f"Value {i_neuron} not supported. Neuron IDs should be between 1 and 999.")


def int2name_tracklet(i_tracklet: int) -> str:
    """

    Parameters
    ----------
    i_tracklet

    Returns
    -------

    """
    if 0 <= i_tracklet < 1000000:
        return f"tracklet_{i_tracklet:07d}"
    else:
        raise ValueError(f"Value {i_tracklet} not supported. Tracklet IDs should be between 0 and 999999.")


def name2int_neuron_and_tracklet(name_neuron: str) -> int:
    return int(name_neuron.split('_')[1])


def int2name_deprecated(i_neuron: int) -> str:
    return f"neuron{i_neuron}"


def int2name_using_mode(i_neuron: int, mode: str) -> str:
    if mode == 'neuron':
        return int2name_neuron(i_neuron)
    elif mode == 'tracklet':
        return int2name_tracklet(i_neuron)
    else:
        raise NotImplementedError


def int2name_dummy(i) -> str:
    return f"zzz_{i:03d}"
