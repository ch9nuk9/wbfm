

def int2name_neuron(i_neuron: int) -> str:
    assert 0 < i_neuron <= 1000, f"Value {i_neuron} not supported"
    return f"neuron_{i_neuron:03d}"


def int2name_tracklet(i_tracklet: int) -> str:
    """

    Parameters
    ----------
    i_tracklet

    Returns
    -------

    """
    assert 0 <= i_tracklet <= 1000000, f"Value {i_tracklet} not supported"
    return f"tracklet_{i_tracklet:07d}"


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
