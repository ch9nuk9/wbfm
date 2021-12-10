

def int2name_neuron(i_neuron: int):
    assert 0 < i_neuron <= 1000, f"Value {i_neuron} not supported"
    return f"neuron_{i_neuron:03d}"


def int2name_tracklet(i_tracklet: int):
    assert 0 <= i_tracklet, f"Value {i_tracklet} not supported"
    return f"tracklet_{i_tracklet:03d}"


def name2int_neuron(name_neuron):
    return int(name_neuron.split('_')[1])


def int2name_deprecated(i_neuron: int):
    return f"neuron{i_neuron}"


def int2name_using_mode(i_neuron: int, mode: str):
    if mode == 'neuron':
        return int2name_neuron(i_neuron)
    elif mode == 'tracklet':
        return int2name_tracklet(i_neuron)
    else:
        raise NotImplementedError
