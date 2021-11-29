

def int2name(i_neuron: int):
    assert 0 < i_neuron <= 1000, f"Value {i_neuron} not supported"
    return f"neuron_{i_neuron:03d}"


def name2int(name_neuron):
    return int(name_neuron.split('_')[1])


def int2name_deprecated(i_neuron: int):
    return f"neuron{i_neuron}"
