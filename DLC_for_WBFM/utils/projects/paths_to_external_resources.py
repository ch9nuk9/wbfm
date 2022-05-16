import os


# As of October 2021
PRETRAINED_DLC_DIR = "/project/neurobiology/zimmer/wbfm/dlc_pretrained"


def get_pretrained_network_path(num_neurons):
    return os.path.join(PRETRAINED_DLC_DIR, f"{num_neurons}", "snapshot-100000")
