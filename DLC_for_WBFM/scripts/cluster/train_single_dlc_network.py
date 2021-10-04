import argparse
from DLC_for_WBFM.utils.pipeline.dlc_pipeline import train_single_dlc_network


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train single DLC network')
    parser.add_argument('--dlc_config', default=None,
                        help='path to deeplabcut config file')
    args = parser.parse_args()
    dlc_config = args.dlc_config

    train_single_dlc_network(dlc_config)
