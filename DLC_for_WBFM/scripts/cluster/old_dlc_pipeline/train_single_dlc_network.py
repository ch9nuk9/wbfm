import argparse
import gc
import tensorflow as tf
from DLC_for_WBFM.utils.pipeline.dlc_pipeline import train_single_dlc_network
import os
# Might be necessary on tf 2.3.0 and the GPU on the cluster... see:
# https://github.com/tensorflow/tensorflow/issues/41990
# See also this for discussions of this being a design flaw in tensorflow:
# https://github.com/tensorflow/tensorflow/issues/36465
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train single DLC network')
    parser.add_argument('--dlc_config', default=None,
                        help='path to deeplabcut config file')
    args = parser.parse_args()
    dlc_config = args.dlc_config

    train_single_dlc_network(dlc_config)

    # Unclear if this is necessary or works:
    # https://github.com/tensorflow/tensorflow/issues/36465
    tf.keras.backend.clear_session()
    gc.collect()
