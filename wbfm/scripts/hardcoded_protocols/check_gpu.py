import tensorflow as tf
import torch


if __name__ == "__main__":
    print(f"Cuda is found by pytorch: {torch.cuda.is_available()}")
    print(f"Cuda is found by tensorflow: {tf.config.list_physical_devices('GPU')}")
    print(f"Cuda is built with tensorflow: {tf.test.is_built_with_cuda()}")
