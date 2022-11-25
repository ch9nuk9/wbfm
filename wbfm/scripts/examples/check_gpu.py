import tensorflow as tf
import torch


if __name__ == "__main__":
    print(f"Cuda is found by pytorch: {torch.cuda.is_available()}")
    print(f"Cuda is found by tensorflow: {tf.test.is_gpu_available()}")
