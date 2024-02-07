# import tensorflow as tf
import torch


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Cuda device is found: {device}")
    # tf.debugging.set_log_device_placement(True)
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    sz = (10000, 1000)
    X = torch.rand(*sz, device=device)
    # X_tf = tf.random.uniform(sz)

    for _ in range(1000):
        Y = torch.matmul(X.T, X)
        print("Mean value: ", Y.mean())
        # Y_tf = tf.linalg.matmul(X_tf, X_tf)
        # print("Mean value (tensorflow): ", Y_tf.mean())

    print("Finished calculations")
