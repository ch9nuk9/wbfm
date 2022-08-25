import torch


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Cuda device is found: {device}")

    sz = (10000, 10000)
    X = torch.rand(*sz, device=device)

    for _ in range(50):
        Y = torch.matmul(X.T, X)
        print("Mean value: ", Y.mean())

    print("Finished calculations")
