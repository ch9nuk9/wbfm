from collections import defaultdict

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn import manifold
import matplotlib.cm as cm


def test(dataloader, model, loss_fn, device="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    correct_per_class = defaultdict(int)
    total_per_class = defaultdict(int)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            correct_vec = (pred.argmax(1) == y).type(torch.float)
            correct += correct_vec.sum().item()

            for k, v in zip(y, correct_vec):
                k = int(k.to("cpu").numpy())
                correct_per_class[k] += v.to("cpu").numpy()
                total_per_class[k] += 1
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 *correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return correct_per_class, total_per_class


def plot_accuracy(correct_per_class=None, total_per_class=None):
    plt.figure(figsize=(10, 5))

    x = list(correct_per_class.keys())
    y = []
    for i in x:
        y.append(correct_per_class[i] / total_per_class[i])

    sns.barplot(x=x, y=y)


def embed_all_points(dataloader, model, device="cpu"):
    model.eval()

    all_embeddings = defaultdict(list)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            embed = model.embed(X)

            for k, v in zip(y, embed):
                k = int(k.to("cpu").numpy())
                v = v.to("cpu").numpy()
                all_embeddings[k].append(v)

    for k, v in all_embeddings.items():
        all_embeddings[k] = np.vstack(v)

    return all_embeddings


def tsne_plot_embeddings(all_feature_spaces=None):
    tsne = manifold.TSNE(
        n_components=2,
        init="random",
        random_state=0,
        perplexity=100,
        n_iter=300,
    )

    all_x_tsne = tsne.fit_transform(np.vstack(all_feature_spaces))

    all_lens = list(map(len, all_feature_spaces))

    colors = cm.nipy_spectral(np.linspace(0, 1, len(all_feature_spaces)))

    plt.figure(figsize=(15, 10))

    last_len = 0
    for i, this_len in enumerate(all_lens):
        this_len += last_len
        plt.scatter(all_x_tsne[last_len:this_len, 0], all_x_tsne[last_len:this_len, 1],
                    c=np.expand_dims(colors[i], axis=0))
        last_len = this_len
