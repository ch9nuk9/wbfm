import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def visualize_model_performance(c):
    """
    Plots the correlation matrix of the model's output, which should be close to diagonal (if trained)

    Parameters
    ----------
    c

    Returns
    -------

    """
    all_vals = c.cpu().numpy()

    norm = TwoSlopeNorm(vmin=np.nanmin(all_vals), vcenter=0, vmax=np.nanmax(all_vals))
    fig = plt.imshow(all_vals[:10, :10], norm=norm, cmap='PiYG')
    plt.colorbar()
    # plt.show()

    return fig
