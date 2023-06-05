import numpy as np
import pandas as pd


def calculate_self_collision_using_pairwise_distances(X_centerline, Y_centerline, i_head=3):
    """
    Uses two matrices X_centerline and Y_centerline (positions of the body segments) to calculate the self-collision

    Summary of the approach:
    1. Calculate the pairwise distances between the head and other points
    2. Average the top quantiles to get a time series
    3. z-score and smooth
    4. Binarize using a threshold

    Parameters
    ----------
    X_centerline
    Y_centerline
    i_head

    Returns
    -------

    """

    # Final shape should be (x_coord, n_timepoints, n_segments)
    xy = np.stack([X_centerline, Y_centerline])

    # Calculate the pairwise distances between the head and other points
    all_dist = np.linalg.norm(xy - xy[:, :, i_head:i_head+1], axis=0)

    # Average several quantiles (0.1 to 0.2) to get a time series
    quantiles = np.arange(0.1, 0.2, 0.01)
    all_dist_quantiles = np.nanquantile(all_dist, quantiles, axis=1)
    all_dist_quantiles = np.nanmean(all_dist_quantiles, axis=0)

    # z-score and smooth
    all_dist_quantiles = (all_dist_quantiles - np.nanmean(all_dist_quantiles)) / np.nanstd(all_dist_quantiles)
    all_dist_quantiles = pd.Series(all_dist_quantiles)
    all_dist_quantiles = all_dist_quantiles.rolling(5, center=True, min_periods=1).mean()

    # Binarize using a threshold
    threshold = -2.5
    binary_self_collision = all_dist_quantiles < threshold

    return binary_self_collision, all_dist_quantiles
