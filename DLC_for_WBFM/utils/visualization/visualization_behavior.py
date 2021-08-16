import numpy as np
from matplotlib import pyplot as plt


def shade_using_behavior(bh, ax=None, behaviors_to_ignore='none', DEBUG=False):
    """
    Shades current plot using a 3-code behavioral annotation:
        0 - FWD (no shade)
        1 - REV (gray)
        2 - Turn (red)
    """

    if ax is None:
        ax = plt.gca()
    bh = np.array(bh)

    block_final_indices = np.where(np.diff(bh))[0]
    block_values = bh[block_final_indices]
    if DEBUG:
        print(block_values)
        print(block_final_indices)

    cmap = {0: None,
            1: 'gray',
            2: 'red'}
    if behaviors_to_ignore != 'none':
        for b in behaviors_to_ignore:
            cmap[b] = None

    block_start = 0
    for val, block_end in zip(block_values, block_final_indices):
        color = cmap[val]
        if DEBUG:
            print(color, val, block_start, block_end)
        if color is not None:
            ax.axvspan(block_start, block_end, alpha=0.5, color=color)
            # plt.axvspan(block_start, block_end, alpha=0.5, color=color)

        block_start = block_end + 1
