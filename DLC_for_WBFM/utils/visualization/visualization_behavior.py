import logging
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt


def shade_using_behavior(bh, ax=None, behaviors_to_ignore='none',
                         cmap=None,
                         DEBUG=False):
    """
    Shades current plot using a 3-code behavioral annotation:
        -1 - Invalid data (no shade)
        0 - FWD (no shade)
        1 - REV (gray)
        2 - Turn (red)
        3 - Quiescent (light blue)
    """

    if cmap is None:
        cmap = {0: None,
                1: 'darkgray',
                2: 'red',
                3: 'lightblue'}
    if ax is None:
        ax = plt.gca()
    bh = np.array(bh)

    block_final_indices = np.where(np.diff(bh))[0]
    block_final_indices = np.concatenate([block_final_indices, np.array([len(bh) - 1])])
    block_values = bh[block_final_indices]
    if DEBUG:
        print(block_values)
        print(block_final_indices)

    if behaviors_to_ignore != 'none':
        for b in behaviors_to_ignore:
            cmap[b] = None

    block_start = 0
    for val, block_end in zip(block_values, block_final_indices):
        if val is None or np.isnan(val):
            # block_start = block_end + 1
            continue
        try:
            color = cmap.get(val, None)
        except TypeError:
            logging.warning(f"Ignored behavior of value: {val}")
            # Just ignore
            continue
        # finally:
        #     block_start = block_end + 1

        if DEBUG:
            print(color, val, block_start, block_end)
        if color is not None:
            ax.axvspan(block_start, block_end, alpha=0.3, color=color)

        block_start = block_end + 1
