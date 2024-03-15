import logging

import numpy as np


def executing_in_notebook() -> bool:
    """
    Check if this code is executed in a jupyter notebook

    From: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook

    """
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def check_plotly_rendering(X: np.ndarray, max_size_for_notebook=200) -> (bool, dict):
    if not executing_in_notebook():
        return False
    if any(np.array(X.shape) > max_size_for_notebook):
        static_rendering_required = True
        logging.warning(
            f"Plotly will crash jupyter notebook if > {max_size_for_notebook} neurons (there are {X.shape}). "
            "Will render static image instead.")
    else:
        static_rendering_required = False

    if static_rendering_required:
        render_opt = dict(renderer="svg")
    else:
        render_opt = dict()
    return static_rendering_required, render_opt
