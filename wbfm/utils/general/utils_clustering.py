import numpy as np
import scipy
from dash_bio.component_factory._clustergram import _Clustergram
from plotly.figure_factory._dendrogram import _Dendrogram
import scipy.cluster.hierarchy as sch
import scipy as scp
from plotly.graph_objs import graph_objs
import plotly.graph_objs as go

from wbfm.utils.external.utils_jupyter import check_plotly_rendering


def ks_statistic(x, y, axis=0):
    """
    Designed for use with scipy.stats.permutation_test

    Vectorize, such that the axis (0) is batches
    Expects dimensions of: batch, time, samples

    Parameters
    ----------
    x
    y
    axis

    Returns
    -------

    """

    pipeline = lambda x: np.nanmedian(x, axis=-1)
    x_cumsum = pipeline(x)
    y_cumsum = pipeline(y)
    xy_diff = np.nanmax(np.abs(x_cumsum - y_cumsum), axis=-1)
    return xy_diff


class CustomDendrogram(_Dendrogram):
    """
    Very similar to the base plotly class, but allows additional args to be passed to scipy.hierarchy.dendrogram

    Specifically, allows a custom list of clusters to be passed, producing more complex clusters than a simple cutoff
    """

    def __init__(self, Z, link_color_func=None, **kwargs):
        self.Z = Z
        self.link_color_func = link_color_func
        if 'X' not in kwargs:
            kwargs['X'] = None
        super().__init__(**kwargs)

    def get_dendrogram_traces(self, X, colorscale, distfun, linkagefun, hovertext, color_threshold=None):
        if self.link_color_func is None:
            # Then default to the original method
            return super().get_dendrogram_traces(
                X, colorscale, distfun, linkagefun, hovertext, color_threshold
            )
        else:
            assert color_threshold is None, "Cannot use both color_threshold and link_color_func"
            link_color_func = self.link_color_func

        # Otherwise, use the custom method, which is almost entirely copied from the original
        if self.Z is None:
            d = distfun(X)
            Z = linkagefun(d)
        else:
            Z = self.Z
        P = sch.dendrogram(
            Z,
            orientation=self.orientation,
            labels=self.labels,
            no_plot=True,
            link_color_func=link_color_func,
        )

        icoord = scp.array(P["icoord"])
        dcoord = scp.array(P["dcoord"])
        ordered_labels = scp.array(P["ivl"])
        color_list = scp.array(P["color_list"])
        # colors = self.get_color_dict(colorscale)

        trace_list = []

        # Modify to just use the same colors as the original
        for i, color in enumerate(color_list):
            # xs and ys are arrays of 4 points that make up the '∩' shapes
            # of the dendrogram tree
            if self.orientation in ["top", "bottom"]:
                xs = icoord[i]
            else:
                xs = dcoord[i]

            if self.orientation in ["top", "bottom"]:
                ys = dcoord[i]
            else:
                ys = icoord[i]
            # color_key = color_list[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(
                type="scatter",
                x=np.multiply(self.sign[self.xaxis], xs),
                y=np.multiply(self.sign[self.yaxis], ys),
                mode="lines",
                # marker=dict(color=colors[color_key]),
                marker=dict(color=color),
                text=hovertext_label,
                hoverinfo="text",
            )

            try:
                x_index = int(self.xaxis[-1])
            except ValueError:
                x_index = ""

            try:
                y_index = int(self.yaxis[-1])
            except ValueError:
                y_index = ""

            trace["xaxis"] = "x" + x_index
            trace["yaxis"] = "y" + y_index

            trace_list.append(trace)

        return trace_list, icoord, dcoord, ordered_labels, P["leaves"]


def custom_create_dendrogram(**kwargs):
    """
    Similar to plotly's create_dendrogram, but allows a custom list of clusters to be passed, producing more complex
    clusters than a simple cutoff. Uses the CustomDendrogram class

    Parameters
    ----------
    X
    kwargs

    Returns
    -------

    """
    dendrogram = CustomDendrogram(**kwargs)
    return graph_objs.Figure(data=dendrogram.data, layout=dendrogram.layout)


class CustomClustergram(_Clustergram):
    """
    Similar to base class, but uses CustomDendrogram class

    Note that the plotly library uses the Clustergram function to initialize the class, and we combine that in one

    """

    def __init__(self, Z=None, link_color_func=None, to_show=True, **kwargs):
        self.Z = Z
        self.link_color_func = link_color_func
        super().__init__(**kwargs)

        (fig, ct, curves_dict) = self.figure()

        self.fig = go.Figure(fig)  # This actually plots?
        self.ct = ct
        self.curves_dict = curves_dict

        if to_show:
            if 'X' in kwargs:
               X = kwargs['X']
               static_rendering_required, render_opt = check_plotly_rendering(X)
               self.fig.show(**render_opt)
            else:
                # Can't display the figure if we don't have the data
                pass

    def _compute_clustered_data(self):
        """
        Same as super class, but uses custom dendrogram
        """

        # initialize return dict
        trace_list = {"col": [], "row": []}

        clustered_column_ids = self._column_ids
        clustered_row_ids = self._row_ids

        # cluster the data and calculate dendrogram

        # For the custom dendrogram, we need to pass the Z matrix, which is the linkage matrix
        # This can be calculated from the data, or passed in directly
        opt = dict(Z=self.Z, link_color_func=self.link_color_func)

        # allow referring to protected member
        # columns
        if self._cluster in ["col", "all"]:
            cols_dendro = CustomDendrogram(
                X=np.transpose(self._data),
                orientation="bottom",
                labels=self._column_ids,
                distfun=lambda X: self._dist_fun(X, metric=self._col_dist),
                linkagefun=lambda d: self._link_fun(
                    d, optimal_ordering=self._optimal_leaf_order
                ),
                **opt
            )
            clustered_column_ids = cols_dendro.labels
            trace_list["col"] = cols_dendro.data

        # rows
        if self._cluster in ["row", "all"]:
            rows_dendro = CustomDendrogram(
                X=self._data,
                orientation="right",
                labels=self._row_ids,
                distfun=lambda X: self._dist_fun(X, metric=self._row_dist),
                linkagefun=lambda d: self._link_fun(
                    d, optimal_ordering=self._optimal_leaf_order
                ),
                **opt
            )
            clustered_row_ids = rows_dendro.labels
            trace_list["row"] = rows_dendro.data

        # now, we need to rearrange the data array to fit the labels

        # first get reordered indices
        rl_indices = [self._row_ids.index(r) for r in clustered_row_ids]
        cl_indices = [self._column_ids.index(c) for c in clustered_column_ids]

        # modify the data here; first shuffle rows,
        # then transpose and shuffle columns,
        # then transpose again
        clustered_data = self._data[rl_indices].T[cl_indices].T

        return trace_list, clustered_data, clustered_row_ids, clustered_column_ids


def get_node_from_tree(tree: scipy.cluster.hierarchy.ClusterNode, id: int, node_history=None):
    """
    Traverses scipy dendrogram tree to get the node with the given id and the full history

    Returns node_history in order from root to leaf, not including the node itself
        Note that the last element should be the same for all function calls, because everything is in one cluster

    Parameters
    ----------
    tree
    id

    Returns
    -------

    """

    if node_history is None:
        node_history = []
    if tree.id == id:
        return tree, node_history
    else:
        if tree.is_leaf():
            return None, node_history
        else:
            left_result, node_history = get_node_from_tree(tree.left, id, node_history)
            if left_result is not None:
                node_history.append(tree)
                return left_result, node_history
            right_result, node_history = get_node_from_tree(tree.right, id, node_history)
            if right_result is not None:
                node_history.append(tree)
                return right_result, node_history
    return None, node_history
