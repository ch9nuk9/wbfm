import numpy as np
from plotly.figure_factory._dendrogram import _Dendrogram
import scipy.cluster.hierarchy as sch
import scipy as scp
from plotly.graph_objs import graph_objs


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

    def get_dendrogram_traces(
        self, X, colorscale, distfun, linkagefun, hovertext, color_threshold=None,
            Z=None, link_color_func=None
    ):
        if link_color_func is None:
            # Then default to the original method
            return super().get_dendrogram_traces(
                X, colorscale, distfun, linkagefun, hovertext, color_threshold
            )
        else:
            assert color_threshold is None, "Cannot use both color_threshold and link_color_func"

        # Otherwise, use the custom method, which is almost entirely copied from the original
        if Z is None:
            d = distfun(X)
            Z = linkagefun(d)
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
        colors = self.get_color_dict(colorscale)

        trace_list = []

        for i in range(len(icoord)):
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
            color_key = color_list[i]
            hovertext_label = None
            if hovertext:
                hovertext_label = hovertext[i]
            trace = dict(
                type="scatter",
                x=np.multiply(self.sign[self.xaxis], xs),
                y=np.multiply(self.sign[self.yaxis], ys),
                mode="lines",
                marker=dict(color=colors[color_key]),
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
    dendrogram = _Dendrogram(**kwargs)
    return graph_objs.Figure(data=dendrogram.data, layout=dendrogram.layout)
