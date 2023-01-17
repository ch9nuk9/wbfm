import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patheffects import withSimplePatchShadow
import mplcursors
import pandas as pd


def example():
    # From: https://mplcursors.readthedocs.io/en/stable/examples/dataframe.html
    df = pd.DataFrame(
        dict(
            Suburb=["Ames", "Somerset", "Sawyer"],
            Area=[1023, 2093, 723],
            SalePrice=[507500, 647000, 546999],
        )
    )

    df.plot.scatter(x="Area", y="SalePrice", s=100)

    def show_hover_panel(get_text_func=None):
        cursor = mplcursors.cursor(
            hover=2,  # Transient
            annotation_kwargs=dict(
                bbox=dict(
                    boxstyle="square,pad=0.5",
                    facecolor="white",
                    edgecolor="#ddd",
                    linewidth=0.5,
                    path_effects=[withSimplePatchShadow(offset=(1.5, -1.5))],
                ),
                linespacing=1.5,
                arrowprops=None,
            ),
            highlight=True,
            highlight_kwargs=dict(linewidth=2),
        )

        if get_text_func:
            cursor.connect(
                event="add",
                func=lambda sel: sel.annotation.set_text(get_text_func(sel.index)),
            )

        return cursor

    def on_add(index):
        item = df.iloc[index]
        parts = [
            f"Suburb: {item.Suburb}",
            f"Area: {item.Area:,.0f}m²",
            f"Sale price: ${item.SalePrice:,.0f}",
        ]

        return "\n".join(parts)

    show_hover_panel(on_add)
    plt.show()


def example_correlation():
    fname = "/home/charles/Current_work/repos/dlc_for_wbfm/wbfm/notebooks/alternative_ideas/tmp_data/df_correlations.h5"
    df = pd.read_hdf(fname)
    ones = np.ones(len(df)) + 0.01*np.random.random(len(df))

    plt.scatter(x=ones, y=df.coefficient, s=100)
    plt.xlabel("Dummy index")
    plt.ylabel("Regression coefficient")
    # plt.title("Size is 1/std of regression")

    def show_hover_panel(get_text_func=None):
        cursor = mplcursors.cursor(
            hover=2,  # Transient
            annotation_kwargs=dict(
                bbox=dict(
                    boxstyle="square,pad=0.5",
                    facecolor="white",
                    edgecolor="#ddd",
                    linewidth=0.5,
                    path_effects=[withSimplePatchShadow(offset=(1.5, -1.5))],
                ),
                linespacing=1.5,
                arrowprops=None,
            ),
            highlight=True,
            highlight_kwargs=dict(linewidth=2),
        )

        if get_text_func:
            cursor.connect(
                event="add",
                func=lambda sel: sel.annotation.set_text(get_text_func(sel.index)),
            )

        return cursor

    def on_add(index):
        item = df.iloc[index]
        parts = [
            f"neuron_name: {item.neuron_name}",
            f"coefficient: {item.coefficient:,.1f}",
            f"coefficient_std: {item.coefficient_std:,.4f}"
        ]

        return "\n".join(parts)

    show_hover_panel(on_add)
    plt.show()


if __name__ == "__main__":
    example_correlation()
