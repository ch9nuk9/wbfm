from dataclasses import dataclass
import mplcursors
import numpy as np
from backports.cached_property import cached_property
from matplotlib import pyplot as plt
from wbfm.gui.examples.interactive_scatterplot_examples import get_nice_cursor_opts
from wbfm.utils.visualization.behavior_comparison_plots import NeuronToUnivariateEncoding


@dataclass
class InteractiveScatterplot:

    project_path: str
    encoder_model: NeuronToUnivariateEncoding = None

    def __post_init__(self):
        self.encoder_model = NeuronToUnivariateEncoding(self.project_path)

    @cached_property
    def df_coefficients(self):
        return self.encoder_model.calc_dataset_per_neuron_summary_df('ratio', 'signed_speed')

    @property
    def df_traces(self):
        return self.encoder_model.all_dfs['ratio']

    def make_interactive_scatterplot(self):
        df = self.df_coefficients

        ones = np.ones(len(df)) + 0.01 * np.random.random(len(df))

        fig = plt.figure(dpi=100)
        plt.scatter(x=ones, y=df.coefficient, s=100)
        plt.xlabel("Dummy index")
        plt.ylabel("Regression coefficient")
        # plt.title("Size is 1/std of regression")

        def annotation_text(sel):
            index = sel.index
            item = df.iloc[index]
            parts = [
                f"neuron_name: {item.neuron_name}",
                f"coefficient: {item.coefficient:,.1f}",
                f"coefficient_std: {item.coefficient_std:,.4f}"
            ]
            return "\n".join(parts)

        # First cursor, for hovering
        hover_cursor = mplcursors.cursor(fig, **get_nice_cursor_opts())

        @hover_cursor.connect("add")
        def on_hover(sel):
            sel.annotation.set_text(annotation_text(sel))

        # Second cursor for clicking, not hovering
        click_cursor = mplcursors.cursor(fig, hover=False)

        @click_cursor.connect("add")
        def on_click(sel):
            sel.annotation.set_text(annotation_text(sel))
            index = sel.index
            item = df.iloc[index]
            neuron_name = item.neuron_name
            trace = self.df_traces[neuron_name]
            newfig, newax = plt.subplots()
            newax.plot(trace)
            newax.set_title(neuron_name)
            self.encoder_model.project_data.shade_axis_using_behavior(ax=newax)
            newfig.show()
            # Also set a persistent tooltip with the same text
            print("===========================================")
            print("Clicked neuron with this information:")
            print(item)

        # Finish
        plt.show()


if __name__ == "__main__":
    fname = "/scratch/neurobiology/zimmer/fieseler/wbfm_projects/2022-11-27_spacer_7b_2per_agar/ZIM2165_Gcamp7b_worm1-2022_11_28"

    obj = InteractiveScatterplot(fname)
    obj.make_interactive_scatterplot()
