import ipywidgets as widgets
from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np
from DLC_for_WBFM.config.class_configuration import *
from DLC_for_WBFM.utils.postprocessing.config_cropping_utils import _get_crop_from_ometiff_virtual


##
## External helpers
##

def update_line(t, z):
    plt.vlines(t, 0,2)
    # line.set_data([t,t], [0,2])
    print("Updating line...")
    plt.show()

def external_update(obj, t, z):
    print("Updating all")
    for update, objs in zip(obj.panel_updaters, obj.panel_objs):
        update(t, z, objs)


class InteractiveConfig():
    """
    Interactive plotting of all information in a config file
    """

    def __init__(self, config, which_neurons=None):
        """Pre-load video data"""

        # Variables for current panels
        self.reset_panels()

        # Preloading
        c = load_config(config)
        self.c = c
        self.num_frames = c.preprocessing.num_frames
        if which_neurons is None:
            self.which_neurons = list(c.traces.which_neurons)
        else:
            self.which_neurons = which_neurons
        self.current_neuron = self.which_neurons[0]
        self.green_data = []
        self.red_data = []

        print(f"Loading video data for {len(self.which_neurons)} neurons...")
        for which_neuron in self.which_neurons:
            g = _get_crop_from_ometiff_virtual(config,
                                                        which_neuron,
                                                        self.num_frames,
                                                        use_red_channel=False)
            r = _get_crop_from_ometiff_virtual(config,
                                                      which_neuron,
                                                      self.num_frames)
            self.green_data.append(g)
            self.red_data.append(r)
        print("Finished loading video data")

        # Build sliders; same for all plotting
        t_slider = widgets.IntSlider(value=0, min=0, max=self.num_frames)
        z_slider = widgets.IntSlider(value=0, min=0, max=c.traces.crop_sz[-1])
        self.all_sliders = [t_slider, z_slider]


    def reset_panels(self):
        self.panel_updaters = []
        self.panel_initializers = []
        self.panel_objs = []

    def update_panels(self, t, z):
        print("Updating all")
        for update, objs in zip(self.panel_updaters, self.panel_objs):
            update(t, z, objs)


    def get_trace_data(self, which_field='red'):
        # Read traces
        with open(self.c.traces.traces_fname, 'rb') as f:
            trace_dat = pickle.load(f)
        trace_dat = np.array(trace_dat[self.current_neuron][which_field])
        return trace_dat

    def add_red_panel(self):
        self.panel_initializers.append(self.initialize_red_panel)
        print(f"Total number of panels: {len(self.panel_initializers)}")

    def initialize_red_panel(self):
        # self.ax.plot(self.get_trace_data())
        # t = 0
        # line = self.ax.plot([t,t], [0,2], 'r')[0]
        # self.panel_objs.append(line)
        # self.ax.plot([t+1,t+1], [0,2], 'r')
        # line = self.ax.vlines(t,0,2, colors='r')
        # plt.title('Trace')
        # self.ax.set_ylim([0,2])

        self.panel_updaters.append(update_line)
        # self.panel_updaters.append(self.update_red_panel)
        # local_update = lambda t, z : self.update_red_panel(line, t, z)
        # self.panel_updaters.append(local_update)

    def update_red_panel(self, t, z, line):

        # ax.vlines(t,0,2, colors='r')
        # seg_old = line.get_segments()
        # ymin = seg_old[0][0, 1]
        # ymax = seg_old[0][1, 1]
        # seg_new = [np.array([[t, ymin], [t, ymax]])]
        # print(seg_new)
        # line.set_segments(seg_new)
        plt.vlines(t, 0,2)
        # line.set_data([t,t], [0,2])
        print("Updating line...")
        plt.show()


    def build_widget(self):

        # Build empty widget
        # output = widgets.Output()
        # with output:
        #     fig, ax = plt.subplots(constrained_layout=True, figsize=(25,15))
        # fig.canvas.toolbar_position = 'bottom'
        # self.fig = fig
        # self.ax = ax

        # Initialize panels
        for init in self.panel_initializers:
            init()

        # Add data and sliders
        args = {'t':(0,self.num_frames-1), 'z':(0,self.c.traces.crop_sz[-1]-1)}
        f = lambda t, z: [f(t, z) for f in self.panel_updaters]
        return interact(f, **args)
        # f = lambda t, z: external_update(self, t, z)
        # return interact(external_update, **args, obj=fixed(self))
        # return interact(self.update_panels, **args)
        # w = interact(self.update_panels,
        #                 t=self.all_sliders[0],
        #                 z=self.all_sliders[1])
        # display(w)
        # w = interact_manual(self.update_panels, i=self.all_sliders)
        # for make_panel in self.panel_updaters:
        #     for slider in self.all_s:
        #         pass
        # controls = widgets.VBox()

        #
        # return w
