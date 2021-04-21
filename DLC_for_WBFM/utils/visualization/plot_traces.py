
import matplotlib.pyplot as plt
from ipywidgets import interact
from matplotlib.ticker import NullFormatter
from matplotlib import transforms
from DLC_for_WBFM.utils.postprocessing.base_DLC_utils import xy_from_dlc_dat

##
## Functions for use with data from 'extract_all_traces'
##

def visualize_traces_with_reference(all_traces,
                                    reference_ind, reference_name,
                                    all_names=None,
                                    to_normalize=True,
                                    to_save=False):
    """
    Plot all neurons on a reference, given by reference_ind
    """
    all_names = check_default_names(all_names, len(all_traces))

    reference_trace = all_traces[reference_ind]

    for i, t_dict in enumerate(all_traces):
        if i == reference_ind:
            continue
        # Plot looped trace and reference
        ax1, ax2 = visualize_mcherry_and_gcamp(reference_trace, reference_name,
                                               make_new_title=False,
                                               to_normalize=to_normalize)
        visualize_mcherry_and_gcamp(t_dict, all_names[i],
                                    make_new_fig=False,
                                    make_new_title=False,
                                    ax1=ax1, ax2=ax2,
                                    to_normalize=to_normalize)
        if to_save:
            plt.savefig(f'traces_{all_names[i]}_ref_{reference_name}')


def visualize_mcherry_and_gcamp(t_dict,
                                name,
                                which_neuron,
                                make_new_fig=True,
                                make_new_title=True,
                                ax1 = None,
                                ax2 = None,
                                to_normalize=False,
                                preprocess_func=None):
    """
    NOTE: preprocess_func is nonfunctional
    """
    if make_new_fig:
        plt.figure(figsize=(35,5))#, fontsize=12)

    if make_new_fig:
        ax1 = plt.subplot(121)
    dat = get_tracking_channel(t_dict)
    if to_normalize:
        dat = dat / np.max(np.array(dat))
    if make_new_title:
        ax1.plot(dat)
        plt.title(f'Red channel for neuron {name}')
    else:
        ax1.plot(dat, label=f'Red channel for neuron {name}')
        ax1.legend()

    if make_new_fig:
        ax2 = plt.subplot(122)
    dat = get_measurement_channel(t_dict)
    if to_normalize:
        dat = dat / np.max(np.array(dat))
    if make_new_title:
        ax2.plot(dat)
        plt.title(f'Green channel for neuron {name}')
    else:
        ax2.plot(dat, label=f'Green channel for neuron {name}')
        ax2.legend()

    set_big_font()

    return ax1, ax2


def visualize_ratio(t_dict,
                    name,
                    which_neuron,
                    tspan=None,
                    background=[0,0],
                    ylim=[0,1],
                    preprocess_func=None):
    """
    Divides the green by the red channel to produce a normalized time series
        Optionally subtracts a background value
    """
    plt.figure(figsize=(35,5))


    red = get_tracking_channel(t_dict)
    green = get_measurement_channel(t_dict)

    if preprocess_func is not None:
        red = preprocess_func(red, which_neuron)
        green = preprocess_func(green, which_neuron)

    dat = (green-background[0]) / (red-background[1])
    if tspan is None:
        plt.plot(dat)
    else:
        plt.plot(tspan, dat)
    plt.xlabel('Seconds')
    plt.ylim(ylim)
    plt.title(f"Ratiometric for neuron {name}")


def visualize_all_traces(all_traces,
                         all_names=None,
                         plot_subfunction=visualize_mcherry_and_gcamp,
                         preprocess_func=None,
                         to_save=False):
    """
    Plots all neurons in a struct using a subfunction with the following API:
        plot_subfunction(t_dict,
                         which_neuron=i,
                         name=all_names[i],
                         preprocess_func=preprocess_func)
    """
    all_names = check_default_names(all_names, len(all_traces))

    for i, t_dict in enumerate(all_traces):
        plot_subfunction(t_dict,
                         which_neuron=i,
                         name=all_names[i],
                         preprocess_func=preprocess_func)
        if to_save:
            plt.savefig(f'traces_{all_names[i]}')


##
## Generally plotting
##


def plot2d_with_max(dat, t, max_ind, max_vals, vmin=100, vmax=400):
    plt.imshow(dat[:,:,0,t], vmin=vmin, vmax=vmax)
    plt.colorbar()
    x, y = max_ind[t,1], max_ind[t,0]
    if z == max_ind[t,2]:
        plt.scatter(x, y, marker='x', c='r')
    plt.title(f"Max for t={t} is {max_vals[t]} xy={x},{y}")

def plot3d_with_max(dat, z, t, max_ind, vmin=100, vmax=400):
    plt.imshow(dat[:,:,z,t], vmin=vmin, vmax=vmax)
    plt.colorbar()
    x, y = max_ind[t,1], max_ind[t,0]
    if z == max_ind[t,2]:
        plt.scatter(x, y, marker='x', c='r')
    plt.title(f"Max for t={t} is on z={max_ind[t,2]}, xy={x},{y}")


def plot3d_with_max_and_hist(dat, z, t, max_ind):
    # From: https://matplotlib.org/2.0.2/examples/pylab_examples/scatter_hist.html
    rot = transforms.Affine2D().rotate_deg(90)
    nullfmt = NullFormatter()         # no labels

    plt.figure(1, figsize=(8, 8))


    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    axIm = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Actually display
    frame = dat[:,:,z,t]
    axIm.imshow(frame, vmin=0, vmax=400)
    x, y = max_ind[t,1], max_ind[t,0]
#     if z == max_ind[t,2]:
#         plt.scatter(x, y, marker='x', c='r')
#     plt.title(f"Max for t={t} is on z={max_ind[t,2]}, xy={x},{y}")

    axHistx.plot(np.max(frame, axis=0))

#     base = plt.gca().transData
    axHisty.plot(np.flip(np.max(frame, axis=1)), range(frame.shape[0]))#, transform=base+rot)



##
## Helper functions
##

def check_default_names(all_names, num_neurons):
    if all_names is None:
        all_names = [str(i) for i in range(num_neurons)]
    return all_names


def set_big_font(size=22):
    # From: https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    import matplotlib

    font = {'weight' : 'bold',
            'size'   : size}
    matplotlib.rc('font', **font)


def get_tracking_channel(t_dict):
    try:
        dat = t_dict['mcherry']
    except:
        dat = t_dict['red']
    return dat

def get_measurement_channel(t_dict):
    try:
        dat = t_dict['gcamp']
    except:
        dat = t_dict['green']
    return dat

def nan_tracking_failures(config,
                          dat,
                          which_neuron,
                          threshold=0.9):
    c = load_config(config)

    _, this_prob = xy_from_dlc_dat(c.tracking.annotation_fname,
                                   which_neuron,
                                   c.preprocessing.num_frames)

    bad_vals = np.array(this_prob) < threshold
    dat[bad_vals] = np.nan
    # print(np.count_nonzero(this_prob < threshold))

    return dat
