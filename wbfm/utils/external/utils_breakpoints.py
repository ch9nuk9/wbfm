import numpy as np
from matplotlib import pyplot as plt


def plot_with_offset_x(x_offset, pw_fit, **kwargs):
    # See the plot_fit function in the piecewise_regression package

    # Get the final results from the fitted model variables
    # Params are in terms of [intercept, alpha, betas, gammas]
    final_params = pw_fit.best_muggeo.best_fit.raw_params
    breakpoints = pw_fit.best_muggeo.best_fit.next_breakpoints

    # Extract what we need from params etc
    intercept_hat = final_params[0]
    alpha_hat = final_params[1]
    beta_hats = final_params[2:2 + len(breakpoints)]

    xx_plot = np.linspace(min(pw_fit.xx), max(pw_fit.xx), 100)

    # Build the fit plot segment by segment. Betas are defined as
    # difference in gradient from previous section
    yy_plot = intercept_hat + alpha_hat * xx_plot
    for bp_count in range(len(breakpoints)):
        yy_plot += beta_hats[bp_count] * \
                   np.maximum(xx_plot - breakpoints[bp_count], 0)

    plt.plot(xx_plot + x_offset, yy_plot, **kwargs)