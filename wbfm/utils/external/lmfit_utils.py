import numpy as np
from lmfit.models import ExponentialModel, ConstantModel
from matplotlib import pyplot as plt


def fit_multi_exponential_model(x, y, to_plot=True, num_exponentials=2,
                                cumulative=False,
                                verbose=0):
    """
    Fit a multi-exponential model to the data (1 is allowed).

    Optionally fit a cumulative model, which should be done if there are empty bins.
        In this case the functional form is still exponential, but it is a constant minus the exponentials.

    Parameters
    ----------
    x
    y
    to_plot
    num_exponentials
    cumulative
    verbose

    Returns
    -------

    """
    mymodel = ExponentialModel(prefix='e0_')
    for i in range(num_exponentials - 1):
        mymodel = mymodel + ExponentialModel(prefix=f'e{i+1}_')

    if cumulative:
        mymodel = mymodel + ConstantModel(prefix='const_')

    # Guess isn't super important, but the decays should be different scales
    p_dict = {}
    for i in range(num_exponentials):
        if not cumulative:
            p_dict.update({f'e{i}_amplitude': 10**(i+1), f'e{i}_decay': 10**(-i+1), 'min': 0})
        else:
            p_dict.update({f'e{i}_amplitude': -10 ** (i + 1), f'e{i}_decay': 10 ** (-i + 1)})
        # params = mymodel.make_params(e1_amplitude=10, e1_decay=10,
        #                              e2_amplitude=10, e2_decay=0.50)
    params = mymodel.make_params(**p_dict)
    if cumulative:
        params['const_c'].set(value=num_exponentials, vary=False)

    result = mymodel.fit(y, params, x=x)
    if verbose >= 1:
        print(result.fit_report())

    if to_plot:
        plt.figure(dpi=100)
        components = result.eval_components(x=x)

        plt.scatter(x, y, color='gray', label='data')
        plt.plot(x, result.best_fit, label='best fit')
        if num_exponentials > 1:
            num_components = num_exponentials
            for i in range(num_exponentials):
                plt.plot(x, components[f'e{i}_'], '--', label=f'exp{i}')
            if cumulative:
                plt.plot(x, np.ones(len(x))*components[f'const_'], '--', label=f'const')
        plt.legend()

    return result


def fit_single_exponential_model(x, y, to_plot=True):
    mymodel = ExponentialModel(prefix='e1_')

    # params = mymodel.make_params(e1_amplitude=20, e1_decay=50)
    params = mymodel.guess(y, x=x)

    result = mymodel.fit(y, params, x=x)
    print(result.fit_report())

    if to_plot:
        plt.figure(dpi=100)
        plt.scatter(x, y, color='gray')
        plt.plot(x, result.best_fit, '--')

    return result
