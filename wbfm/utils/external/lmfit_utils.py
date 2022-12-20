import numpy as np
from lmfit.models import ExponentialModel
from matplotlib import pyplot as plt


def fit_multi_exponential_model(x, y, to_plot=True, num_exponentials=2, verbose=0):
    mymodel = ExponentialModel(prefix='e0_')
    for i in range(num_exponentials - 1):
        mymodel = mymodel + ExponentialModel(prefix=f'e{i+1}_')

    # Guess isn't super important, but the decays should be different scales
    p_dict = {}
    for i in range(num_exponentials):
        p_dict.update({f'e{i}_amplitude': 10**(i+1), f'e{i}_decay': 10**(-i+1)})
        # params = mymodel.make_params(e1_amplitude=10, e1_decay=10,
        #                              e2_amplitude=10, e2_decay=0.50)
    params = mymodel.make_params(**p_dict)

    result = mymodel.fit(y, params, x=x)
    if verbose >= 1:
        print(result.fit_report())

    if to_plot:
        plt.figure(dpi=100)
        components = result.eval_components(x=x)

        plt.scatter(x, y, color='gray', label='data')
        plt.plot(x, result.best_fit, label='best fit')
        if num_exponentials > 1:
            for i in range(num_exponentials):
                plt.plot(x, components[f'e{i}_'], '--', label=f'exp{i}')
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
