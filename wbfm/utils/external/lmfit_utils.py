import numpy as np
from lmfit.models import ExponentialModel
from matplotlib import pyplot as plt


def fit_double_exponential_model(x, y, to_plot=True):
    mymodel = ExponentialModel(prefix='e1_') + ExponentialModel(prefix='e2_')

    # Guess isn't super important, but the decays should be different scales
    params = mymodel.make_params(e1_amplitude=10, e1_decay=10,
                                 e2_amplitude=25, e2_decay=0.50)

    result = mymodel.fit(y, params, x=x)
    print(result.fit_report())

    if to_plot:
        plt.figure(dpi=100)
        components = result.eval_components(x=x)

        plt.scatter(x, y, color='gray', label='data')
        plt.plot(x, result.best_fit, label='combined fit')
        plt.plot(x, components['e1_'], '--', label='exp1')
        plt.plot(x, components['e2_'], '--', label='exp2')
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
