import numpy as np
from matplotlib import pyplot as plt

from wbfm.utils.external.custom_errors import AnalysisOutOfOrderError


class TrackletSplitter:
    # Note: I don't want to save the dataframe in this class, because splitting modifies it

    def __init__(self, features=None, split_model='l2',
                 penalty=0.5, verbose=0):
        if features is None:
            features = ['z', 'volume', 'brightness_red']

        self.features = features
        self.split_model = split_model
        self.penalty = penalty
        self.verbose = verbose

        self._means_to_subtract = None

    def get_means_to_subtract(self, df=None):
        if not self._means_to_subtract:
            if df is None:
                raise AnalysisOutOfOrderError("Must call with dataframe first time")
            print("Initializing the means to subtract from the features (probably just z)")
            df_z = df.loc[:, (slice(None), 'z')]
            av = np.array(df_z.mean(axis=1))
            self._means_to_subtract = [av, None, None]

        return self._means_to_subtract

    def get_signal_from_tracklet(self, tracklet):
        return get_signal_from_tracklet(tracklet, self.features, means_to_subtract=self.get_means_to_subtract())


def get_signal_from_tracklet(tracklet, features, means_to_subtract=None):
    if not means_to_subtract:
        means_to_subtract = [None] * len(features)
    signal_list = []
    for f, av in zip(features, means_to_subtract):
        signal = np.array(tracklet[f])
        ind_to_keep = ~np.isnan(signal)
        signal = signal[ind_to_keep]
        if len(signal) == 0:
            return None
        if av is not None:
            signal -= av[ind_to_keep]
        signal /= np.max(signal)
        signal_list.append(signal)
    signal = np.vstack(signal_list).T
    return signal
