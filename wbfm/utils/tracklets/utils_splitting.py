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

    def get_split_points_using_feature_jumps(self, df_working_copy, original_name, jump=1):
        from ruptures.exceptions import BadSegmentationParameters
        _ = self.get_means_to_subtract(df_working_copy)
        tracklet = df_working_copy[original_name]
        signal = self.get_signal_from_tracklet(tracklet)
        if signal is None:
            return []
        try:
            split_list = split_signal(signal, self.penalty, jump=jump)
            # Convert back to original times
            ind_no_nan = tracklet.dropna(axis=0).index
            split_list = [ind_no_nan[i] for i in split_list if i < len(ind_no_nan)]
        except BadSegmentationParameters:
            split_list = []

        return split_list

    def plot_split_points_and_tracklet(self, df, original_name, split_list):
        import ruptures as rpt
        tracklet = df[original_name]
        signal = self.get_signal_from_tracklet(tracklet)
        # Convert to local times
        ind_no_nan = list(tracklet.dropna(axis=0).index)
        split_list = [ind_no_nan.index(i) for i in split_list]
        rpt.display(signal, [], split_list, figsize=(10, 6))
        plt.show()

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


def split_signal(signal, penalty, jump):
    import ruptures as rpt
    # Note: the timing is quadratic (I think) in the jump parameter;
    model = "l2"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=3, jump=jump).fit(signal)
    my_bkps = algo.predict(pen=penalty)
    return my_bkps
