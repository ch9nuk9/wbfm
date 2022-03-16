import numpy as np
import ruptures as rpt


class TrackletSplitter:

    def get_means_to_subtract(self, df):
        if not self._means_to_subtract:
            print("Initializing the means to subtract from the features (probably just z)")
            num_frames = df.shape[0]
            av = np.zeros(num_frames)
            for t in range(num_frames):
                av[t] = df.loc[0, (slice(None), 'z')]
            self._means_to_subtract = [av, None, None]

        return self._means_to_subtract

    def __init__(self, features=None, split_model='l2', to_subtract_mean=True,
                 penalty=0.5, verbose=0):
        if features is None:
            features = ['z', 'volume', 'brightness_red']

        self.features = features
        self.split_model = split_model
        self.to_subtract_mean = to_subtract_mean
        self.penalty = penalty
        self.verbose = verbose

        self._means_to_subtract = None

    def get_split_points_using_feature_jumps(self, df_working_copy, original_name):
        means_to_subtract = self.get_means_to_subtract(df_working_copy)
        tracklet = df_working_copy[original_name]
        signal = get_signal_from_tracklet(tracklet, self.features, means_to_subtract=means_to_subtract)
        split_list = split_signal(signal, self.penalty)

        return split_list


def get_signal_from_tracklet(tracklet, features, means_to_subtract=None):
    if not means_to_subtract:
        means_to_subtract = [None] * len(features)
    signal_list = []
    for f, av in zip(features, means_to_subtract):
        signal = np.array(tracklet[f])
        signal = signal[~np.isnan(signal)]
        if av:
            signal -= av
        signal /= np.max(signal)
        signal_list.append(signal)
    signal = np.vstack(signal_list).T
    return signal


def split_signal(signal, penalty=0.5):
    model = "l2"  # "l2", "rbf"
    algo = rpt.Pelt(model=model, min_size=3, jump=5).fit(signal)
    my_bkps = algo.predict(pen=penalty)
    return my_bkps
