from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from DLC_for_WBFM.utils.pipeline.matches_class import MatchesWithConfidence
from DLC_for_WBFM.utils.projects.utils_neuron_names import int2name


@dataclass
class TrackedNeuron:

    name: str
    initialization_frame: int

    tracklet_matches: MatchesWithConfidence = MatchesWithConfidence()
    tracklet_covering_ind: list = None

    verbose: int = 0

    def __post_init__(self):
        if self.tracklet_covering_ind is None:
            self.tracklet_covering_ind = []

    # For use when assigning matches and iterating over time
    @property
    def next_gap(self):
        return self.tracklet_covering_ind[-1] + 1

    def add_tracklet(self, tracklet_name, confidence, tracklet: pd.DataFrame):
        self.tracklet_matches.add_match([self.name, tracklet_name, confidence])

        tracklet_covering = np.where(tracklet['z'].notnull())[0]
        self.tracklet_covering_ind.extend(tracklet_covering)
        # self.next_gap = tracklet_covering[-1] + 1

        if self.verbose >= 2:
            print(f"Added tracklet {tracklet_name} to neuron {self.name} with next gap currently: {self.next_gap}")


@dataclass
class TrackedWorm:

    neurons: Dict[str, TrackedNeuron] = None

    verbose: int = 0

    @property
    def num_neurons(self):
        return len(self.neurons)

    def get_next_neuron_name(self):
        return int2name(self.num_neurons + 1)

    def __post_init__(self):
        if self.neurons is None:
            self.neurons = {}

    def initialize_new_neuron(self, name=None, initialization_frame=0):
        if name is None:
            name = self.get_next_neuron_name()

        new_neuron = TrackedNeuron(name, initialization_frame, verbose=self.verbose-1)
        self.neurons[name] = new_neuron

        return new_neuron

    def tracks_with_gap_at_or_after_time(self, t):
        return {name: neuron for name, neuron in self.neurons.items() if t > neuron.next_gap}
