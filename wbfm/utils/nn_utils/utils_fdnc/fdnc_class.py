import logging
from dataclasses import dataclass

import numpy as np
from wbfm.utils.external.custom_errors import NoNeuronsError
from wbfm.utils.nn_utils.fdnc_predict import load_fdnc_template, load_fdnc_options
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.physical_units import PhysicalUnitConversion


@dataclass
class fDNCTracker:
    prediction_options: dict
    match_confidence_threshold: float
    physical_unit_conversion: PhysicalUnitConversion

    template: np.ndarray = None

    _already_warned = False  # To avoid spamming installation warnings

    @staticmethod
    def load_from_project_data(project_data: ProjectData,
                               initialize_template=True, t=None):
        # Project settings
        cfg = project_data.project_config
        tracks_cfg = cfg.get_tracking_config()
        prediction_options = load_fdnc_options()
        fdnc_updates = tracks_cfg.config['leifer_params']['core_options']
        prediction_options.update(fdnc_updates)
        match_confidence_threshold = tracks_cfg.config['leifer_params']['match_confidence_threshold']
        physical_unit_conversion = PhysicalUnitConversion.load_from_config(cfg)

        # Actual object
        obj = fDNCTracker(prediction_options, match_confidence_threshold, physical_unit_conversion)

        # Further initialization
        if initialize_template:
            if t is None:
                t = 0
            obj.initialize_template_from_volume(project_data, t)

        return obj

    def initialize_template_from_volume(self, project_data, t=0):

        custom_template = project_data.get_centroids_as_numpy(t)
        custom_template = self.physical_unit_conversion.zimmer2leifer(custom_template)
        template, template_label = load_fdnc_template(custom_template)

        self.template = template

    def initialize_leifer_template(self):
        template, template_label = load_fdnc_template()
        self.template = template[:, :3]

    def get_pts(self, project_data, i):
        these_pts = project_data.get_centroids_as_numpy(i)
        if len(these_pts) == 0:
            raise NoNeuronsError
        return self.physical_unit_conversion.zimmer2leifer(these_pts)

    def predict_matches_from_points(self, pts):
        try:
            from fDNC.src.DNC_predict import predict_matches, filter_matches
            matches, _ = predict_matches(test_pos=pts, template_pos=self.template, **self.prediction_options)
            matches = filter_matches(matches, self.match_confidence_threshold)
        except NoNeuronsError:
            matches = []
        except ImportError:
            logging.warning("fDNC is not installed. Skipping prediction using this method")
            self._already_warned = True

        return matches

    def predict_matches_from_time(self, project_data, t):
        pts = self.get_pts(project_data, t)
        return self.predict_matches_from_points(pts)

    def visualize_matches_open3d(self, pts, matches=None):
        if matches is None:
            matches = self.predict_matches_from_points(pts)
        from wbfm.utils.visualization.visualization_tracks import visualize_tracks
        visualize_tracks(self.template, pts, matches=matches)

        return matches
