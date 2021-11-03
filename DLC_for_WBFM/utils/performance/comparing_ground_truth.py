import numpy as np
from fDNC.src.DNC_predict import filter_matches


def calc_true_positive(gt: dict, test: dict):
    num_tp = 0
    for k, v in gt.items():
        if test.get(k, None) == v:
            num_tp += 1
    return num_tp


def calc_mismatches(gt: dict, test: dict):
    num_mm = 0
    for k, v in test.items():
        if gt.get(k, None) != v:
            num_mm += 1
    return num_mm


def calc_missing_matches(gt: dict, test: dict):
    num_mm = 0
    for k, v in gt.items():
        if k not in test:
            num_mm += 1
    return num_mm


def calc_summary_scores_for_training_data(m_final,
                                          min_confidence=0.0,
                                          max_possible=None):
    """
    Assumes the true matches are trivial, e.g. (1,1)

    max_possible defaults to assuming the maximum match in the first column (template) is all that is possible
    """
    m_final = filter_matches(m_final, min_confidence)
    if max_possible is None:
        max_possible = np.max(m_final[:, 0]).astype(int) + 1
    m0to1_dict = {m[0]: m[1] for m in m_final}

    num_tp = 0
    num_outliers = 0
    for m0, m1 in m0to1_dict.items():
        if m0 > max_possible:
            continue
        if m0 == m1:
            num_tp += 1
        else:
            num_outliers += 1
    num_missing = max_possible - num_tp - num_outliers

    return num_tp, num_outliers, num_missing, max_possible


def get_confidences_of_tp_and_outliers(m_final):
    """
    Assumes the true matches are trivial, e.g. (1,1)

    max_possible defaults to assuming the maximum match in the first column (template) is all that is possible
    """
    m0to1_dict = {m[0]: m[1] for m in m_final}
    m0toconf_dict = {m[0]: m[2] for m in m_final}

    conf_tp = []
    conf_outliers = []
    for m0, m1 in m0to1_dict.items():
        if m0 == m1:
            conf_tp.append(m0toconf_dict[m0])
        else:
            conf_outliers.append(m0toconf_dict[m0])

    return conf_tp, conf_outliers
