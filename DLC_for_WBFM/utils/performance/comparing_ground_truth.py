
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
