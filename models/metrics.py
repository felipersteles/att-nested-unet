import numpy as np

def dice_coefficient(pred, target):
    intersection = np.sum(pred * target)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(target))

def calc_jaccard_index(pred, target):
    intersection = np.sum(pred * target)
    union = np.sum(pred) + np.sum(target) - intersection

    if intersection == 0:
        return 0.0

    return intersection / union

def precision_score(pred, target):
    true_positives = np.sum((target == 1) & (pred == 1))
    false_positives = np.sum((target == 0) & (pred == 1))
    precision = true_positives / (true_positives + false_positives + 1e-6)