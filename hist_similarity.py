import numpy as np


def euclidean(hist1, hist2):
    return minkowski(hist1, hist2, p=2)


def weighted_euclidean(hist1, hist2, weights):
    return weighted_minkowski(hist1, hist2, p=2, weights=weights)


def weighted_minkowski(hist1, hist2, p, weights):
    if p <= 0:
        return
    return np.power(np.sum(weights * np.power(np.abs(hist1 - hist2), p)), 1 / p)


def minkowski(hist1, hist2, p):
    return weighted_minkowski(hist1, hist2, p, np.ones(len(hist1)))


def bhattacharyya(hist1, hist2):
    pdf1 = hist1 / np.sum(hist1)
    pdf2 = hist2 / np.sum(hist2)
    bc = np.sum(np.sqrt(pdf1 * pdf2))
    return -np.log(bc)


def quadratic_form(hist1, hist2, m):
    return np.sqrt(((hist1 - hist2) @ m) @ np.transpose(hist1 - hist2))
