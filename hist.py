import cv2
import numpy as np
import os
import re
from matplotlib import pyplot as plt

import hist_similarity
from model.color_model import ColorModel
from model.similarity_method import SimilarityMethod

ROOT_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.join(ROOT_DIR, 'assets', 'hist')
MODELS_DIR = os.path.join(ROOT_DIR, 'assets', 'comparison-models')
BASE_EXT = '.npy'


def compute_histogram(color_model, bin_count, image_path):
    image = cv2.imread(image_path)

    if color_model == ColorModel.RGB:
        return compute_histogram_rgb(image, hist_size=bin_count)
    elif color_model == ColorModel.HSV:
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return compute_histogram_hsv(image_hsv, image_gray, hist_size=bin_count)
    else:
        print('not supported')


def compute_histogram_rgb(image, hist_size):
    return np.array([cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[hist_size], ranges=[0, 256],
                                  accumulate=False),  # blue
                     cv2.calcHist(images=[image], channels=[1], mask=None, histSize=[hist_size], ranges=[0, 256],
                                  accumulate=False),  # green
                     cv2.calcHist(images=[image], channels=[2], mask=None, histSize=[hist_size], ranges=[0, 256],
                                  accumulate=False)])  # red


def compute_histogram_hsv(image_hsv, image_gray, hist_size):
    return np.array([cv2.calcHist(images=[image_hsv], channels=[0], mask=None, histSize=[hist_size], ranges=[0, 180],
                                  accumulate=False),  # hue
                     cv2.calcHist(images=[image_hsv], channels=[1], mask=None, histSize=[hist_size], ranges=[0, 256],
                                  accumulate=False),  # saturation
                     cv2.calcHist(images=[image_gray], channels=[0], mask=None, histSize=[hist_size], ranges=[0, 256],
                                  accumulate=False)])  # value


def normalize_histogram(hist):
    flattened_hist = hist.flatten()
    normalized_hist = (flattened_hist - flattened_hist.min()) / (flattened_hist.max() - flattened_hist.min())

    return normalized_hist


def save_histogram(color_model, bin_count, filename, hist):
    histogram_filepath = get_histogram_filepath(color_model, bin_count, filename)

    np.save(histogram_filepath, hist)

    return f'{histogram_filepath}{BASE_EXT}'


def load_histogram(filename):
    return np.load(filename)


def get_histogram_filepath(color_model, bin_count, filename):
    filename = re.sub(r'\.[^.]*$', '', filename)

    histogram_folder_path = os.path.join(BASE_DIR, color_model.lower(), str(bin_count))
    if not os.path.exists(histogram_folder_path):
        os.makedirs(histogram_folder_path)

    return os.path.join(histogram_folder_path, filename)


def compare(hist1, color_model=ColorModel.RGB, method=SimilarityMethod.EUCLIDEAN, bin_count=256, params=None):
    results = {}

    hist_folder_path = os.path.join(BASE_DIR, color_model.lower(), str(bin_count))
    for hist_file in os.listdir(hist_folder_path):
        hist2 = load_histogram(os.path.join(hist_folder_path, hist_file))
        hist_file = re.sub(r'\.[^.]*$', '', hist_file)

        if method == SimilarityMethod.EUCLIDEAN:
            results[hist_file] = hist_similarity.euclidean(hist1, hist2)
        elif method == SimilarityMethod.MINKOWSKI and params != {} and params['minkowski'] != 0:
            results[hist_file] = hist_similarity.minkowski(hist1, hist2, params['minkowski'])
        elif method == SimilarityMethod.BHATTACHARYYA:
            results[hist_file] = hist_similarity.bhattacharyya(hist1, hist2)
        elif method == SimilarityMethod.WEIGHTED_EUCLIDEAN and params != {} and params['weighted_model'] != '':
            model = params['weighted_model']
            model_filepath = os.path.join(MODELS_DIR,
                                          f'{SimilarityMethod.WEIGHTED_EUCLIDEAN}-{model}-{str(bin_count)}{BASE_EXT}')
            weights = np.load(model_filepath)
            results[hist_file] = hist_similarity.weighted_euclidean(hist1, hist2, weights=weights)
        elif method == SimilarityMethod.QUADRATIC_FORM and params != {} and params['quadratic_form_model'] != '':
            model = params['quadratic_form_model']
            model_filepath = os.path.join(MODELS_DIR,
                                          f'{SimilarityMethod.QUADRATIC_FORM}-{model}-{str(bin_count)}{BASE_EXT}')
            correlation_matrix = np.load(model_filepath)
            results[hist_file] = hist_similarity.quadratic_form(hist1, hist2, m=correlation_matrix)

    return results


def sort(results):
    sorted_results = sorted(results.items(), key=lambda e: e[1])
    print(sorted_results)
    return sorted_results


def plot_histogram_1d(image):
    # define colors to plot the histograms
    colors = ('b', 'g', 'r')
    # compute and plot the image histograms
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [16], [0, 256])
        plt.plot(hist, color=color)
    plt.title('Image Histogram GFG')
    plt.show()
