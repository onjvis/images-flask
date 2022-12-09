from enum import StrEnum


class SimilarityMethod(StrEnum):
    EUCLIDEAN = 'euclidean',
    MINKOWSKI = 'minkowski',
    WEIGHTED_EUCLIDEAN = 'weighted_euclidean'
    QUADRATIC_FORM = 'quadratic_form'
    BHATTACHARYYA = 'bhattacharyya'
