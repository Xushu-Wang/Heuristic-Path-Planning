import numpy as np


def euclidean_heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def chebyshev_heuristic(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def mixed_heuristic(a, b, epsilon):
    return chebyshev_heuristic(a, b) + epsilon * euclidean_heuristic(a, b)


def path_distance(path):
    dist = 0

    for i in range(len(path) - 1):
        dist += euclidean_heuristic(path[i], path[i+1])

    return dist