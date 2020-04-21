import numpy as np


def region_growing(bin_mat, seed):
    new_points = [seed]
    array_for_next = [[0, 1], [1, 0], [0, -1], [-1, 0]]

    mask = np.zeros_like(bin_mat, dtype=np.bool)
    mask[seed[0], seed[1]] = True

    while len(new_points) is not 0:
        stack_point = []
        for candidate_point in new_points:
            for array in array_for_next:
                next = (candidate_point[0] + array[0], candidate_point[1] + array[1])
                print(bin_mat[next[0], next[1]])
                if (bin_mat[next[0], next[1]] == 0) and (mask[next[0], next[1]] != True):
                    mask[next[0], next[1]] = True
                    stack_point.append(next)
        new_points = stack_point

    return mask