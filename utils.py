import numpy as np
import cv2
import math

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
                    #cv2.imshow("region", 255 * mask.astype(np.uint8))
                    #cv2.waitKey(100)
        new_points = stack_point

    return mask


def get_8_neighbor_with_mask(mask, center_rc):
    bin_mat = 255 * mask.astype(np.uint8)
    test = 255 * bin_mat[:, :, np.newaxis].astype(np.uint8)
    test = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)

    if bin_mat[center_rc[0], center_rc[1]] == 0:
        raise
    bin_shape = bin_mat.shape
    edges_rc = []

    # check vertical edge
    for j in range(1, bin_shape[0], 1):
        v_derivative = bin_mat[j, center_rc[1]] - bin_mat[j - 1, center_rc[1]]
        if v_derivative != 0:
            edges_rc.append(np.array([j, center_rc[1]]))
            cv2.circle(test, (center_rc[1], j), 3, (0, 0, 200), -1)

    # check horizonal edge
    for i in range(1, bin_shape[1], 1):
        h_derivative = bin_mat[center_rc[0], i] - bin_mat[center_rc[0], i - 1]
        if h_derivative != 0:
            edges_rc.append(np.array([center_rc[0], i]))
            cv2.circle(test, (i, center_rc[0]), 3, (200, 0, 0), -1)

    cv2.imshow("test", test)
    cv2.waitKey(1)
    return edges_rc


def get_8neighbors_with_radius(src_radius, center_rc):
    # get_vertical points
    TPS_src_xy = []
    init_relative_vector = np.array([0, src_radius])
    center_xy_array = np.array([center_rc[0], center_rc[1]])
    step_theta = 45
    theta = 0
    for i in range(8):
        rotate_mat = np.array([[math.cos(math.radians(theta)), -math.sin(math.radians(theta))],
                               [math.sin(math.radians(theta)), math.cos(math.radians(theta))]])
        src_wall_point = center_xy_array + np.dot(rotate_mat, np.transpose(init_relative_vector))
        TPS_src_xy.append(np.array([src_wall_point[0], src_wall_point[1]], dtype=np.float32))
        theta += step_theta
    TPS_src_xy = np.array(TPS_src_xy, dtype=np.float32)
    return TPS_src_xy


def thin_plate_spline(sshape, tshape, src_mat, n_matches=8):
    tps = cv2.createThinPlateSplineShapeTransformer()

    sshape = np.array(sshape, dtype=np.float32)
    tshape = np.array(tshape, dtype=np.float32)
    sshape = sshape.reshape((1, -1, 2))
    tshape = tshape.reshape((1, -1, 2))

    matches = list()
    for i in range(n_matches):
        matches.append(cv2.DMatch(i, i, 0))

    tps.estimateTransformation(sshape, tshape, matches)

    ret, tshape_ = tps.applyTransformation(sshape)

    tps.estimateTransformation(tshape, sshape, matches)

    distorted_mat = tps.warpImage(src_mat)
    return distorted_mat


def convert_rc_to_cv2point(rc):
    return [rc[1], rc[0]]
