import cv2
import numpy as np
import sys

left_points = [[147, 64], [436, 74], [441, 296], [254, 171], [353, 237], [149, 137], [150, 210], [379, 104], [413, 267],
               [322, 272], [288, 205], [219, 136], [186, 173], [319, 136], [381, 169], [222, 243], [224, 313],
               [285, 102], [439, 200], [348, 70]]

right_points = [[195, 56], [488, 116], [440, 361], [271, 174], [354, 267], [190, 121], [185, 184], [409, 137],
                [415, 319], [319, 287], [298, 215], [246, 135], [213, 161], [337, 155], [397, 208], [235, 230],
                [230, 290], [309, 114], [460, 261], [381, 92]]


# read keypoints from files
####################################
def read_keypoints():
    left_matrix = [[], [], []]
    right_matrix = [[], [], []]

    for i in range(len(left_points)):
        left_matrix[0].append(left_points[i][0])
        left_matrix[1].append(left_points[i][1])
        left_matrix[2].append(1)

        right_matrix[0].append(right_points[i][0])
        right_matrix[1].append(right_points[i][1])
        right_matrix[2].append(1)

    return left_matrix, right_matrix


# normalize left and right matrices
###################################
def normalization(left_matrix, right_matrix):
    left_x_mean = sum(left_matrix[0]) / len(left_matrix[0])
    left_y_mean = sum(left_matrix[1]) / len(left_matrix[1])

    right_x_mean = sum(right_matrix[0]) / len(right_matrix[0])
    right_y_mean = sum(right_matrix[1]) / len(right_matrix[1])

    left_d = 0
    right_d = 0

    for i in range(len(left_matrix[0])):
        left_d += np.sqrt(((left_matrix[0][i] - left_x_mean) ** 2)
                          + ((left_matrix[1][i] - left_y_mean) ** 2))
    left_d = left_d / len(left_matrix[0])

    for i in range(len(right_matrix[0])):
        right_d += np.sqrt(((right_matrix[0][i] - right_x_mean) ** 2)
                           + ((right_matrix[1][i] - right_y_mean) ** 2))
    right_d = right_d / len(right_matrix[0])

    left_T = np.array([
        [np.sqrt(2) / left_d, 0, -(np.sqrt(2) / left_d) * left_x_mean],
        [0, np.sqrt(2) / left_d, -(np.sqrt(2) / left_d) * left_y_mean],
        [0, 0, 1]
    ])

    right_T = np.array([
        [np.sqrt(2) / right_d, 0, -(np.sqrt(2) / right_d) * right_x_mean],
        [0, np.sqrt(2) / right_d, -(np.sqrt(2) / right_d) * right_y_mean],
        [0, 0, 1]
    ])

    left_matrix_normalized = np.matmul(left_T, left_matrix)
    right_matrix_normalized = np.matmul(right_T, right_matrix)

    return left_matrix_normalized, right_matrix_normalized, left_T, right_T


# formulate A to solve Af = 0
###################################
def compute_A(left_matrix_normalized, right_matrix_normalized):
    A = []
    for i in range(len(left_matrix_normalized[0])):
        A.append([
            right_matrix_normalized[0][i] * left_matrix_normalized[0][i],
            right_matrix_normalized[0][i] * left_matrix_normalized[1][i],
            right_matrix_normalized[0][i],
            right_matrix_normalized[1][i] * left_matrix_normalized[0][i],
            right_matrix_normalized[1][i] * left_matrix_normalized[1][i],
            right_matrix_normalized[1][i],
            left_matrix_normalized[0][i],
            left_matrix_normalized[1][i],
            1
        ])

    return A


###################################
def main():
    if len(sys.argv) < 8:
        print('Usage: epipolarLine.py left right cb_cols cb_rows point_x point_y output')
        sys.exit(1)

    left_filename = sys.argv[1]
    right_filename = sys.argv[2]
    cb_cols = int(sys.argv[3])
    cb_rows = int(sys.argv[4])
    point_x = float(sys.argv[5])
    point_y = float(sys.argv[6])
    output_filename = sys.argv[7]

    # Store the keypoints of the manually-selected corresponding points in homogeneous form
    left_matrix, right_matrix = read_keypoints()

    # Normalize the matrices
    left_matrix_normalized, right_matrix_normalized, left_T, right_T = normalization(left_matrix, right_matrix)

    # Formulate the A matrix to solve the equation Af = 0
    A = compute_A(left_matrix_normalized, right_matrix_normalized)

    # Compute the SVD of A to obtain the least squares solution, and then obtain the normalized fundamental matrix
    U_temp, D_temp, V_t_temp = np.linalg.svd(A)
    V_temp = np.transpose(V_t_temp)
    f = []
    for i in range(len(V_temp)):
        f.append(V_temp[i][-1])
    F_temp = np.reshape(f, (3, 3))

    # Enforce the singularity constraint
    U_final, D_new, V_t_final = np.linalg.svd(F_temp)
    D_new[2] = 0
    D_final = np.diag(D_new)
    F_new = np.matmul(np.matmul(U_final, D_final), V_t_final)

    # Denormalize the fundamental matrix
    F = np.matmul(np.matmul(np.transpose(right_T), F_new), left_T)

    # Convert the given test point to homogeneous coordinates
    test_point = [[point_x], [point_y], [1]]
    l_right = np.matmul(F, test_point)

    # Final output with the corresponding epipolar line of the given test point
    x = np.array([0, 640])
    y = (-l_right[0][0] / l_right[1][0]) * x - (l_right[2][0] / l_right[1][0])
    p1 = (np.float32(x[0]), np.float32(y[0]))
    p2 = (np.float32(x[1]), np.float32(y[1]))
    left_image = cv2.imread("left.jpg")
    cv2.circle(left_image, (np.float32(test_point[0][0]), np.float32(test_point[1][0])), radius=2, thickness=3,
               color=(0, 255, 0))
    cv2.imshow("left_image", left_image)
    right_image = cv2.imread("right.jpg")
    image_final = cv2.line(right_image, p1, p2, (0, 255, 0), 1)
    composite_image = cv2.hconcat((left_image, image_final))
    cv2.imwrite(output_filename, composite_image)
    # cv2.imshow("epipolar line", composite_image)
    # cv2.imshow("Epipolar line", image_final)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


###################################
if __name__ == '__main__':
    main()
