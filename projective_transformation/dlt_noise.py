import cv2
import numpy as np
import sys

TEST_POINT = (150, 90)

REF_POINTS = [(130, 100),
              (140, 107),
              (150, 110),
              (160, 107),
              (170, 100)]


# read given reference points and test point
###################################
def read_keypoints():
    left_matrix = [[], [], []]
    right_matrix = [[], [], []]

    for i in range(len(REF_POINTS)):
        left_matrix[0].append(REF_POINTS[i][0])
        left_matrix[1].append(REF_POINTS[i][1])
        left_matrix[2].append(1)

        right_matrix[0].append(REF_POINTS[i][0])
        right_matrix[1].append(REF_POINTS[i][1])
        right_matrix[2].append(1)

    left_matrix = np.array(left_matrix)
    right_matrix = np.array(right_matrix)

    return left_matrix, right_matrix


# add Gaussian noise with mean=0 and std to the reference points
###################################
def add_noise(mean, std, noisy_matrix):
    mu = [mean, mean]
    sigma = [std, std]
    cov = [sigma, sigma]
    noise = np.random.multivariate_normal(mu, cov, len(noisy_matrix))

    for i in range(len(noise)):
        noisy_matrix[0][i] += noise[i][0]
        noisy_matrix[1][i] += noise[i][1]

    return noisy_matrix


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


# formulate A to solve Ah = 0
###################################
def compute_A(left_matrix_normalized, right_matrix_normalized):
    A = []
    for i in range(len(left_matrix_normalized[0])):
        A.append([
            0, 0, 0,
            -left_matrix_normalized[0][i], -left_matrix_normalized[1][i], -1,
            right_matrix_normalized[1][i] * left_matrix_normalized[0][i],
            right_matrix_normalized[1][i] * left_matrix_normalized[1][i],
            right_matrix_normalized[1][i] * 1
        ])
        A.append([
            left_matrix_normalized[0][i], left_matrix_normalized[1][i], 1,
            0, 0, 0,
            -right_matrix_normalized[0][i] * left_matrix_normalized[0][i],
            -right_matrix_normalized[0][i] * left_matrix_normalized[1][i],
            -right_matrix_normalized[0][i] * 1
        ])

    return A


# synthesize the canvas to display all points
###################################
def create_canvas():
    global canvas_obj
    canvas_obj = np.zeros((512, 512, 3), dtype="uint8")
    return canvas_obj


###################################
def main():
    if len(sys.argv) < 5:
        print('Usage: dlt.py std n_iterations output normalize')
        sys.exit(1)

    std = float(sys.argv[1])
    n_iterations = int(sys.argv[2])
    output_filename = sys.argv[3]
    normalize = bool(int(sys.argv[4]))

    if normalize:
        print('With normalization')
    else:
        print('Without normalization')

    final_points_distribution = []
    for i in range(n_iterations):
        # Store the keypoints of the manually-selected corresponding points in homogeneous form
        left_matrix, right_matrix = read_keypoints()

        # Add Gaussian noise to the matrices
        noisy_matrix = add_noise(0, std, right_matrix)

        if normalize:
            # Normalize the matrices
            left_matrix_normalized, right_matrix_normalized, left_T, right_T = normalization(left_matrix, noisy_matrix)
        else:
            left_matrix_normalized = left_matrix
            right_matrix_normalized = right_matrix

        # Formulate the A matrix to solve the equation Ah = 0
        A = compute_A(left_matrix_normalized, right_matrix_normalized)

        # Compute the SVD of A to obtain the least squares solution, and then obtain the normalized homography matrix
        U_temp, D_temp, V_t_temp = np.linalg.svd(A)
        V_temp = np.transpose(V_t_temp)
        h = []
        for i in range(len(V_temp)):
            h.append(V_temp[i][-1])
        H_temp = np.reshape(h, (3, 3))

        if normalize:
            # Denormalize the homography matrix
            H = np.matmul(np.matmul(np.linalg.inv(right_T), H_temp), left_T)
        else:
            H = H_temp

        # Convert the given test point to homogeneous coordinates
        test_point = [[TEST_POINT[0]], [TEST_POINT[1]], [1]]
        final_point_temp = np.matmul(H, test_point)
        final_point = final_point_temp / final_point_temp[2][0]
        final_points_distribution.append([final_point[0][0], final_point[1][0]])

    # Final Monte Carlo simulations output
    final_image = create_canvas()
    cv2.circle(final_image, (TEST_POINT[0], TEST_POINT[1]), radius=1, thickness=2, color=(0, 0, 255))
    for i in range(len(REF_POINTS)):
        cv2.circle(final_image, (REF_POINTS[i][0], REF_POINTS[i][1]), radius=1, thickness=1, color=(0, 0, 255))
    for i in range(len(final_points_distribution)):
        cv2.circle(final_image,
                   (np.float32(final_points_distribution[i][0]), np.float32(final_points_distribution[i][1])), radius=1,
                   thickness=1, color=(255, 255, 0))
    cv2.imwrite(output_filename, final_image)
    # cv2.imshow("Monte Carlo Simulations Experiment", final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


###################################
if __name__ == '__main__':
    main()
