#!/usr/bin/python

import cv2
import heapq
import numpy as np
from scipy import ndimage
import sys
import time


# Function to return the data of neighboring pixels (if they exist)
#######################################
def get_neighbor(seed_coordinates, connectivity_8, image_gradients):
    neighbor_data = []
    # 8-connectivity
    if(connectivity_8 == 1):
        for row in range(seed_coordinates[0]-1, seed_coordinates[0]+2):
            for col in range(seed_coordinates[1]+1, seed_coordinates[1]-2, -1):
                if((row < 0) or (row >= image_gradients.shape[0]) or (col < 0) or
                        (col >= image_gradients.shape[1]) or (row==seed_coordinates[0] and col==seed_coordinates[1])):
                    continue
                else:
                    neighbor_data.append((image_gradients[row][col], [row, col]))
    # 4-connectivity
    else:
        for row in range(seed_coordinates[0]-1, seed_coordinates[0]+2):
            for col in range(seed_coordinates[1]+1, seed_coordinates[1]-2, -1):
                if ((row < 0) or (row >= image_gradients.shape[0]) or (col < 0) or
                    (col >= image_gradients.shape[1]) or (row==seed_coordinates[0] and col==seed_coordinates[1]) or
                    (row==seed_coordinates[0]-1 and col==seed_coordinates[1]+1) or
                    (row==seed_coordinates[0]+1 and col==seed_coordinates[1]+1) or
                    (row==seed_coordinates[0]-1 and col==seed_coordinates[1]-1) or
                    (row==seed_coordinates[0]+1 and col==seed_coordinates[1]-1)):
                    continue
                else:
                    neighbor_data.append((image_gradients[row][col], [row, col]))
    # Return neighboring pixels data
    return neighbor_data
# --------------------------------------


def main():
    if len(sys.argv) < 4:
      print('Usage: wshedSegment.py inputImageFile inputSeedFile outputImageFile')
      sys.exit(1)

    inputImageFile = sys.argv[1]
    inputSeedFile = sys.argv[2]
    outputImageFile = sys.argv[3]
    # --------------------------------------

    # Prepare input and output images
    #######################################
    # Read input image, smoothen with Gaussian noise and compute intensity gradients
    img = cv2.imread(inputImageFile, 0)
    smooth_img_gradient = ndimage.gaussian_gradient_magnitude(img, sigma=0.8)
    # Define empty final segmented image
    segmented_image = np.empty((smooth_img_gradient.shape[0], smooth_img_gradient.shape[1]))
    # -------------------------------------

    # Some initializations
    #######################################
    # Initialize elements of the pixel label matrix with -1 denoting background pixels
    pixel_labels = np.empty((smooth_img_gradient.shape[0], smooth_img_gradient.shape[1]))
    pixel_labels.fill(-1)
    # Define empty priority queue
    priority = []
    # Set 8-connectivity as true/false
    connectivity_8 = 0
    # Define empty dictionary to ensure duplicate entries are not pushed into the priority queue
    d = {}
    # -------------------------------------

    # Prepare seed data from input seed file - seed_data is a 2D array with 3 cols (intensity, x, y)
    #######################################
    file_seeds = open(inputSeedFile, "r")
    seeds_string = file_seeds.readlines()
    seed_data = []
    for line in seeds_string:
        seed_record_string = line.split()
        seed_data.append([int(float(seed_record_string[0])), int(float(seed_record_string[1])), int(float(seed_record_string[2]))])
    # -------------------------------------

    # Initialize those elements of the pixel label matrix that have corresponding labels in seed data
    # Also obtain neighbors of those pixels and push their data - (intensity, [x,y]) - into the priority heap queue
    #######################################
    for seed_record in seed_data:
        pixel_labels[seed_record[1]][seed_record[2]] = seed_record[0]
        neighbor_data = get_neighbor((seed_record[1], seed_record[2]), connectivity_8, smooth_img_gradient)
        for n in neighbor_data:
            if str(n) not in d:
                heapq.heappush(priority, n)
                d[str(n)] = True
    # -------------------------------------

    # Iterate until all elements of priority heap queue are labeled and entered into the pixel label matrix
    # Push and pop elements to/from the priority heap according to the logic of the algorithm
    #######################################
    while len(priority) > 0:
        # Pop a new pixel from priority heap queue
        new_pixel = heapq.heappop(priority)
        d.pop(str(new_pixel))
        # Obtain data of the new pixel's neighbors
        new_neighbor_data = get_neighbor((new_pixel[1][0], new_pixel[1][1]), connectivity_8, smooth_img_gradient)
        # Keep track of the label (if exists) of each neighbor
        neighbor_labels = []
        # Iterate over all neighbors; push a neighbor into the priority heap queue if it does not have a label
        for neighbor in new_neighbor_data:
            # Label does not exist
            if pixel_labels[neighbor[1][0]][neighbor[1][1]] == -1:
                if str(neighbor) not in d:
                    heapq.heappush(priority, neighbor)
                    d[str(neighbor)] = True
            # Label exists
            else:
                neighbor_labels.append(pixel_labels[neighbor[1][0]][neighbor[1][1]])
        # All labeled neighbors are not from the same segment - the new pixel is a watershed
        if len(set(neighbor_labels)) > 1:
            pixel_labels[new_pixel[1][0]][new_pixel[1][1]] = 0
        # All labeled neighbors are from the same segment - the new pixel also belongs to the same segment
        elif len(set(neighbor_labels)) == 1:
            pixel_labels[new_pixel[1][0]][new_pixel[1][1]] = neighbor_labels[0]
        elif len(set(neighbor_labels)) == 0:
            print('This case should not occur!')
    # -------------------------------------

    # Create the output segmented image according to the labels computed in the pixel label matrix
    #######################################
    # Map labels to color values
    label_to_color_map = {}
    label_to_color_map[0] = 0
    iter_color = 125
    for iter_label in range(1, 10):
        label_to_color_map[iter_label] = iter_color
        iter_color = iter_color + 7
    # Obtain the final segmented image output
    for row in range(segmented_image.shape[0]):
        for col in range(segmented_image.shape[1]):
            final_label = pixel_labels[row][col]
            if(final_label == 0):
                segmented_image[row][col] = label_to_color_map[final_label]
            else:
                segmented_image[row][col] = label_to_color_map[final_label]
    # -------------------------------------

    # Write the segmented image output to a file
    #######################################
    cv2.imwrite(outputImageFile, segmented_image)


if __name__ == '__main__':
    main()