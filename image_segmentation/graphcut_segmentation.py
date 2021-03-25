import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import sys
import time
import warnings
# --------------------------------------


# Disjoint-set data structure for the union-find algorithm
class Disjoint_set:
    def __init__(self, total_pixels):
        self.total_nodes = total_pixels
        # (n*3) matrix containing the parent, rank (of tree) and size (of tree) corresponding to each node
        self.data = np.empty((self.total_nodes, 3), dtype=np.int64)
        # (1*n) matrix containing the internal difference corresponding to each node
        self.Int = np.zeros((self.total_nodes), dtype=np.float64)
        for i in range(self.total_nodes):
            # Parent node
            self.data[i][0] = i
            # Rank of tree
            self.data[i][1] = 0
            # Size of component
            self.data[i][2] = 1
    # Find algorithm
    def find(self, x):
        if self.data[x][0] == x:
            return x
        else:
            self.data[x][0] = self.find(self.data[x][0])
            return self.data[x][0]
    # Union algorithm
    def union(self, x, y):
        x_parent = self.find(x)
        y_parent = self.find(y)
        # Rank of tree1 < Rank of tree2
        if self.data[x_parent][1] < self.data[y_parent][1]:
            # Join tree1 to tree2
            self.data[x_parent][0] = y_parent
            # Increase size of tree2 by size of tree1
            self.data[y_parent][2] += self.data[x_parent][2]
        # Rank of tree1 > Rank of tree2
        elif self.data[x_parent][1] > self.data[y_parent][1]:
            # Join tree2 to tree1
            self.data[y_parent][0] = x_parent
            # Increase size of tree1 by size of tree2
            self.data[x_parent][2] += self.data[y_parent][2]
        # Rank of tree1 = Rank of tree2
        else:
            # Join either tree1 or tree2 to the other; any combination is fine
            self.data[x_parent][0] = y_parent
            # Increment rank of new parent tree by 1
            self.data[y_parent][1] += 1
            # Increase size of new parent tree by size of old parent tree
            self.data[y_parent][2] += self.data[x_parent][2]
# --------------------------------------

# Generate the graph consisting of all nodes (pixels) and edges
def create_graph(img, sigma):
    # Gaussian filter applied on the input image
    img = gaussian_filter(img, sigma)
    rows = img.shape[0]
    cols = img.shape[1]
    total_edges = (rows-1)*cols + (cols-1)*rows
    graph = []
    for channel in range(img.shape[2]):
        edges = np.empty((total_edges, 3), dtype=np.float64)
        index = 0
        for current_row in range(rows):
            for current_col in range(cols):
                # Find edges from top to bottom
                if current_row < (rows - 1):
                    edges[index][0] = (cols*current_row) + current_col
                    edges[index][1] = cols*(current_row+1) + current_col
                    edges[index][2] = abs(img[current_row][current_col][channel] - img[current_row + 1][current_col][channel])
                    index = index + 1
                # Find edges from left to right
                if current_col < (cols - 1):
                    edges[index][0] = (cols*current_row) + current_col
                    edges[index][1] = (cols*current_row) + current_col + 1
                    edges[index][2] = abs(img[current_row][current_col + 1][channel] - img[current_row][current_col][channel])
                    index = index + 1
        # Sort the edges into an ordered list by non-decreasing edge weight
        order = edges[:, 2].argsort()
        edges = edges[order]
        graph.append(edges)
    return graph
# --------------------------------------

# Threshold function for segmentation
def tau(K, tree_size):
    threshold = K / tree_size
    return threshold
# --------------------------------------

# MINT function
def MINT(Int_C1, size_C1, Int_C2, size_C2, K):
    MINT_val = min(Int_C1 + tau(K, size_C1), Int_C2 + tau(K, size_C2))
    return MINT_val
# --------------------------------------

# Graph-based segmentation
def compute_segmentation(graph, dim, K):
    segments_allChannels = []
    for channel in range(len(graph)):
        segment_currentChannel = Disjoint_set(dim[0]*dim[1])
        for i in range(graph[channel].shape[0]):
            # Start node
            x = int(graph[channel][i][0])
            # End node
            y = int(graph[channel][i][1])
            # Weight of edge
            w = graph[channel][i][2]
            # Parent of start node
            x_parent = segment_currentChannel.find(x)
            # Parent of end node
            y_parent = segment_currentChannel.find(y)
            # Start and end nodes lie in separate components and edge weight is less than internal difference
            if (x_parent != y_parent) and (w <= MINT(segment_currentChannel.Int[x_parent],
                                                     segment_currentChannel.data[x_parent][2],
                                                     segment_currentChannel.Int[y_parent],
                                                     segment_currentChannel.data[y_parent][2], K)):
                # Combine the nodes into a single new component
                segment_currentChannel.union(x_parent, y_parent)
                # Update the internal difference of the new component
                Int_newSegment = max(w, segment_currentChannel.Int[x_parent], segment_currentChannel.Int[y_parent])
                segment_currentChannel.Int[x_parent] = Int_newSegment
                segment_currentChannel.Int[y_parent] = Int_newSegment
        segments_allChannels.append(segment_currentChannel)
    return segments_allChannels
# --------------------------------------

# Final output generation according to randomized colors for each segment
def compute_final_image(segments_allChannels, dim):
    rows = dim[0]
    cols = dim[1]
    # Define the final output segmented image
    final_segmented_image = np.empty((rows, cols, 3), dtype=np.float64)
    segments_final = Disjoint_set(rows*cols)
    # Group pixels from left to right iff they belong to the same component across all 3 color channels
    for current_row in range(rows):
        for current_col in range(cols - 1):
            x = (cols*current_row)+ current_col
            y = (cols*current_row) + current_col + 1
            check = []
            for channel in range(len(segments_allChannels)):
                check.append(segments_allChannels[channel].find(x) == segments_allChannels[channel].find(y))
            if all(check):
                segments_final.union(x, y)
    # Group pixels from top to bottom iff they belong to the same component across all 3 color channels
    for current_col in range(cols):
        for current_row in range(rows - 1):
            x = (cols*current_row) + current_col
            y = cols*(current_row+1) + current_col
            check = []
            for channel in range(len(segments_allChannels)):
                check.append(segments_allChannels[channel].find(x) == segments_allChannels[channel].find(y))
            if all(check):
                segments_final.union(x, y)
    # Randomly assign colors to each segment
    for current_row in range(rows):
        for current_col in range(cols):
            current_pixel = segments_final.find((cols * current_row) + current_col)
            np.random.seed(current_pixel+255)
            color = np.random.randint(low=0, high=256, size=3)
            final_segmented_image[current_row][current_col] = color
    return final_segmented_image
# --------------------------------------


def main():
    if len(sys.argv) < 4:
      print('Usage: graphSegment.py inputImageFile k outputImageFile')
      sys.exit(1)

    inputImageFile = sys.argv[1]
    k = int(sys.argv[2])
    outputImageFile = sys.argv[3]
    # --------------------------------------

    if '.gif' in inputImageFile:
        img = Image.open(inputImageFile)
        fileName = inputImageFile.split('.gif')[0]
        img.save(fileName + ".png", 'png')
        inputImageFile = fileName + '.png'

    inputImage = plt.imread(inputImageFile)
    inputImage_normalized = cv2.normalize(inputImage, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    rows = inputImage_normalized.shape[0]
    cols = inputImage_normalized.shape[1]
    graph = create_graph(inputImage_normalized, sigma=0.8)
    if not sys.warnoptions:
        warnings.simplefilter('ignore')
    segments_allChannels = compute_segmentation(graph, (rows, cols), k)
    final_segmented_image = compute_final_image(segments_allChannels, (rows, cols))
    cv2.imwrite(outputImageFile, final_segmented_image)
    # --------------------------------------


if __name__ == '__main__':
    main()