import numpy as np
import cv2
import sys


############################################################
#
#                       KMEANS
#
############################################################

# implement distance metric - e.g. squared distances between pixels
def distance(a, b):
    return np.linalg.norm(a - b)


# k-means works in 3 steps
# 1. initialize
# 2. assign each data element to current mean (cluster center)
# 3. update mean
# then iterate between 2 and 3 until convergence, i.e. until ~smaller than 5% change rate in the error

def update_mean(img, clustermask):
    """This function should compute the new cluster center, i.e. numcluster mean colors"""
    flat = img.flatten()
    flat.reshape((int(flat.shape[0] / 3), 3))
    w, h, _ = clustermask.shape
    cluster_assignees={}
    for cid,_ in enumerate(current_cluster_centers):
        cluster_assignees[cid] = []
    for x in range(w):
        for y in range(h):
            cid = clustermask[x, y][0]
            cluster_assignees[cid].append(img[x,y])
    for cid, pixels in cluster_assignees.items():
        current_cluster_centers[cid] = np.mean(np.array(pixels),axis=0)
    return clustermask


def assign_to_current_mean(img, result, clustermask):
    """The function expects the img, the resulting image and a clustermask.
    After each call the pixels in result should contain a cluster_color corresponding to the cluster
    it is assigned to. clustermask contains the cluster id (int [0...num_clusters]
    Return: the overall error (distance) for all pixels to there closest cluster center (mindistance px - cluster center).
    """
    overall_dist = 0
    w, h, _ = img.shape
    for x in range(w):
        for y in range(h):
            ipixel = img[x, y]
            dists = {}
            for i, c in enumerate(current_cluster_centers):
                dists[i] = distance(ipixel, c)
            cid, dist = min(dists.items(), key=lambda d: d[1])
            clustermask[x, y] = cid
            result[x, y] = current_cluster_centers[cid]
            overall_dist += dist
    return overall_dist

def refill_real(img, result, clustermask, cluster_colors):
    """fill real with the vals of cluster_colors
    """
    overall_dist = 0
    w, h, _ = img.shape
    for x in range(w):
        for y in range(h):
            cid = clustermask[x, y]
            result[x, y] = cluster_colors[cid]

def initialize(img):
    """inittialize the current_cluster_centers array for each cluster with a random pixel position"""
    w, h, _ = img.shape
    for c in current_cluster_centers:
        x = np.random.randint(w)
        y = np.random.randint(h)
        c[:] = img[x, y]


def kmeans(img):
    """Main k-means function iterating over max_iterations and stopping if
    the error rate of change is less then 2% for consecutive iterations, i.e. the
    algorithm converges. In our case the overall error might go up and down a little
    since there is no guarantee we find a global minimum.
    """
    max_iter = 10
    max_change_rate = 0.02
    dist = sys.float_info.max

    clustermask = np.zeros((h1, w1, 1), np.uint8)
    result = np.zeros((h1, w1, 3), np.uint8)

    # initializes each pixel to a cluster
    # iterate for a given number of iterations or if rate of change is
    # very small
    initialize(img)
    i = 0
    while i < max_iter and dist > max_change_rate:
        assign_to_current_mean(img, result, clustermask)
        clustermask = update_mean(img, clustermask)
        i += 1
    refill_real(img, result, clustermask, cluster_colors)
    return result


# num of cluster
numclusters = 3
# corresponding colors for each cluster
cluster_colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [128, 128, 128]],np.uint8)
# initialize current cluster centers (i.e. the pixels that represent a cluster center)
current_cluster_centers = np.zeros((numclusters, 3), np.uint8)

# load image
imgraw = cv2.imread('./images/Lenna.png')
scaling_factor = 0.5
imgraw = cv2.resize(imgraw, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

# compare different color spaces and their result for clustering
# YOUR CODE HERE or keep going with loaded RGB colorspace img = imgraw
image = imgraw
h1, w1 = image.shape[:2]

# execute k-means over the image
# it returns a result image where each pixel is color with one of the cluster_colors
# depending on its cluster assignment
res = kmeans(image)

h1, w1 = res.shape[:2]
h2, w2 = image.shape[:2]
vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
vis[:h1, :w1] = res
vis[:h2, w1:w1 + w2] = image

while True:
    cv2.imshow("Color-based Segmentation Kmeans-Clustering", vis)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
cv2.destroyAllWindows()
