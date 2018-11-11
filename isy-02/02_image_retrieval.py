import cv2
import glob
import numpy as np
from queue import PriorityQueue
from math import sqrt


############################################################
#
#              Simple Image Retrieval
#
############################################################


# implement distance function
# euclidean dist
def distance(a, b):
    return np.sqrt(np.sum((a-b)**2))


def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11
    offset = int(keypointSize / 2)
    # keypoints = np.dstack(np.meshgrid(np.arange(w / 11), np.arange(h / 11), indexing='ij'))
    for x in range(offset + 1, h - offset, keypointSize):
        for y in range(offset + 1, w - offset, keypointSize):
            keypoints.append(cv2.KeyPoint(x, y, keypointSize))
    return keypoints


# 1. preprocessing and load
imageFilenames = glob.glob('./images/db/*/*.jpg')
images = []
for imageFilename in imageFilenames:
    images.append(cv2.imread(imageFilename, 1))

# 2. create keypoints on a regular grid (cv2.KeyPoint(r, c, keypointSize), as keypoint size use e.g. 11)
descriptors = []
keypoints = create_keypoints(256, 256)

# 3. use the keypoints for each image and compute SIFT descriptors
#    for each keypoint. this compute one descriptor for each image.
sift = cv2.xfeatures2d.SIFT_create()
descriptorsNImages = []
for image in images:
    descriptor = sift.compute(image, keypoints)[1]
    descriptorsNImages.append((descriptor, image))

def imagesSortedbyFeatureDist(queryImg, descriptorsNImages):
    # 4. use one of the query input image to query the 'image database' that
    #    now compress to a single area. Therefore extract the descriptor and
    #    compare the descriptor to each image in the database using the L2-norm
    #    and save the result into a priority queue (q = PriorityQueue())
    queryImgDescriptor = sift.compute(queryImg, keypoints)[1]
    distanceNImages = [(distance(des, queryImgDescriptor), img) for des, img in descriptorsNImages]
    # sortedDescriptorsNImages = sorted(descriptorsNImages, key=lambda des: distance(des[0], queryImgDescriptor), reverse=True)
    que = PriorityQueue()
    for dis, img in distanceNImages:
        # while True:
        #     cv2.imshow(""+str(dis), img)
        #     ch = cv2.waitKey(1) & 0xFF
        #     if ch == ord('q'):
        #         break
        que.put((dis, img))

    enlargedQImg = cv2.resize(queryImg, (queryImg.shape[0] * 2, queryImg.shape[1] * 2))
    queLen = que.qsize()
    i = 0
    firstRow = []
    secondRow = []
    while i < queLen and not que.empty():
        des,img = que.get()
        print(des)
        img = cv2.resize(img,(queryImg.shape[0],queryImg.shape[1]))
        if i < int((queLen+1) / 2):
            firstRow.append(img)
        else:
            # secondRow.append(np.ones(images[0].shape, images[0].dtype) * 255)
            secondRow.append(img)
        i+=1
    if queLen % 2 != 0:
        white = np.ones(images[0].shape, images[0].dtype) * 255
        secondRow.append(white)
    firstRowConc = cv2.hconcat(firstRow)
    secondRowConc = cv2.hconcat(secondRow)
    return cv2.hconcat([enlargedQImg, cv2.vconcat([firstRowConc, secondRowConc])])

queryFilenames = glob.glob('./images/db/*.jpg')
queries = []
for queryFilename in queryFilenames:
    qAndImagesSorted = imagesSortedbyFeatureDist(cv2.imread(queryFilename, 1),descriptorsNImages)
    queries.append(qAndImagesSorted)
resultImg = cv2.vconcat(queries)

# 5. output (save and/or display) the query results in the order of smallest distance
cv2.imwrite("result.jpg", resultImg)
while True:
    cv2.imshow("res", resultImg)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
cv2.destroyAllWindows()
