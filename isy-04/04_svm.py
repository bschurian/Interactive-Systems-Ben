import numpy as np
import cv2
import glob
from sklearn import svm
import re


############################################################
#
#              Support Vector Machine
#              Image Classification
#
############################################################
def create_keypoints(w, h):
    keypoints = []
    keypointSize = 11
    offset = int(keypointSize / 2)
    # keypoints = np.dstack(np.meshgrid(np.arange(w / 11), np.arange(h / 11), indexing='ij'))
    for x in range(offset + 1, h - offset, keypointSize):
        for y in range(offset + 1, w - offset, keypointSize):
            keypoints.append(cv2.KeyPoint(x, y, keypointSize))
    return keypoints


# 1. Implement a SIFT feature extraction for a set of training images ./images/db/train/** (see 2.3 image retrieval)
# use 256x256 keypoints on each image with subwindow of 15x15px
imageFilenames = glob.glob('./images/db/train/*/*.jpg')
images = []
for imageFilename in imageFilenames:
    images.append(cv2.imread(imageFilename, 1))
descriptors = []
keypoints = create_keypoints(256, 256)
sift = cv2.xfeatures2d.SIFT_create()
for image in images:
    descriptor = sift.compute(image, keypoints)[1]
    descriptors.append(descriptor)

# 2. each descriptor (set of features) need to be flattened in one vector
# That means you need a X_train matrix containing a shape of (num_train_images, num_keypoints*num_entry_per_keypoint)
# num_entry_per_keypoint = histogram orientations as talked about in class
# You also need a y_train vector containing the labels encoded as integers
num_train_images = len(images)
num_keypoints = len(keypoints)
# TODO:Wieso
num_entry_per_keypoint = 128

X_train = np.reshape(descriptors, (num_train_images, num_keypoints * num_entry_per_keypoint))

classes_list = []
imageClasses = []
for imageFilename in imageFilenames:
    match = re.search('..images.db.train.*?([a-z]+).*.jpg', imageFilename).group(1)
    imageClasses.append(match)
    classes_list.append(match)
classes_list = list(set(classes_list))
classes = {}
for id, class_name in enumerate(classes_list):
    classes[class_name] = id

y_train = [classes[imageClass] for imageClass in imageClasses]

# 3. We use scikit-learn to train a SVM classifier. Specifically we use a LinearSVC in our case. Have a look at the documentation.
# You will need .fit(X_train, y_train)
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)

# 4. We test on a variety of test images ./images/db/test/ by extracting an image descriptor
# the same way we did for the training (except for a single image now) and use .predict()
# to classify the image
test_imageFilenames = glob.glob('./images/db/test/*.jpg')
test_images = []
for imageFilename in test_imageFilenames:
    test_images.append(cv2.imread(imageFilename, 1))
test_descriptors = []
for test_image in test_images:
    test_descriptor = sift.compute(test_image, keypoints)[1]
    test_descriptors.append(test_descriptor)
test_num_train_images = len(test_images)
test_num_keypoints = len(keypoints)
test_num_entry_per_keypoint = 128
X_Test = np.reshape(test_descriptors, (test_num_train_images, test_num_keypoints * test_num_entry_per_keypoint))
test_imageClasses = []
for test_imageFilename in test_imageFilenames:
    test_match = re.search('..images.db.test.*?([a-z]+).*.jpg', test_imageFilename).group(1)
    # special case because the file and folders have different names
    test_imageClasses.append(test_match + 's')

y_Test = [classes[test_imageClass] for test_imageClass in test_imageClasses]

# 5. output the class + corresponding name
x_test_preds_ids = lin_clf.predict(X_Test)
#unique names using whitespace
x_test_preds_strings = [classes_list[id]+" "*i for i,id in enumerate(x_test_preds_ids)]

test_imagesNPreds = zip(test_images, x_test_preds_strings)

while True:
    for test_image, pred in test_imagesNPreds:
        cv2.imshow(pred, test_image)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
cv2.destroyAllWindows()
