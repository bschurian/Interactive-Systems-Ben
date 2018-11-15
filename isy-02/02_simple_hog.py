import numpy as np
import cv2
import math
import matplotlib
import glob
import math

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


###############################################################
#
# Write your own descriptor / Histogram of Oriented Gradients
#
###############################################################


def plot_histogram(hist, bins,filename):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2

    plt.bar(center, hist, align='center', width=width)
    plt.show()


def compute_simple_hog(imgcolor, keypoints,filename):
    # convert color to gray image and extract feature in gray
    imggray = cv2.cvtColor(imgcolor, cv2.COLOR_BGR2GRAY)

    imggray = np.float32(imggray) / 255.0

    # compute x and y gradients (sobel kernel size 5)
    xSobel = cv2.Sobel(imggray, cv2.CV_64F, 1, 0, ksize=5)
    ySobel = cv2.Sobel(imggray, cv2.CV_64F, 0, 1, ksize=5)

    # compute magnitude and angle of the gradients
    # magnitudes, angles = cv2.cartToPolar(xSobel, ySobel, angleInDegrees=False)
    magnitudes = cv2.magnitude(xSobel,ySobel)
    angles = cv2.phase(xSobel,ySobel,angleInDegrees=False)
    # print(magnitudes, angles)

    # go through all keypoints and and compute feature vector
    descr = np.zeros((len(keypoints), 8), np.float32)
    count = 0
    for kp in keypoints:
        x0, y0 = kp.pt
        x0 = int(x0)
        y0 = int(y0)
        offset = int(kp.size / 2)
        # extract angle in keypoint sub window
        kp_angles = angles[x0 - offset:x0 + offset + 1, y0 - offset:y0 + offset + 1]
        # extract gradient magnitude in keypoint subwindow
        kp_magnitudes = magnitudes[x0 - offset:x0 + offset + 1, y0 - offset:y0 + offset + 1]

        # create histogram of angle in subwindow BUT only where magnitude of gradients is non zero! Why? Find an
        # answer to that question use np.histogram
        usefull_kp_angles = [angle for (angle, magnitude) in zip(kp_angles.flatten(), kp_magnitudes.flatten()) if
                             magnitude != 0]
        (hist, bins) = np.histogram(usefull_kp_angles, bins=8, range=(0.0, 2*math.pi),density=True)

        plot_histogram(hist, bins,filename)

        descr[count] = hist

        count += 1
    return descr

keypoints = [cv2.KeyPoint(15, 15, 11)]

# test for all test images
res = []
filenames = glob.glob('./images/hog_test/*.jpg')
for filename in filenames:
    img = cv2.imread(filename)
    res.append(compute_simple_hog(img, keypoints,filename))
cv2.destroyAllWindows()