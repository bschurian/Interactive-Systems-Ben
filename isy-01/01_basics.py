import numpy as np
import cv2
import math
import time

######################################################################
# IMPORTANT: Please make yourself comfortable with numpy and python:
# e.g. https://www.stavros.io/tutorials/python/
# https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

# Note: data types are important for numpy and opencv
# most of the time we'll use np.float32 as arrays
# e.g. np.float32([0.1,0.1]) equal np.array([1, 2, 3], dtype='f')


######################################################################
# A2. OpenCV and Transformation and Computer Vision Basic

# (1) read in the image Lenna.png using opencv in gray scale and in color
# and display it NEXT to each other (see result image)
# Note here: the final image displayed must have 3 color channels
#            So you need to copy the gray image values in the color channels
#            of a new image. You can get the size (shape) of an image with rows, cols = img.shape[:2]

# why Lenna? https://de.wikipedia.org/wiki/Lena_(Testbild)

# (2) Now shift both images by half (translation in x) it rotate the colored image by 30 degrees using OpenCV transformation functions
# + do one of the operations on keypress (t - translate, r - rotate, 'q' - quit using cv::warpAffine
# http://docs.opencv.org/3.1.0/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
# Tip: you need to define a transformation Matrix M
# see result image

# cap = cv2.VideoCapture(0)
# _, frame = cap.read()
img = cv2.imread('images/Lenna.png')
grayImg = cv2.imread('images/Lenna.png', 0)
# frame = np.concatenate(img,grayImg)
# grayImg = np.expand_dims(grayImg,axis=1)
irows, icols, icol = img.shape
grayImg = np.repeat(grayImg, 3).reshape((irows, icols, icol))
frame = np.concatenate((grayImg,img), axis=1)
rows, cols, col = frame.shape

Sx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
Sy = Sx.transpose()

while (True):
    mode = 0
    # Capture frame-by-frame

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1
    if ch == ord('t'):
        mode = 2
    if ch == ord('r'):
        mode = 3
    # if ch == ord('f'):
    #     mode = 4

    if ch == ord('q'):
        break
    if mode == 1:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if mode == 2:
        M = np.float32([[1, 0, 100], [0, 1, 0]])
        frame = cv2.warpAffine(frame, M, (cols, rows))
    if mode == 3:
        M = cv2.getRotationMatrix2D((cols/2,rows/2),30,1)
        frame = cv2.warpAffine(frame, M, (cols, rows))
    # if mode == 4:
    #     M = np.float32([[1, 0, 100], [0, 1, 0]])
    #     frame = cv2.warpAffine(frame, M, (cols, rows))

    # Display the resulting frame
    cv2.imshow('frame', frame)

print(frame.shape)

cv2.destroyAllWindows()
