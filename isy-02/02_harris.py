import cv2
import numpy as np

# Load image and convert to gray and floating point
img = cv2.imread('./images/Lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Define sobel filter and use cv2.filter2D to filter the grayscale image
sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel_x = cv2.filter2D(gray,-1, sobelmask_x)
sobel_y = cv2.filter2D(gray,-1, sobelmask_y)
# Compute G_xx, G_yy, G_xy and sum over all G_xx etc. 3x3 neighbors to compute
# entries of the matrix M = \sum_{3x3} [ G_xx Gxy; Gxy Gyy ]
# Note1: this results again in 3 images sumGxx, sumGyy, sumGxy
# Hint: to sum the neighbor values you can again use cv2.filter2D to do this efficiently
Gxx = sobel_x ** 2
Gyy = sobel_y ** 2
Gxy = np.multiply(sobel_x, sobel_y)

addition_kernel = np.ones((3,3))
sumGxx = cv2.filter2D(Gxx, -1, addition_kernel)
sumGyy = cv2.filter2D(Gyy, -1, addition_kernel)
sumGxy = cv2.filter2D(Gxy, -1, addition_kernel)

# Define parameter
k = 0.04
threshold = 0.01

# Compute the determinat and trace of M using sumGxx, sumGyy, sumGxy. With det(M) and trace(M)
# you can compute the resulting image containing the harris corner responses with
det = sumGxx * sumGyy - sumGxy * sumGxy
trace = sumGxx + sumGyy
harris = det - k * trace**2

# Filter the harris 'image' with 'harris > threshold*harris.max()'
# this will give you the indices where values are above the threshold.
# These are the corner pixel you want to use
harris_thres = np.zeros(harris.shape)
harris_thres[harris > threshold * harris.max()] = [255]

# The OpenCV implementation looks like this - please do not change
harris_cv = cv2.cornerHarris(gray, 3, 3, k)

# intialize in black - set pixels with corners in with
harris_cv_thres = np.zeros(harris_cv.shape)
harris_cv_thres[harris_cv > threshold * harris_cv.max()] = [255]

# just for debugging to create such an image as seen
# in the assignment figure.
img[harris>threshold*harris.max()]=[255,0,0]


# please leave this - adjust variable name if desired
print("====================================")
print("DIFF:", np.sum(np.absolute(harris_thres - harris_cv_thres)))
print("====================================")

cv2.imwrite("Harris_own.png", harris_thres)
cv2.imwrite("Harris_cv.png", harris_cv_thres)
cv2.imwrite("Image_with_Harris.png", img)

while True:
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break

    cv2.imshow('harris',harris_thres)
    cv2.imshow('harris_cv',harris_cv_thres)
    cv2.imshow('img', img)
cv2.destroyAllWindows()