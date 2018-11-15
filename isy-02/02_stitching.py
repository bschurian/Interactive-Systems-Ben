import numpy as np
import cv2
import math
import sys
from ImageStitcher import *

############################################################
#
#                   Image Stitching
#
############################################################

# 1. load panorama images
imgs = []
filenames = ["images/pano3.jpg","images/pano2.jpg", "images/pano1.jpg"]
# filenames = ["images/pano6.jpg","images/pano5.jpg", "images/pano4.jpg"]
for filename in filenames:
    imgs.append(cv2.imread(filename, 1))

# order of input images is important is important (from right to left)
imageStitcher = ImageStitcher(imgs)  # list of images
(matchlist, result) = imageStitcher.stitch_to_panorama()

if not matchlist:
    print("We have not enough matching keypoints to create a panorama")
else:
    print("Done")
    while True:
        # output all matching images
        # cv2.imshow("singles",imgs)
        # output result
        # Note: if necessary resize the image
        cv2.imshow("Panorama", result)
        name = ""
        if "images/pano1.jpg" in filenames:
            name = "1To3"
        elif "images/pano4.jpg" in filenames:
            name = "4To6"
        else:
            name = "error"
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            break
cv2.imwrite("panorama-" + name + ".jpg", result)
cv2.imwrite("last-drawn-matches-" + name + ".jpg", matchlist[len(matchlist)-1])
cv2.destroyAllWindows()
