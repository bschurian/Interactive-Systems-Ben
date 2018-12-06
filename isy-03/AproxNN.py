import numpy as np
import cv2
from matplotlib import pyplot as plt


def draw_matches(img1, img2, kp1, kp2, matches):
    """For each pair of points we draw a line between both images and circles,
    then connect a line between them.
    Returns a new image containing the visualized matches
    """

    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
    vis[0:h1, 0:w1] = img1
    vis[0:h2, w1:] = img2

    for (idx2, idx1) in matches:
        # x - columns
        # y - rows
        (x1, y1) = kp1[idx1].pt
        (x2, y2) = kp2[idx2].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(vis, (int(x1), int(y1)), 4, (255, 255, 0), 1)
        cv2.circle(vis, (int(x2) + w1, int(y2)), 4, (255, 255, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(vis, (int(x1), int(y1)), (int(x2) + w1, int(y2)), (255, 0, 0), 1)
    return vis


marker = cv2.imread('images/marker.jpg', 1)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(marker, None)

cap = cv2.VideoCapture(0)
#my cam is very dark at the beginning
#rm this loop if your cam is better
while True:
    ret, frame = cap.read()

    cv2.imshow("press q if camera lights up",frame)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
cv2.destroyAllWindows()
while True:

    ret, frame = cap.read()

    cv2.imshow("frame",frame)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break

    # plt.imshow(frame), plt.show()
    # continue

    # find the keypoints and descriptors with SIFT
    kp2, des2 = sift.detectAndCompute(frame, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()

    rawmatches = bf.knnMatch(des1, des2, k=2)
    matches = []

    # loop over the raw matches and filter them
    for m in rawmatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. David Lowe's ratio = 0.75)
        # in other words - panorama images need some useful overlap
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    print("matches:", len(matches))
    # we need to compute a homography - more next course
    # computing a homography requires at least 4 matches
    if len(matches) < 20:
        continue

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = draw_matches(marker, frame, kp1, kp2, matches)


    cv2.imshow("im",img3)
cv2.destroyAllWindows()
