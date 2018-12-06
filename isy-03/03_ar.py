import cv2
import numpy as np

# global constants
min_matches = 10
FLANN_INDEX_KDTREE = 0
indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=50)

# initialize flann and SIFT extractor
# note unfortunately in the latest OpenCV + python is a minor bug in the flann
# flann = cv2.FlannBasedMatcher(indexParams, {})
# so we use the alternative but slower Brute-Force Matcher BFMatcher
sift = cv2.xfeatures2d.SIFT_create()
flann = cv2.BFMatcher()

# extract marker descriptors
marker = cv2.imread('images/marker.jpg', 1)
keypointsMarker, descriptorsMarker = sift.detectAndCompute(marker, None)


def render_virtual_object(img, x_start, y_start, x_end, y_end, quad):
    # define vertices, edges and colors of your 3D object, e.g. cube
    vertices = np.float32([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0],
                           [0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]])
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7), (0, 3), (1, 2), (5, 6), (4, 7)]

    color_lines = (0, 0, 0)

    # define quad plane in 3D coordinates with z = 0
    quad_3d = np.float32([[x_start, y_start, 0], [x_end, y_start, 0],
                          [x_end, y_end, 0], [x_start, y_end, 0]])

    h, w = img.shape[:2]
    # define intrinsic camera parameter
    K = np.float64([[w, 0, 0.5 * (w - 1)],
                    [0, w, 0.5 * (h - 1)],
                    [0, 0, 1.0]])

    # find object pose from 3D-2D point correspondences of the 3d quad using Levenberg-Marquardt optimization
    # in order to work we need K (given above and YOUR distortion coefficients from Assignment 2 (camera calibration))
    # H, status = cv2.findHomography(,,cv2.RANSAC, 3.0)
    dist_coef = np.array([])


    # compute extrinsic camera parameters using cv2.solvePnP
    # define quad plane in 3D coordinates with z = 0
    quad_3d = np.float32([[x_start, y_start, 0], [x_end, y_start, 0], [x_end, y_end, 0], [x_start, y_end, 0]])

    # compute extrinsic camera parameters using cv2.solvePnP
    ret, rvec, tvec = cv2.solvePnP(quad_3d, quad, K, distCoeffs=dist_coef)

    # transform vertices: scale and translate form 0 - 1, in window size of the marker
    scale = [x_end - x_start, y_end - y_start, x_end - x_start]
    trans = [x_start, y_start, -x_end - x_start]

    verts = scale * vertices + trans

    # call cv2.projectPoints with verts, and solvePnP result, K, and dist_coeff
    # returns a tuple that includes the transformed vertices as a first argument
    verts, _= cv2.projectPoints(verts,rvec,tvec,K,dist_coef)

    # we need to reshape the result of projectPoints
    verts = verts.reshape(-1, 2)

    # render edges
    for i, j in edges:
        (x_start, y_start), (x_end, y_end) = verts[i], verts[j]
        cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), color_lines, 2)


cap = cv2.VideoCapture(0)
# my cam is very dark at the beginning
# rm this loop if your cam is better
while True:
    ret, frame = cap.read()

    cv2.imshow("press q if camera lights up", frame)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break
cv2.destroyAllWindows()

while True:
    ret, frame = cap.read()

    # # TODO RM
    # frame = marker

    cv2.imshow("frame", frame)
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break

    # detect and compute descriptor in camera image
    # and match with marker descriptor
    keypointsFrame, descriptorsFrame = sift.detectAndCompute(frame, None)
    matches = flann.knnMatch(descriptorsMarker, descriptorsFrame, k=2)

    # filter matches by distance [Lowe2004]
    lM_Befor = len(matches)
    matches = [match[0] for match in matches if len(match) == 2 and
               match[0].distance < match[1].distance * 0.75]
    print(lM_Befor,len(matches))
    # if there are less than min_matches we just keep going looking
    # early break
    if len(matches) < min_matches:
        cv2.imshow('too few matches', frame)
        continue

    # extract 2d points from matches data structure
    p0 = [keypointsFrame[m.trainIdx].pt for m in matches]
    p1 = [keypointsMarker[m.queryIdx].pt for m in matches]
    # transpose vectors
    p0, p1 = np.array([p0, p1])

    # we need at least 4 match points to find a homography matrix
    if len(p0) < 4:
        cv2.imshow('too few matchpoints', frame)
        continue

    # find homography using p0 and p1, returning H and status
    # H - homography matrix
    # status - status about inliers and outliers for the plane mapping
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)

    # on the basis of the status object we can now filter RANSAC outliers
    mask = mask.ravel() != 0
    if mask.sum() < min_matches:
        cv2.imshow('Interactive Systems: AR Tracking', frame)
        continue

    # take only inliers - mask of Outlier/Inlier
    # p0, p1 = p0[mask], p1[mask]
    # get the size of the marker and form a quad in pixel coords np float array using w/h as the corner points
    p0, p1 = p0[mask], p1[mask]
    wm, hm, _ = marker.shape
    quad = np.array([[0, 0], [0, hm], [wm, hm], [wm, 0]], np.float32)

    # perspectiveTransform needs a 3-dimensional array
    # quad = np.array([quad])
    quad_transformed = cv2.perspectiveTransform(quad.reshape(1, -1, 2), H)
    # transform back to 2D array
    quad = quad_transformed.reshape(-1, 2)

    # render quad in image plane and feature points as circle using cv2.polylines + cv2.circle
    vis = np.ones(frame.shape)

    # render virtual object on top of quad
    render_virtual_object(frame, 0, 0, hm, wm, quad)

    cv2.imshow('Interactive Systems: AR Tracking', frame)
cv2.destroyAllWindows()
