import cv2

cap = cv2.VideoCapture(0)
# cv2.namedWindow('Interactive Systems: Towards AR Tracking')
while True:

    # 1. read each frame from the camera (if necessary resize the image)
    #    and extract the SIFT features using OpenCV methods
    #    Note: use the gray image - so you need to convert the image
    # 2. draw the keypoints using cv2.drawKeypoints
    #    There are several flags for visualization - e.g. DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

    # close the window and application by pressing a key

    # code taken from: https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray, None)
    outImage = None
    # img = cv2.drawKeypoints(gray, kp, outImage)
    img = cv2.drawKeypoints(gray, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, outImage=img)
    cv2.imshow('stream', img)


    # termination criteria
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('q'):
        break


cv2.destroyAllWindows()
