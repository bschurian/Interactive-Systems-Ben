import numpy as np
import cv2

cap = cv2.VideoCapture(0)
mode = 0
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # wait for key and switch to mode
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord('1'):
        mode = 1
    if ch == ord('2'):
        mode = 2
    if ch == ord('3'):
        mode = 3
    if ch == ord('4'):
        mode = 4
    # ...

    if ch == ord('q'):
        break

    if mode == 1:
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
    if mode == 2:
        # @TODO
        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if mode == 4:
        frame = cv2.Canny(frame, 100, 200)

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
