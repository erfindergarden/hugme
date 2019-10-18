#!/usr/bin/env python
import freenect
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
#kernel = np.ones((3,3), np.uint8)
#kernel_big = np.ones((9,9), np.uint8)

cv2.namedWindow('Original')
cv2.namedWindow('Normalized')

bgSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=50, detectShadows=0)

print("Your OpenCV version: {}".format(cv2.__version__))

while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Original', frame)
    normal = cv2.equalizeHist(frame)
    cv2.imshow('Normalized', normal)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
