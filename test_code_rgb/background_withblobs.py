
#import time
import datetime
import argparse
from collections import deque
import sys
import numpy as np
import cv2
import imutils

now = datetime.datetime.now
cap = cv2.VideoCapture(0)
blobsNotFound = []

cv2.namedWindow('Original')
cv2.namedWindow('Mask')
cv2.namedWindow('Ende')

print("Your OpenCV version: {}".format(cv2.__version__))

#fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
fgbg = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=50, detectShadows=False)

kernel = np.ones((3,3), np.uint8)
kernel_big = np.ones((9,9), np.uint8)


while True:

    _, frame = cap.read()

    resized = imutils.resize(frame, 600)


    #convert to grayscale

    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


    #Blur image to remove noise

    blur = cv2.GaussianBlur(resized, (3, 3), 0)



    #CLAHE (Contrast Limited Adaptive Histogram Equalization)

    #threshed = cv2.threshold(resized,127,255,cv2.THRESH_BINARY)


    fgMask = fgbg.apply(blur)

    eroded = cv2.erode(fgMask, kernel, iterations=2)

    dilated = cv2.dilate(eroded, kernel, iterations=1)

    ex = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_big)

    #Farbbild zur Debug-Ausgabe
    col = cv2.cvtColor(ex, cv2.COLOR_GRAY2BGR)



    cv2.putText(col, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (5, 329), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255),1)

    cv2.putText(col,"hugme RGB", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255)

    params = cv2.SimpleBlobDetector_Params()

   # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 500000
     
    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1
     
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5
     
    # Filter by Inertia
    params.filterByInertia =False

    params.minInertiaRatio = 0.5

    #filter by colour
    params.filterByColor=False
    params.blobColor=255
     
    detector = cv2.SimpleBlobDetector_create(params)
 
    # Detect blobs.
    reversemask=255-fgMask
    keypoints = detector.detect(reversemask)

    if keypoints:
        print ("found %d blobs" % len(keypoints))
        if len(keypoints) > 4:
            # if more than four blobs, keep the four largest
            keypoints.sort(key=(lambda s: s.size))
            keypoints=keypoints[0:3]
    else:
        print ("no blobs")

    im_with_keypoints = cv2.drawKeypoints(col, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #show original Kamera

    cv2.imshow('Original', resized)


    #show Maske

    cv2.imshow('Mask', fgMask)

    #show Endresultat

    cv2.imshow('Ende',im_with_keypoints )


    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
