
#import time
import datetime
import argparse
from collections import deque
import sys
import numpy as np
import cv2
import imutils
import math

now = datetime.datetime.now
cap = cv2.VideoCapture(0)
blobsNotFound = []

cv2.namedWindow('Original')
#cv2.namedWindow('Blur')
cv2.namedWindow('HSV')
#cv2.namedWindow('Mask')
#cv2.namedWindow('Res')
cv2.namedWindow('Final')

print("Your OpenCV version: {}".format(cv2.__version__))

kernel = np.ones((3,3), np.uint8)
kernel_big = np.ones((9,9), np.uint8)

                      
while True:

    _, frame = cap.read()

    resized = imutils.resize(frame, 600)

                      
    #Blur image to remove noise

    blur = cv2.GaussianBlur(resized, (3, 3), 0)

    
    # Switch image from BGR colorspace to HSV which is better for color detection
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    


    # define range of blue color in HSV
    lower_blue = np.array([100,50,50])
    upper_blue = np.array([130,255,255])
    

    # define range of blue color in HSV
    #lower_yellow = np.array([80,100,100])
    #upper_yellow = np.array([100,255,255])


    # Threshold the HSV image to get only yellow colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)


    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(blur,blur, mask=mask)


    #eroded = cv2.erode(mask, kernel, iterations=2)

    dilated = cv2.dilate(mask, kernel, iterations=1)
    
    ex = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_big)

    #Farbbild zur Debug-Ausgabe
    col = cv2.cvtColor(ex, cv2.COLOR_GRAY2BGR)



    cv2.putText(col, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (5, 329), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255),1)

    cv2.putText(col,"color tracking", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255)


    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
     
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 10000
     
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
    reversemask=255-mask
    keypoints = detector.detect(reversemask)


    if keypoints:
        print ("found %d blobs" % len(keypoints))
        if len(keypoints) > 4:
              # if more than four blobs, keep the four largest
            keypoints.sort(key=(lambda s: s.size))
            keypoints=keypoints[0:3]
        else:
            print ("no blobs")


    # Draw green circles around detected blobs
    im_with_keypoints = cv2.drawKeypoints(col, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #im_with_keypoints = cv2.circle(col, (x, y), r, (255, 255, 0), 2)

    #show original Kamera
    
    cv2.imshow('Original', resized)

    #show Gray

    cv2.imshow('HSV', hsv)

    #show Blur

    #cv2.imshow('Blur', blur)
    

    #show Maske

    #cv2.imshow('Mask', mask)

    #res

    cv2.imshow('Res', res)


    #show Keypoints

    cv2.imshow('Final', im_with_keypoints)



    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
