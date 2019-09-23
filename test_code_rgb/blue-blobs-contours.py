
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

cv2.namedWindow('Original')
#cv2.namedWindow('Blur')
cv2.namedWindow('HSV')
#cv2.namedWindow('Mask')
#cv2.namedWindow('Res')
cv2.namedWindow('Final')

print("Your OpenCV version: {}".format(cv2.__version__))

kernel = np.ones((3,3), np.uint8)
kernel_big = np.ones((9,9), np.uint8)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())



# define range of blue color in HSV
lower_blue = np.array([90,50,50])
upper_blue = np.array([130,255,255])
    
pts = deque(maxlen=args["buffer"])



                      
while True:

    _, frame = cap.read()

    resized = imutils.resize(frame, 600)

                      
    #Blur image to remove noise

    blur = cv2.GaussianBlur(resized, (3, 3), 0)

    
    # Switch image from BGR colorspace to HSV which is better for color detection
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    

    # define range of blue color in HSV
    #lower_yellow = np.array([80,100,100])
    #upper_yellow = np.array([100,255,255])


    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)


    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(blur,blur, mask=mask)


    eroded = cv2.erode(mask, kernel, iterations=2)

    dilated = cv2.dilate(eroded, kernel, iterations=1)
    
    #ex = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_big)

    #Farbbild zur Debug-Ausgabe
    col = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)



    cv2.putText(col, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (5, 329), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255),1)

    cv2.putText(col,"color tracking", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255)
    
    
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None


    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


        # only proceed if the radius meets a minimum size
        if radius > 10:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(col, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(col, center, 5, (0, 0, 255), -1)

    # update the points queue
    pts.appendleft(center)


    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
 
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(col, pts[i - 1], pts[i], (0, 0, 255), thickness)

    #show original Kamera
    
    cv2.imshow('Original', resized)

    #show HSV

    cv2.imshow('HSV', hsv)

    #show Blur

    #cv2.imshow('Blur', blur)
    

    #show Maske

    #cv2.imshow('Mask', mask)

    #res

    cv2.imshow('Res', res)


    #show Keypoints

    cv2.imshow('Final', col)



    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
