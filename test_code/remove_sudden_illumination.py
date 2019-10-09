#!/usr/bin/env python
import freenect
import cv2
import numpy as np

def makeIlluminationInvariantRGB(frame):
    result = frame
    cn = 3
    bitshift = 3;

    #for row in result.rows:
     #   for col in result.cols:
     #       result.data[row*result.cols*cn + col*cn + 0] = result.data[row*result.cols*cn + col*cn + 0] >> bitshift
     #       result.data[row*result.cols*cn + col*cn + 1] = result.data[row*result.cols*cn + col*cn + 1] >> bitshift
     #       result.data[row*result.cols*cn + col*cn + 2] = result.data[row*result.cols*cn + col*cn + 2] >> bitshift
    for val in result.data:
        val = val >> bitshift
    return result

cap = cv2.VideoCapture(0)

cv2.namedWindow('Original')
cv2.namedWindow('Result')

print("Your OpenCV version: {}".format(cv2.__version__))

#while True:
_, frame = cap.read()
print("frame: {}".format(frame.shape))
#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', frame)
result = makeIlluminationInvariantRGB(frame)
cv2.imshow('Result', result)

#k = cv2.waitKey(30) & 0xff
#if k == 27:
#    break

cap.release()
cv2.destroyAllWindows()
