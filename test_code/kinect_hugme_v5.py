#!/usr/bin/env python
import freenect
import datetime
import cv2
import frame_convert2
import numpy as np
import math, time, datetime, io, sys
from collections import deque
import gpiozero

### GLOBAL OBJECTS
KERNEL = np.ones((3,3), np.uint8)
KERNEL_BIG = np.ones((9,9), np.uint8)
# Create two independent background substractors, because RGB and depth image might need different parameters:
# NOTE: ADAPT THE BG SUBSTRACTOR PARAMETERS ON THE LOCATION YOU SET UP THE KINECT TO GET BEST RECOGNITION:
backSubDepth = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=0)
backSubRgb = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=50, detectShadows=0)

### GLOBAL CONSTANTS
CACHE_SIZE = 4 # size of the list that stores previous distance values, must be 4 or greater
if CACHE_SIZE < 4: CACHE_SIZE = 4
BLOB_MAX_SIZE = 40000
BLOB_MIN_SIZE = 3000
IMG_DEPTH = 0
IMG_RGB = 1
THRESHOLD = 814
DEPTH = 152
TIME_BETWEEN_FRAMES = 0.1 # good values for testing .3 (fast), 1 (slow)
RELAY_PIN = 21

### GLOBAL VARIABLES
pre_distances = deque([10000] * CACHE_SIZE) # stores previous distances of the two biggest blobs to recognize valid movement
usedRecognitionMode = 0

### CLASSES
class Blob:
    def __init__(self, x=0, y=0, width=0, height=0, size=0, x_center=0, y_center=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size = size
        self.x_center = x_center
        self.y_center = y_center

    def distanceTo(self, otherBlob):
        #print("x_center=%d, y_center=%d" % (self.x_center(), self.y_center()))
        if self.width == 0 or self.height == 0 or otherBlob.width == 0 or otherBlob.height == 0:
            return 10000 # if not two valid blobs, return an invalid distance
        return round(math.sqrt(math.pow(self.x_center - otherBlob.x_center,2) + math.pow(self.y_center - otherBlob.y_center,2)))
 
    @classmethod
    def getBlobs(cls, labels, stats, centroids):
        blobs = [0] * 2
        blobs[0] = Blob()
        blobs[1] = Blob()
        i = 1
        
        # Note: One of the labels is the "background" label which has the size of the whole image
        for label in range(0, np.max(labels)):
            size = stats[label, cv2.CC_STAT_AREA]
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            blob = Blob(x, y, width, height, size, round(centroids[label][0]), round(centroids[label][1]))
            
            # Debug output for all captured blobs. Ignores mini blobs of a size less than 100 pixels:
            if size > 100:
                print("Blob %3d: %3d,%3d %dx%d size=%d" % (i, blob.x_center, blob.y_center, blob.width, blob.height, blob.size))

            # Stores the biggest two blobs of the accepted pixel size range into a list "blobs":
            if BLOB_MIN_SIZE < blob.size < BLOB_MAX_SIZE:
                if blobs[0].size < blob.size or blobs[1].size < blob.size:
                    if blobs[0].size <= blobs[1].size:
                        blobs[0] = blob
                    else:
                        blobs[1] = blob
            i += 1

        #Output the two biggest blobs:
        print
        for b in blobs:
            print("Blob: %3d,%3d %dx%d size=%d" % (b.x, b.y, b.width, b.height, b.size))
            
        return blobs

### METHODS
def get_img(mode, showRaw=0):
    # This was intended to inhibit the stream warnings to the console, but it did not work:
    #text_trap = io.StringIO()
    #sys.stderr = text_trap
    if (mode == IMG_RGB):
        frame = freenect.sync_get_video()[0] # gets the Kinect RGB image
        fgMask = backSubRgb.apply(frame, learningRate=-1)
        ret, fgMask = cv2.threshold(fgMask,127,255,cv2.THRESH_BINARY)
        #fgMask = cv2.erode(fgMask, KERNEL, iterations = 1) # morphological erode with 3x3
        #cv2.imshow('FGMaskRaw', fgMaskRaw)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, KERNEL_BIG) # closes gaps smaller than 9x9 pixels 
    elif (mode == IMG_DEPTH):
        frame = freenect.sync_get_depth()[0] # gets the Kinect depth image
        fgMask = 255 * np.logical_and(frame >= DEPTH - THRESHOLD,
                                 frame <= DEPTH + THRESHOLD)
        fgMask = fgMask.astype(np.uint8)
        fgMask = backSubDepth.apply(fgMask, learningRate=-1)
        # It looks like these operations are not needed for the depth image:
        #ret, fgMask = cv2.threshold(fgMask,127,255,cv2.THRESH_BINARY)
        #fgMask = cv2.erode(fgMask, KERNEL, iterations = 1) # morphological erode with 3x3
        #fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, KERNEL_BIG) # closes gaps smaller than 9x9 pixels 

    if showRaw == 1: cv2.imshow('Raw', frame)
    # Problem: this function gives us sometimes only one blob instead of two
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(fgMask)
    
    # Reset output to stdout:
    #sys.stderr = sys.__stderr__
    return ret, frame, fgMask, labels, stats, centroids

def checkHugEvent(blobs):
    global pre_distances
    distance = blobs[0].distanceTo(blobs[1])
    valid = 1
    print
    print(pre_distances)
    print("Distance: %d" % (distance))

    # If an unrecognized value (distance = 10000) is surrounded by two valid distances, interpolate the wrong value.
    # This is a very simple error correction when just one of two valid blobs got recognized.
    #for i in range(1, CACHE_SIZE - 1): 
    #    if pre_distances[i] == 10000 and pre_distances[i-1] < 10000 and pre_distances[i-1] > pre_distances[i+1]:
    #        pre_distances[i] = round(pre_distances[i-1] + pre_distances[i+1] / 2) # interpolate unrecognized distance

    # When only one blob has been recognized this time, check if previous distances fit the proper motion:
    if distance == 10000 and pre_distances[CACHE_SIZE - 1] < 150:
        for i in range(0, CACHE_SIZE - 1):
            print("pre %d: %d, pre %d: %d" % (i, pre_distances[i], i+1, pre_distances[i + 1]))
            if pre_distances[i] <= pre_distances[i + 1]:
                valid = 0
                break
    else:
        print("NO EVENT")
        print(pre_distances)
        valid = 0

    pre_distances.append(distance)
    pre_distances.popleft()

    if valid == 1:
        # time.sleep(1) # maybe wait 1 second,
                        # because a blob is already detected when arms cross over
        print("Light on")
        #relay.on() # here the relay will be turned on
        #time.sleep(5)
        #relay.off()

        dummy = raw_input("Press key for next loop...") # Warten auf Tastatur, muss im Realbetrieb aus.
        pre_distances = deque([10000] * CACHE_SIZE) # reset previous distances

### INITIALIZATION
# Activate windows only for debug: 
cv2.namedWindow('Raw')
cv2.namedWindow('FGMask')
#cv2.namedWindow('FGMaskRaw')
#relay = gpiozero.OutputDevice(RELAY_PIN, active_high=False, initial_value=False)
print('Press ESC in window to stop')

# It seems the first captures are wrong and/or the library needs some time
# to initialize the background image, so some images are skipped at start:
for num in range(1,10):
    get_img(IMG_DEPTH)
    #get_img(IMG_RGB)

### LOOP
while 1:
    #relay.toggle() #test relay
    usedRecognitionMode = IMG_DEPTH
    ret, frame, fgMask, labels, stats, centroids = get_img(IMG_DEPTH, showRaw=1)
    print("\033[H\033[J") # clear screen
    blobs = Blob.getBlobs(labels, stats, centroids)
    # If no blob was found, try RGB image:
    if blobs[0].width == 0 and blobs[0].height == 0 and blobs[1].width == 0 and blobs[1].height == 0:
        usedRecognitionMode = IMG_RGB
        ret, frame, fgMask, labels, stats, centroids = get_img(IMG_RGB, showRaw=1)
        blobs = Blob.getBlobs(labels, stats, centroids)

    checkHugEvent(blobs)
    #time.sleep(TIME_BETWEEN_FRAMES)
    #dummy = raw_input("Press key for next loop...")
    col = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
    cv2.putText(col,("RGB" if usedRecognitionMode == IMG_RGB else "DEPTH"), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 255)
    cv2.putText(col, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (5, 429), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255),1)
    for blob in blobs:
        cv2.circle(col, (int(blob.x_center), int(blob.y_center)), int(blob.width/2), (0, 255, 255), 2)
    cv2.imshow('FGMask', col)
    if cv2.waitKey(30) & 0xff == 27:
        break

