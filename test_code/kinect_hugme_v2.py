
#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import numpy as np
import math, time, io, sys
from collections import deque
import gpiozero

# GLOBAL VARIABLES
kernel = np.ones((3,3), np.uint8)
kernel_big = np.ones((9,9), np.uint8)
backsub = cv2.createBackgroundSubtractorKNN()
backSub_depth = cv2.createBackgroundSubtractorKNN(history=100,dist2Threshold=400.0, detectShadows=True)
pre_distances = deque([10000,10000,10000,10000]) # stores previous distances of the two biggest blobs to recognize valid movement
BLOB_MAX_SIZE = 40000
BLOB_MIN_SIZE = 500
IMG_DEPTH = 0
IMG_RGB = 1
THRESHOLD = 152
DEPTH = 814
TIME_BETWEEN_FRAMES = .3 # good values for testing .3 (fast), 1 (slow)
RElAY_PIN = 21
relay = gpiozero.OutputDevice(RElAY_PIN, active_high=False, initial_value=False)

# CLASSES
class Blob:
    def __init__(self, x, y, width, height, size):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.size = size

    def x_center(self):
        return round(self.x + self.width / 2)
 
    def y_center(self):
        return round(self.y + self.height / 2)
 
    def distanceTo(self, otherBlob):
        #print("x_center=%d, y_center=%d" % (self.x_center(), self.y_center()))
        if self.width == 0 or self.height == 0 or otherBlob.width == 0 or otherBlob.height == 0:
            return 10000 # if not two valid blobs, return an invalid distance
        return round(math.sqrt(math.pow(self.x_center() - otherBlob.x_center(),2) + math.pow(self.y_center() - otherBlob.y_center(),2)))
 
    @classmethod
    def getBlobs(cls, labels, stats):
        blobs = [0,0]
        blobs[0] = Blob(0,0,0,0,0)
        blobs[1] = Blob(0,0,0,0,0)
        i = 1
    
        for label in range(0, np.max(labels)): # ignore first label, which is the background
            size = stats[label, cv2.CC_STAT_AREA]
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            blob = Blob(x, y, width, height, size)
            
            # Debug output for all captured blobs:
            print("Blob %3d: %3d,%3d %dx%d size=%d" % (i, blob.x, blob.y, blob.width, blob.height, blob.size))
            
            # Get biggest two blobs in the allowed pixel size range:
            if BLOB_MAX_SIZE > blob.size > BLOB_MIN_SIZE:
                if blobs[0].size < blob.size and blobs[1].size < blob.size:
                    if blobs[0].size <= blobs[1].size:
                        blobs[0] = blob
                    else:
                        blobs[1] = blob
            i += 1

        #Output the two biggest blobs:
        #for b in blobs:
        #    print("Blob: %3d,%3d %dx%d size=%d" % (b.x, b.y, b.width, b.height, b.size))
            
        return blobs

# METHODS
def get_img(mode):
    # This was intended to inhibit the stream warnings to stdout, but did not work.
    #text_trap = io.StringIO()
    #sys.stderr = text_trap
    if (mode == IMG_RGB):
        frame = freenect.sync_get_video()[0] # gets the Kinect RGB image
        fgMask = backSub.apply(frame, learningRate=-1)
        ret, fgMask = cv2.threshold(fgMask,127,255,cv2.THRESH_BINARY)
        #fgMask = cv2.erode(fgMask, kernel, iterations = 1) # morphological erode with 3x3
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_big) # closes gaps small than 9x9 pixels 
    elif (mode == IMG_DEPTH):
        frame = freenect.sync_get_depth()[0] # gets the Kinect depth image
        frame = 255 * np.logical_and(frame >= DEPTH - THRESHOLD,
                                 frame <= DEPTH + THRESHOLD)
        frame = frame.astype(np.uint8)
        fgMask = backSub_depth.apply(frame, learningRate=-1)
        ret, fgMask = cv2.threshold(fgMask,127,255,cv2.THRESH_BINARY)
        fgMask = cv2.erode(fgMask, kernel, iterations = 1) # morphological erode with 3x3
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_big) # closes gaps small than 9x9 pixels 

    # Problem: this function gives us sometimes only one blob instead of two
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(fgMask)
    # This didn't seem to work:
    #for label in range(np.max(labels)):
    #    if 2000 < stats[label, cv2.CC_STAT_AREA] < 900:
    #       labels[labels==label]=0
    
    # Reset output to stdout:
    #sys.stderr = sys.__stderr__
    return ret, frame, fgMask, labels, stats

def imshow_components(labels):
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img [label_hue==0] = 0
    return labeled_img

def checkHugEvent(blobs):
    global pre_distances
    distance = blobs[0].distanceTo(blobs[1])
    print
    print(pre_distances)
    print("Distance: %d" % (distance))
    #for i in range(1,2):
    #    if (pre_distances[i] == 10000 and pre_distances[i-1] > pre_distances[i+1]:
    #        pre_distances[i] = round((pre_distances[i-1] - pre_distances[i+1]) / 2)
    # When only one blob has been recognized, check if previous distances fit the proper motion:
    if distance == 10000 and pre_distances[0] > pre_distances[1] > pre_distances[2] > pre_distances[3]:
        # time.sleep(1) # maybe wait 1 second,
                        # because a blob is already detected when crossing arms
        print("Light on")
        relay.on() # here the relay will be turned on
        time.sleep(5)
        relay.off()
        # Digital IO Pin aus
        dummy = raw_input("Press key for next loop...") # Warten auf Tastatur, muss im Realbetrieb aus.
        pre_distances = deque([10000, 10000, 10000, 10000]) # reset previous distances
    else:
        print("no event")
        pre_distances.append(distance)
        pre_distances.popleft()
        print(pre_distances)

# INIT
# Activate windows only for debug: 
#cv2.namedWindow('Video')
cv2.namedWindow('FGMask')
#cv2.namedWindow('Labels')
print('Press ESC in window to stop')

def show_video():
    cv2.imshow('Video', frame_convert2.video_cv(freenect.sync_get_video()[0]))

# It seems the first captures are wrong and/or the library needs some time
# to initialize the background image, so some images are skipped at start:
for num in range(1,10):
    #get_img(IMG_DEPTH)
    get_img(IMG_RGB)
# LOOP
while 1:
    #relay.toggle() #test relay
    time.sleep(1)
    #ret, frame, fgMask, labels, stats = get_img(IMG_RGB) # switch back to IMG_RGB
    ret, frame, fgMask, labels, stats = get_img(IMG_DEPTH)
    print("\033[H\033[J") # clear screen
    blobs = Blob.getBlobs(labels, stats)
    # If no blob was found, try depth image:
    #if blobs[0].width == 0 and blobs[0].height == 0 and blobs[1].width == 0 and blobs[1].height == 0:
    #ret, frame, fgMask, labels, stats = get_img(IMG_DEPTH)
    #    blobs = Blob.getBlobs(labels, stats)

    checkHugEvent(blobs) 
    time.sleep(TIME_BETWEEN_FRAMES)
#   dummy = raw_input("Press key for next loop...")
    labeled_img = imshow_components(fgMask)
    #cv2.imshow('Labels', labeled_img)
    cv2.imshow('FGMask', fgMask)
    #show_video()
 #  cv2.imshow('Video', frame_convert2.video_cv(frame))
    if cv2.waitKey(10) == 27:
        break

