
#!/usr/bin/env python
import freenect
import cv2
import frame_convert2
import numpy as np

cv2.namedWindow('Video')
cv2.namedWindow('FGMask')
cv2.namedWindow('Labels')
print('Press ESC in window to stop')
kernel = np.ones((3,3), np.uint8)
kernel_big = np.ones((9,9), np.uint8)
backSub = cv2.createBackgroundSubtractorKNN()

def get_video():
    frame = freenect.sync_get_video()[0]#Oder get depth
    fgMask = backSub.apply(frame, learningRate=-1)#Das hier kann bei depth weg
    ret,fgMask = cv2.threshold(fgMask,127,255,cv2.THRESH_BINARY)
    fgMask = cv2.erode(fgMask, kernel, iterations = 1)#Morphological erode mit 3x3
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel_big)#Schliest loecher kleiner als 9x9
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(fgMask)
    for label in range(np.max(labels)):
        if 2000 < stats[label, cv2.CC_STAT_AREA] < 900:
            labels[labels==label]=0
        else:
    return frame, fgMask, labels

def imshow_components(labels):
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img [label_hue==0] = 0
    return labeled_img

while 1:
    frame, fgMask, labels = get_video()
    labeled_img = imshow_components(labels)
    cv2.imshow('Labels', labeled_img)
    cv2.imshow('FGMask', fgMask)
    cv2.imshow('Video', frame_convert2.video_cv(frame))
    if cv2.waitKey(10) == 27:
        break
