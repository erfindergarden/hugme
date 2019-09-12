#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:57:45 2019

@author: Joni und Andy
"""

import numpy as np
import cv2
import freenect
import datetime
import time
import frame_convert2
from multiprocessing import Process


now = datetime.datetime.now
threshold = 800
current_depth = 160

def change_threshold(value):
    global threshold
    threshold = value
    
def change_depth(value):
    global current_depth
    current_depth = value
    
def show_depth():
    global threshold
    global current_depth

    depth, timestamp = freenect.sync_get_depth()
    depth = 255 * np.logical_and(depth >= current_depth - threshold,
                                 depth <= current_depth + threshold)
    depth = depth.astype(np.uint8)
    cv2.imshow('Depth', depth)



def show_video():
    cv2.imshow('Video', frame_convert2.video_cv(freenect.sync_get_video()[0]))
    
cv2.namedWindow('Depth')
cv2.namedWindow('Video')
cv2.createTrackbar('threshold', 'Depth', threshold,     500,  change_threshold)
cv2.createTrackbar('depth',     'Depth', current_depth, 2048, change_depth)

def process_a():
    while 1:
        time.sleep(1)
        t_n = now()
        video = freenect.sync_get_video()[0] # mit dieser Funktion wird ein Frame des Kinect videos geholt
        depth = freenect.sync_get_depth()[0] # mit dieser Funktion wird ein Frame des Tiefenbildes geholt
        depth = depth.astype(np.uint8)
        cv2.imwrite("/home/pi/kinect/video/video_"+datetime.datetime.strftime(t_n, '%H_%M_%S')+".png", video)
        cv2.imwrite("/home/pi/kinect/depth/depth_"+datetime.datetime.strftime(t_n, '%H_%M_%S')+".png", depth)
        
def process_b():
    while 1:
        show_depth()
        show_video()
        if cv2.waitKey(10) == 27:
            break
            
            
if __name__ == '__main__':
    Process(target=process_a).start()
    #Process(target=process_b).start()

