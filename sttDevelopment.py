import cv2
import jetson.inference
import jetson.utils
import argparse
import sys
import os
import time
import json
import base64
import numpy as np
from PIL import Image
# import ArUco checker
import arucoChecker as ac 


# constants
# video_pixels = (640,360) # the logitec and asus camera that is being used are using those resolutions
# resolution_percentage = 100 # how many percent out of screen_pixels
fps = 30
threshold = 0.5
c_time = 5
# video processing
camera_path = '/dev/video0'
# string directories
folder = './beta_run1'
calibration_json = folder + '/calibration_data.json'
calibration_URI = folder + '/calibration.mp4'
video_URI = folder + '/video.mp4'
video_json = folder + '/video_data.json'
bounded_video = folder + '/boundedVid.mp4'
# detection string constants 
network = "ssd-mobilenet-v2"
overlay = "box,labels,conf"
# empty list
area_calibration_frames = []
data = []
class sttDevelopment(object):
    def __init__(self, data):
        self._data = data

    def area_bound_calibration(self, ac):



    def calibration_video(self):
        ''' 
        take camera input and save a video as a mp4 file
        Args:
            input_URI - camera path
            output_URI - name of the file to store it.
        '''
        # Create an object to read from camera
        video = cv2.VideoCapture(input_URI)
        # We need to check if camera is opened previously or not
        if (video.isOpened() == False): 
            print("Error reading video file")
        # We need to set resolutions. so, convert them from float to integer.
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        size = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result = cv2.VideoWriter(video_URI, fourcc , fps , size)
        

        # create start time and frame counter
        time_start = time.time()
        time_now = start
        fcounter = 0
        end = 0
        while(video.isOpened):
            ret, frame = video.read()
            time_now= time.time()
            if ret == True: 
                #flip frame y axis = 1, x axis = 0 , -1 for flipping axis.
                # frame = cv2.flip(frame,1) # can't use this because we want to check the markers
                #write frame
                result.write(frame)
                fcounter += 1
                # Display the frame saved in the file
                cv2.imshow('Frame', frame)
                delta_t = time_now - time_start
                # Press q on keyboard to stop the process
                if delta_t > c_time :
                    break
                    
            # Break the loop
            else:
                break
        # check total time for all frames
        time_end = time_now
        video_length = round(time_end-time_start,2)
        # print(frame_width)
        # print(frame_height)
        print('total frames = ' + str(fcounter))
        print('video duration = ' + str(video_length))
        # When everything done, release the video capture and video write objects
        video.release()
        result.release()
            
        # Closes all the frames
        cv2.destroyAllWindows()
        
        print("The video was successfully saved")
        return fcounter, video_length	