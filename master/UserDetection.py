import cv2
import jetson.inference
import jetson.utils
import time
import numpy as np
from PIL import Image

# constants
overlay = "box,labels,conf"
fps = 30
camera_path = '/dev/video0'
display = 'display://0'
network = "ssd-mobilenet-v2"
threshold = 0.5

class UserDetection(object):
    def __init__(self):
        
        self._input = jetson.utils.videoSource(camera_path, ['--input-width= 640', '--input-height = 480'])
        
        self._finish = False
        self._net = jetson.inference.detectNet(network, threshold)


        # frame_width = int(self._input.get(3))
        # frame_height = int(self._input.get(4))
        # self._size = (frame_width, frame_height)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # result = cv2.VideoWriter(calibration_URI, fourcc , fps , self._size) 

    def run_detection(self, stopwatch):
        # print the detections
        # print("detected {:d} objects in image".format(len(detections)))
        img = self._input.Capture() #save as cuda image capsule - 
        # frame_width = int(img.get(3))
        # frame_height = int(img.get(4))
        # size = (frame_width, frame_height)
        # print('camera pixel size : ' + str(size))
        detections = self._net.Detect(img)
        
        data = []
        person_bool = False
        for detection in detections:
            if detection.ClassID == 1 and person_bool == False:
                left = int(detection.Left)
                right = int(detection.Right)
                top = int(detection.Top)
                bot = int(detection.Bottom)
                ctr = int(detection.Center[0])
                # getting actual location of the user
                magicnumber = 5.0
                centerx = ctr
                centery = bot - (abs(right - left)/magicnumber)
                center = (centerx,centery)
                area = int(detection.Area)

                # if len(time_arr) == 0:
                #     time_start = round(time.time(),2)
                #     delta_time = time_start - time_start
                #     time_arr.append(delta_time)
                # else:
                #     delta_time = round(time.time()-time_start,2)
                #     time_arr.append(delta_time)
                # print(str(time) + ',' + str(left) + ',' + str(bot) + ',' + str(right) + ',' + str(top)  )
                data = (
                        {
                            "time": stopwatch,
                            "leftBot": {
                                "x" : left,
                                "y" : bot
                            },
                            "rightTop": {
                                "x" : right,
                                "y" : top
                            },
                            "center": {
                                "x" : center[0],
                                "y" : center[1]
                            }
                            
                        }
                )
                # print ('center - 3D = ' + str(center))
                
                # print('Left-Bottom (X,Y) =  ('+ str(left) + ',' + str(bot) + ')')
                # print('Right-Bottom (X,Y) =  ('+ str(right) + ',' + str(bot) + ')')
                # print('Left-Top (X,Y) =  ('+ str(left) + ',' + str(top) + ')')
                # print('Right-Top (X,Y) =  ('+ str(right) + ',' + str(top) + ')')
                # print('Center (X,Y) = ('+ str(center[0]) + ',' + str(center[1]) + ')')
                # print('Area = ' + str(area) )
                person_bool = True
                #store values to json here
                # print(person_bool)
                # check if there is no person recognized in the frame
                success = True

        if person_bool == False:
            data = None
            success = False
        
        # render the image
        # output.Render(img)

        # update the title bar
        #output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

        # print out performance info
        # net.PrintProfilerTimes()

        #  it on input/output EOS
        # if not input.IsStreaming() or not output.IsStreaming():
        # 	break
        return success, data, img
