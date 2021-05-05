''' This file is 1/2 of 
    the Soccer Training Technology's developed at UCSD for 
    1. MAE 199 - Spring 2021 with prof Jack Silberman and prof Mauricio de Oliveira
    2. Triton AI - STT 

    sttDevelopment.py 
    1. checks boundary area using arucoChecker.py
    2. detect user and using homogeneity to transforms it's location.


    device: Jetson Xavier NX
    creator: Jan Dustin Tengdyantono
'''

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
import shutil
import imutils
import matplotlib as plt
from PIL import Image
# import ArUco checker
import ArucoChecker as ac 
# import user detection
from UserDetection import UserDetection
# constants
userID = "Dustin"
assessment = "test and calibration"
device = "JTX-XVR"

# video_pixels = (640,360) # the logitec and asus camera that is being used are using those resolutions
fps = 30
threshold = 0.5
c_time = 6 # calibration video record time

#warping constants
warped_pixel = 360
percentage = 100 # how many percent out of screen_pixels
grid_res = 5
# video processing
camera_path = '/dev/video0'
display = 'display://0'
# string directories
folder = './classrun1'
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
# assessments
res_to_real = 140.0 /(2*grid_res)
class STTDevelopment(object):
    def __init__(self, cal_data, cal_frames, data):
        self._cal_data = cal_data
        self._cal_frames = cal_frames
        self._data = data
        self._display = jetson.utils.videoOutput(display)
        self._mtx = None
        self._loc_usr = None
        self._hmgrphy = None
        self._img_wrp = None
        self._stopwatch = 0
        self._data_JSON = []
        self._targets = [(0,0,'center')]
        self._check = False
        self._checker_iter = 1 #the first iteration is the initial starting point
        self._check_init = False
        self._distance_covered_2p = 0 # will be updated every checkpoint checked
        self._total_distance_covered = 0  # variable to help add all checkpoint distances
        self._current_speed = 0
        self._checkpoints_data = []

    # helper functions
    
    def mapFromToX(self, x):
        a = 0.0
        b = 100.0
        c = -grid_res
        d = grid_res
        y=round((x-a)/(b-a)*(d-c)+c,2)
        return y

    def mapFromToY(self, x):
        a = 0.0
        b = 100
        c = grid_res
        d = -grid_res
        y=round((x-a)/(b-a)*(d-c)+c,2)
        return y
    
    def delete_folder(self):
        if os.path.isdir(folder):
                print("The folder " + folder + ' detected')
                print("However, area calibration is not successful.")
                print("Removing Directory.")
                shutil.rmtree(folder)
                #fix camera then re runn the function

    def bound_area_unit_calibration(self):
        '''
            changing 360px to percentage
            then reflected to -grid_red to grid_res in both axises
        '''
        self._x_warped = self._x_warped * percentage / warped_pixel
        self._y_warped = self._y_warped * percentage / warped_pixel
        self._x_warped = self.mapFromToX(self._x_warped)
        self._y_warped = self.mapFromToY(self._y_warped)


    def camera_check(self):
        # Create an object to read from camera
        video = cv2.VideoCapture(camera_path)
        # We need to check if camera is opened previously or not
        if (video.isOpened() == False): 
            print("Error reading video file")
        # We need to set resolutions. so, convert them from float to integer.
        # frame_width = int(video.get(3))
        # frame_height = int(video.get(4))
        # size = (frame_width, frame_height)
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # result = cv2.VideoWriter('display://0', fourcc , fps , size)
        print(" the camera can't see every aruco markers, please adjust the camera")
        print(" if camera is in position, press q to continue ")
        while(video.isOpened):
            ret, frame = video.read()
            if ret == True:
                # result.write(frame)
                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    def begin_assessments(self):
            os.system('clear')
            ready = input("press 1 to start \n")
            if ready == '1' or ready == 1:
                targets = self._targets
                trgt = targets[0]
                print('Starting Point: ' + str(trgt[2]))
                time.sleep(1)
                print('Count Down: 3')
                time.sleep(1)
                print('Count Down: 2')
                time.sleep(1)
                print('Count Down: 1')
                time.sleep(1)
                print('Start!')
            else:
                print('No other button represent anything, please press S to continue to assessment')
                self.begin_assessments()

    def distance_two_points(self):
        i = self._checker_iter
        delta_x = self._targets[i][0] - self._targets[i-1][0]
        delta_y = self._targets[i][1] - self._targets[i-1][1]
        dist = round(np.sqrt(abs(delta_x**2 + delta_y**2))*res_to_real ,2)
        self._distance_covered_2p = dist
        self._total_distance_covered += dist        

    def speed_check(self):
        self._current_speed = round(self._total_distance_covered / self._stopwatch,2)

    def threshold_loc(self):
        userx = np.absolute(self._x_warped)
        usery = np.absolute(self._y_warped)
        trgt = self._targets[self._checker_iter]
        trgtx = np.absolute(trgt[0])
        trgty = np.absolute(trgt[1])
        # check
        dx = userx - trgtx
        dy = usery - trgty
        dxy = np.sqrt(dx**2 + dy**2)
        if dxy <= 1:
            self._check = True
        else:
            self._check = False

    # functions
    def calibration_video(self):
        ''' 
        take camera input and save a video as a mp4 file
        Args:
            input_URI - camera path
            output_URI - name of the file to store it.
        '''
        if not os.path.isdir(folder):
            print("folder " + folder + ' is not detected')
            print("Creating folder")
            os.mkdir(folder)
        # Create an object to read from camera
        video = cv2.VideoCapture(camera_path)
        # We need to check if camera is opened previously or not
        if (video.isOpened() == False): 
            print("Error reading video file")
        # We need to set resolutions. so, convert them from float to integer.
        frame_width = int(video.get(3))
        frame_height = int(video.get(4))
        size = (frame_width, frame_height)
        print('camera pixel size : ' + str(size))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        result = cv2.VideoWriter(calibration_URI, fourcc , fps , size)
        

        # create start time and frame counter
        time_start = time.time()
        time_now = time_start
        fcounter = 0
        while(video.isOpened):
            ret, frame = video.read()
            
            time_now= time.time()
            if ret == True: 
                #flip frame y axis = 1, x axis = 0 , -1 for flipping axis.
                # frame = cv2.flip(frame,1) # can't use this because we want to check the markers
                #write frame
                result.write(frame)
                self._cal_frames.append(frame)
                fcounter += 1
                # Display the frame saved in the file
                cv2.imshow('Frame', frame)
                delta_t = time_now - time_start
                # break after 
                if delta_t > c_time and cv2.waitKey(1):
                    break 
            # Break the loop
            else:
                break
        # check total time for all frames
        time_end = time_now
        video_length = round(time_end-time_start,2)
        # print(frame_width)
        # print(frame_height)
        # print('total frames = ' + str(fcounter))
        # print('video duration = ' + str(video_length))
        # When everything done, release the video capture and video write objects
        video.release()
        result.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        
        print("The video was successfully saved")
        return fcounter, video_length	

    def area_bound_calibration(self):
        ''' 
            checking area bound by checking ArUco markers
            mtx - camera matrix model: logitec


        '''
        mtx = np.mat([[1.35007461e+03, 0.00000000e+00, 9.64381787e+02],[0.00000000e+00, 1.34859409e+03, 6.10803328e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        dist = np.mat([ 0.10485946, -0.24625137, 0.00903166, -0.00257263, -0.09894589])
        side_length = 19 # in millimeters

        fcounter, video_length = self.calibration_video()
        # print fcounter
        # print video_length

        cal = ac.ArucoChecker(mtx,dist,side_length)
        abc = cal.area_calibration_run(self._cal_frames)
        mtx,hmgrphy,img_wrp, success = abc
        # print('check point')
        if success:
            self._mtx = mtx
            # self._loc_usr = loc_usr
            self._hmgrphy = hmgrphy
            self._img_wrp = img_wrp
            # print (self._mtx)
        if not success:
            self.camera_check()
            
            self.area_bound_calibration()
    def warping(self,data):
        '''
            changing every iteration!
            ~ data is saved using data_append_json
        
        '''
        loc_3d_x = data["center"]["x"]
        loc_3d_y = data["center"]["y"]
        self._user_warped_location = np.matmul(self._mtx, np.array([[loc_3d_x],[loc_3d_y],[1]]))
        self._user_warped_location /= self._user_warped_location[2]
        self._x_warped = round(self._user_warped_location[0],2)
        self._y_warped = round(self._user_warped_location[1],2)

    def geospatial_plotting(self):
        '''
            changing every iteration!
            if data is updated.
        '''
        plt.plot(self._targets[0], self._targets[1],'ro')
        plt.axis([-11, 11, -11, 11])
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        plt.show()

    def data_append_json(self):
        '''
            changing every iteration!
            - adding data to a dictionary, appended to a list
        '''
        data = (
            {
                "time": self._stopwatch,
                "center": {
                    "x" : self._x_warped,
                    "y" : self._y_warped
                },
                "check": self._check,
                "check_iter" : self._checker_iter

                # "leftBot": {
                #     "x" : left,
                #     "y" : bot
                # },
                # "rightTop": {
                #     "x" : right,
                #     "y" : top
                # },

            } )  
                
        data_dic = {
                "userID" : userID,
                "assessment": assessment,
                "device": device,
                "data": data
            }

        self._data_JSON.append(data_dic)
    
    def assessment_RandomN(self, number = 8):
        coefficient = percentage/grid_res
        x = 0
        y = 0
        self._targets = []
        text = 'Center'
        self._targets.append((0,0,text))

        while (len(self._targets) != number): 
            x = np.random.random_integers(0, coefficient)*grid_res
            xgame = self.mapFromToX(x)
            y = np.random.random_integers(0, coefficient)*grid_res
            ygame = self.mapFromToY(y)
            text = '('+ str(xgame) + str(ygame) + ')'
            print(xgame,ygame)
            
            doubled = False
            for i in range(len(self._targets)):
                if (xgame == self._targets[i][0]) and (ygame == self._targets[i][1]):
                        doubled = True
                if doubled == False:
                    text = '(' + str(xgame) + ',' + str(ygame) +')'
                    self._targets.append((xgame,ygame,text))

        # print (self._targets)
        # self._target_circle = plt.Circle((x,y), 1.5 , color='r',fill=False)
        # #adding circle around the marker.
        # ax.add_patch(cir)
    def assessment_FivePoint(self, number = 5):
        self._targets = []
        center = 'Center'
        self._targets.append((0,0,center))

        for i in range(number):
            rand = np.random.random_integers(1, 4)
            if rand == 1:
                xgame = -grid_res
                ygame = grid_res
                text = 'Top - Left'
            if rand == 2:
                xgame = grid_res
                ygame = grid_res
                text = 'Top - Right'
            if rand == 3:
                xgame = -grid_res
                ygame = -grid_res
                text = 'Bottom - Left'
            if rand == 4:
                xgame = grid_res
                ygame = -grid_res
                text = 'Bottom - Right'
            self._targets.append((xgame,ygame,text))
            
            self._targets.append((0,0,center))
    
    def checker_algorithm(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        targets = self._targets
        length = len(targets) 

        trgt = targets[self._checker_iter]
        self.threshold_loc()
        if (self._check):
            print('Great Job!')
            self.distance_two_points()
            self.speed_check()
            data = (self._stopwatch, self._total_distance_covered, self._current_speed)
            self._checkpoints_data.append(data)

            print('time : ' + str(self._stopwatch) + ' s')
            print('distance : ' + str(self._total_distance_covered) + 'cm')
            print('speed : ' + str(self._current_speed) + 'cm/s')
            self._check = False
            self._checker_iter += 1

        else:
            print('Target : ' + trgt[2])
            print('Your Loc : (' + str(self._x_warped) + ',' + str(self._y_warped) + ')')

        if self._checker_iter == length:
            self._finish = True
            os.system('clear')
            print('time : ' + str(self._stopwatch) + ' s')
            print('distance : ' + str(self._total_distance_covered) + 'cm')
            print('speed : ' + str(self._current_speed) + 'cm/s')
    
    def menu_pick(self):
        assessment_id = input("Please choose one; \n1. Five Points \n2. Random N \n")
        if assessment_id == 1:
            self.assessment_FivePoint()
        if assessment_id == 2:
            self.assessment_RandomN()

    def stt_main(self):
        # self._input = jetson.utils.videoSource(camera_path)
        # self._display = jetson.utils.videoOutput(display)
        self._finish = False
        # self._net = jetson.inference.detectNet(network, threshold)
        # img = self._input.Capture()
        # print the detections
        # print("detected {:d} objects in image".format(len(detections)))

        # initializing user detection dependencies
        ud = UserDetection()
        # starting time count
        os.system('cls' if os.name == 'nt' else 'clear')
        # menu
        self.menu_pick()
        # count down
        self.begin_assessments()
        time_start = time.time()
        while (self._finish == False):
            self._stopwatch = round(time.time() - time_start, 2)
            success, data, img = ud.run_detection(self._stopwatch) 
            if success == True:
                time_error = time.time()
                # adding warp data as global
                stt.warping(data)
                # calibrate to 0-100 then to -10 to 10
                stt.bound_area_unit_calibration()
                # algorithm kick in
                stt.checker_algorithm()
                # saving data to JSON
                stt.data_append_json()
                # plotting
                # stt.geospatial_plotting()
                # display image ##color isn't right
                img = jetson.utils.cudaToNumpy(img)
                cv2.imshow("live",img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print(' there is no user bounded')
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self._check = False
        
            
            

if __name__ == '__main__':
    cal_data = []
    cal_frames = []
    data = []
    stt = STTDevelopment(cal_data, cal_frames, data)
    stt.delete_folder()
    # check if calibration video exists
    if not os.path.isfile(calibration_URI):
        stt.area_bound_calibration()
        stt.stt_main()

    # else:
    #     print(' please modify folder location or delete current folder')
    # stt.algorithm_random_number(5)
