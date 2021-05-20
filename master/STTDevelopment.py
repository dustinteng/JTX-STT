''' This file is 1/3 of 
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
threshold = 20
c_time = 6 # calibration video record time

#warping constants
warped_pixel = 360.0
percentage = 100 # how many percent out of screen_pixels
# video processing
camera_path = '/dev/video0'
display = 'display://0'
# string directories
folder = './data'
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
side_length = 140
# assessment strings:
five_point = '/FivePoint'
randomN = '/RandomN'

class STTDevelopment(object):
    def __init__(self):
        self._data = []
        self._display = jetson.utils.videoOutput(display)
        self._trf_mtx = None
        self._stopwatch = 0
        self._data_JSON = []
        self._targets = [(0,0,'center')]
        self._check = False
        self._checker_iter = 1 #the first iteration is the initial starting point
        self._distance_covered_2p = 0 # will be updated every checkpoint checked
        self._total_distance_covered = 0  # variable to help add all checkpoint distances
        self._current_speed = 0
        self._checkpoints_data = []
        self._userID = str
        self._assessment = None
        self._ass_no = 1
        self._cur_dir = '/home/user/projects/JTX-STT/user_data'

    # assessments
    def assessment_RandomN(self, number = 8):
        coefficient = side_length/2
        self._targets = []
        text = 'Center'
        self._targets.append((0,0,text))

        while (len(self._targets) != number): 
            xgame = np.random.random_integers(-coefficient, coefficient)
            ygame = np.random.random_integers(-coefficient, coefficient)
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
        coefficient = side_length/2
        self._targets = []
        center = 'Center'
        self._targets.append((0,0,center))

        for i in range(number):
            rand = np.random.random_integers(1, 4)
            if rand == 1:
                xgame = -coefficient
                ygame = coefficient
                text = 'Top - Left'
            if rand == 2:
                xgame = coefficient
                ygame = coefficient
                text = 'Top - Right'
            if rand == 3:
                xgame = -coefficient
                ygame = -coefficient
                text = 'Bottom - Left'
            if rand == 4:
                xgame = coefficient
                ygame = -coefficient
                text = 'Bottom - Right'
            self._targets.append((xgame,ygame,text))
            
            self._targets.append((0,0,center))


    # helper functions
    def check_username(self):
        if os.path.isdir(self._cur_dir):
            self._cur_dir = self._cur_dir + '/' +self._userID
            if os.path.isdir(self._cur_dir):
                print('user : ' + self._userID + ' is found')
            else:
                os.mkdir(self._cur_dir)
                print('creating new user account')
                print('user '+self._userID+' is registered')

    def check_folder_number(self):
        '''
            checks how many folder of assessment user has
        '''
        # check how many folders are there under certain assessment
        for base, dirs, files in os.walk(self._cur_dir):
            for directories in dirs:
                self._ass_no += 1

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
            } )  
                

        self._data_JSON.append(data)

    def save_json(self):
        data = self._data_JSON

        data_save = {
            "userID" : self._userID,
            "task_archetype": self._assessment,
            "task_archetype_number": self._ass_no,
            "device": device,
            "calibration":self._cal_json,
            "data": data
        }
        os.mkdir(self._cur_dir +'/'+ self._assessment + str(self._ass_no))
        with open(self._cur_dir +'/'+ self._assessment + str(self._ass_no) +'/'+'assessment_data.json', 'w') as f:
            json.dump(data_save, f)

    # def geospatial_plotting(self):
    #     '''
    #         changing every iteration!
    #         if data is updated.
    #     '''
    #     plt.plot(self._targets[0], self._targets[1],'ro')
    #     plt.axis([-11, 11, -11, 11])
    #     plt.axhline(0, color='black')
    #     plt.axvline(0, color='black')
    #     plt.show()
            
    # def check_assessment_folders(self):
    #     # if you create your own assessment, create a constant then put it in the list
    #     assessments = [five_point, randomN] # < -- add here
    #     for string in assessments:
    #         if not os.path.isdir(self._cur_dir + string):
    #             os.mkdir(self._cur_dir + string) 
    #             print('creating ' + string + ' assessment folder')    
    

    # Soccer Training Technology Algorithm
    # assessment menu
    def begin_assessments(self):
            os.system('clear')
            ready = raw_input("press 1 to start \n")
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

    # algorithm helper
    # after warping
    def mapFromToX(self, x):
        coefficient = side_length/2
        a = 0.0
        b = 360
        c = -coefficient
        d = coefficient
        y=round((x-a)/(b-a)*(d-c)+c,2)
        return y

    def mapFromToY(self, x):
        coefficient = side_length/2
        a = 0.0
        b = 360
        c = coefficient
        d = -coefficient
        y=round((x-a)/(b-a)*(d-c)+c,2)
        return y

    def distance_two_points(self):
        i = self._checker_iter
        delta_x = self._targets[i][0] - self._targets[i-1][0]
        delta_y = self._targets[i][1] - self._targets[i-1][1]
        dist = round(np.sqrt(abs(delta_x**2 + delta_y**2)) ,2)
        self._distance_covered_2p = dist
        self._total_distance_covered += dist        

    def speed_check(self):
        self._current_speed = round(self._total_distance_covered / self._stopwatch,2)

    # functions    
    def warping(self,data):
        '''
            convert camera pixel location into real live 2D
            changing every iteration! 
            ~ data is saved to heap using data_append_json
        '''
        
        loc_3d_x = data["center"]["x"]
        loc_3d_y = data["center"]["y"]
        self._user_warped_location = np.matmul(self._trf_mtx, np.array([[loc_3d_x],[loc_3d_y],[1]]))
        self._user_warped_location /= self._user_warped_location[2]
        self._x_warped = round(self._user_warped_location[0],2)
        self._y_warped = round(self._user_warped_location[1],2)
        self._x_warped = self.mapFromToX(self._x_warped)
        self._y_warped = self.mapFromToY(self._y_warped)

        
        

    def checker_algorithm(self):
        '''
            simple checkpoint algorithm
            args: a list of checkpoint target coordinates
        '''
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

    def threshold_loc(self):
        '''
            how large the threshold for every check point is
        '''
        userx = np.absolute(self._x_warped)
        usery = np.absolute(self._y_warped)
        trgt = self._targets[self._checker_iter]
        trgtx = np.absolute(trgt[0])
        trgty = np.absolute(trgt[1])
        # check
        dx = userx - trgtx
        dy = usery - trgty
        dxy = np.sqrt(dx**2 + dy**2)
        # if less than how many centimeters
        if dxy <= 20:
            self._check = True
        else:
            self._check = False
    
    def user_login(self):
        '''
            simple interface menu to log in user
        '''
        print("Hi I'm Beta, your virtual AI helper\n")
        self._userID = str(raw_input("Confirm with me, what is your name?\n"))
        right_username = False
        while right_username == False:
            boolean = raw_input("is your username: " + self._userID + ' ? [y/n]')
            if boolean == 'Y' or boolean == 'y':
                right_username = True
            else:
                self._userID = raw_input("Please reconfirm with me, what is your name?\n")

        # checking existing usernames
        self.check_username()
        os.system('cls' if os.name == 'nt' else 'clear')
            

    def menu_pick(self):
        '''
            simple interface assessment page 
        '''
        mainmenu = str(raw_input("press 1 to do assessment \npress 2 to check your progress \n   0 = X_X = 0\n\n"))
        if mainmenu == '1':
            assessment_id = str(raw_input("Please choose one of the assessments; \n1. Five Points \n2. Random N \n"))
            if assessment_id == '1':
                self.assessment_FivePoint()
                self._assessment = 'FivePoint'

            if assessment_id == '2':
                self.assessment_RandomN()
                self._assessment = 'RandomN'

        elif mainmenu == '2':
            print('Need more data, please do the assessments')
            time.sleep(2)
            self.menu_pick()
        
        elif mainmenu == '0':
            self._shutdown = True
            print('Bye - Bye')
            time.sleep(2)
        os.system('cls' if os.name == 'nt' else 'clear')

    def stt_main(self):
        '''
            the back bone of STT
        '''
        # area calibration
        self._trf_mtx = ac.ArUco_main()
        self._cal_json = ac.calibrate_JSON()
        #initializing 
        self._finish = False
        # initializing user detection dependencies
        ud = UserDetection()
        # starting time count
        os.system('cls' if os.name == 'nt' else 'clear')
        # menu
        self._shutdown = False
        while not (self._shutdown):
            self.user_login()
            self.check_folder_number()
            while not (self._shutdown):
                self.menu_pick()
                # count down
                self.begin_assessments()
                time_start = time.time()
                while (self._finish == False and self._shutdown == False):
                    self._stopwatch = round(time.time() - time_start, 2)
                    success, data, img= ud.run_detection(self._stopwatch) 
                    if success == True:
                        # warping to real life mapping
                        stt.warping(data)
                        # algorithm kick in
                        stt.checker_algorithm()
                        # saving data to JSON
                        stt.data_append_json()
                        # plotting
                        # stt.geospatial_plotting()
                        img = jetson.utils.cudaToNumpy(img)
                        cv2.imshow("live",img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    else:
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print(' there is no user bounded')
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    self._check = False
                self.save_json()
                

                if self._shutdown == False:
                    again = raw_input("Are you ready for another assessment? [y,n]\n")
                    if again == 'Y' or again == 'y':
                        continue
                    else:
                        print("See you later Alpha! \n")
                        break

if __name__ == '__main__':
    stt = STTDevelopment()
    ac = ac.ArucoChecker()
    stt.stt_main()