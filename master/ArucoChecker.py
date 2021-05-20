''' This file is 2/3 of 
    the Soccer Training Technology's developed at UCSD for 
    1. MAE 199 - Spring 2021 with prof Jack Silberman and prof Mauricio de Oliveira
    2. Triton AI - STT 

    arucoChecker check area play boundary using
    camera: Logitec HD Pro Webcam C920 (mtx and dist are calibrated)
    device: Jetson Xavier NX

    This file deals with 
    1. area bound reading
    2. averaging plane transformation matrix
    3. saving matrix transformation data to a JSON file

    creator: Jan Dustin Tengdyantono
'''

import cv2
import numpy as np
import time
import os
import json


# constants
a_width, a_height = 360,360 #2D warped 
local_counter_max = 100

class ArucoChecker(object):
    def __init__(self):
        # camera properties
        self._cam_path = '/dev/video0'
        self._mtx = np.mat([[1.35007461e+03, 0.00000000e+00, 9.64381787e+02],[0.00000000e+00, 1.34859409e+03, 6.10803328e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self._dist = np.mat([ 0.10485946, -0.24625137, 0.00903166, -0.00257263, -0.09894589])
        self._side_length = 19 # in mm
        self._fps = 30
        # ArUco and calibration properties
        self._cal_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self._cal_frames = []
        self._cal_time = 6 # C * 30 fps
        # warping properties
        self._trf_mtx = None
        
        # directory properties
        self._cur_dir = '/home/user/projects/JTX-STT/user_data'
        


    # helper function
    def camera_check(self):
        '''
            Display real time camera frame
            params : camera-path string
            return : None
        '''
        # Create an object to read from camera
        video = cv2.VideoCapture(self._cam_path)
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

    # functions: AruCo markers 1/3 ( runs after recording video )
    def detecting_markers_location(self, image):
        ''' checking roam bound using ArUco markers 
            args = image
            returns = boolean'''
        # print(type(image))
        # image = Image.fromarray(np.array(image))
        
        markers, ids, rejected = cv2.aruco.detectMarkers(image, dictionary = self._cal_dict, cameraMatrix = self._mtx, distCoeff = self._dist)
        if not np.any(markers):
            return False, [], [], []
        
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(markers, self._side_length, self._mtx, self._dist)

        if not np.any(tvecs):
            return False, [], [], []

        assert (len(markers) == len(ids) == len(tvecs)), "Must have same number of markers, ids and tvecs"

        image_copy = np.copy(image)
        cv2.aruco.drawDetectedMarkers(image_copy, markers)
        for i in range(len(ids)):
            cv2.aruco.drawAxis(image_copy, self._mtx, self._dist, rvecs[i], tvecs[i], 0.05)
            marker = markers[i][0]
            x, y, z = marker[0][0], marker[0][1], tvecs[i][0][2]
            image_area_marked = cv2.putText(image_copy, "id = {0:d}".format(ids[i][0]), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return True, tvecs, ids, image_area_marked 
    # functions: AruCo markers 2/3 ( runs after recording video )
    def warp_markers(self, image, images_count):
        ''' 
            checking detected markers 
            args: 	image = current image
                    ids = what are the id's visible
                    images_count = non-faulty images
            returns: location of each inner verteces - markers are not included to the frame
            
        '''
        X = False
        images_count = images_count
        local_counter = 0
        try:
            markers, ids, rejected = cv2.aruco.detectMarkers(image=image, dictionary = self._cal_dict, cameraMatrix = self._mtx, distCoeff = self._dist)
            mark1 = np.where(ids == [1])[0][0]
            mark2 = np.where(ids == [2])[0][0]
            mark3 = np.where(ids == [3])[0][0]
            mark4 = np.where(ids == [4])[0][0]
            # mark5 = np.where(ids == [5])[0][0]
            X = True

        except IndexError:
            print("one or more markers are not detected")
            
        local_counter += 1

        if X == True:
            one = np.float32(markers[mark1][0][2])
            two = np.float32(markers[mark2][0][3])
            three = np.float32(markers[mark3][0][1])
            four = np.float32(markers[mark4][0][0])
            # xarray = []
            # yarray = []
            # for i in range(4):
            #     five = np.float32(markers[mark5][0][i])
            #     xarray.append(five[0])
            #     yarray.append(five[1])
            # midx = np.average(xarray)
            # midy = np.average(yarray)
            data = [one,two,three,four,images_count]
            # mid = np.asarray([midx,midy])
            return X, data
        else:
            data = [0,0,0,0,0]
            return X, data
    # functions: AruCo markers 3/3 ( runs after recording video )
    def area_calibration_run(self, frames):
        '''
            checking play area boundary by averaging the location of the markers
            uses both detecting_markers_location and area_bound_locations functions

            args: frames

            returns: transformation matrix
        '''
        N = len(frames)
        one = []
        two = []
        three = []
        four = []
        # five = []
        images_count = 0
        cal_frame_trgt = 10
        # checking detected markers and averaging the locations
        for n in range(N):
            if images_count < cal_frame_trgt:
                frame = frames[n]
                image = frame
                # image = cv2.flip(image,1) #do not use this, aruco can't be read and stuffs will then get hard.
                #detecting area bound
                naeloob, tvecs, ids, image_area_marked = self.detecting_markers_location(image)
                success, data = self.warp_markers(image, images_count)
                if success == True:
                    one.append(data[0])
                    two.append(data[1])
                    three.append(data[2])
                    four.append(data[3])
                    # five.append(mid) #change this later
                    images_count = data[4] + 1 #checking image count for average geometric translation matrix
        print(str(images_count) + ' stable images')
        # error checking if the camera can detect all bourdaries
        if images_count == cal_frame_trgt :
        # averaging the location of each markers
            loc_one = np.float32(sum(one)/len(one))
            loc_two = np.float32(sum(two)/len(two))
            loc_three = np.float32(sum(three)/len(three))
            loc_four = np.float32(sum(four)/len(four))
            # loc_mid = np.float32(sum(five)/len(five))

            # perspective transformation
            point2 = np.float32([[0,0],[a_width,0],[0,a_height],[a_width,a_height]])
            point1 = np.float32([loc_one,loc_two,loc_three,loc_four])
            matrix = cv2.getPerspectiveTransform(point1,point2)
            image_warped = cv2.warpPerspective(image,matrix,(a_width,a_height))
            # show ArUco markers area boundary
            print(' Transformation Matrix:')
            print(matrix)
            image = image_area_marked
            #print(type(image_warped))
            #showing area
            cv2.imshow("Area Calibration", image)
            # showing warped images
            cv2.imshow("Warped", image_warped)
            success = True 
            time.sleep(2)
            cv2.destroyAllWindows()
            return matrix, success
        
        if images_count < cal_frame_trgt :
            ('not successfully calibrating play area')
            success = False
            matrix = None
            return matrix, success


    # functions: record video 
    def record_calibration_video(self):
        ''' 
        appending frames a variable - heap memory allocation
        params: input_URI - camera path
        returns: None
        '''
        ############## check again after making folders
        # if not os.path.isdir(self._cur_dir):
        #     print("folder " + folder + ' is not detected')
        #     print("Creating folder")
        #     os.mkdir(folder)
        # Create an object to read from camera

        video = cv2.VideoCapture(self._cam_path)
        # We need to check if camera is opened previously or not
        if (video.isOpened() == False): 
            print("Error reading video file")

        # check video resolution
        # frame_width = int(video.get(3))
        # frame_height = int(video.get(4))

        # initializing frames video writer 
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # result = cv2.VideoWriter(calibration_URI, fourcc , fps , size)

        # create start time and frame counter
        time_start = time.time()
        time_now = time_start
        while(video.isOpened):
            ret, frame = video.read()
            time_now= time.time()
            if ret == True:
                # result.write(frame)
                self._cal_frames.append(frame)
                # Display the frame saved in the file
                # cv2.imshow('Frame', frame)
                delta_t = time_now - time_start
                # break after 
                if delta_t > self._cal_time and cv2.waitKey(1):
                    break 
            # Break the loop
            else:
                break
        # When everything done, release the video capture and video write objects
        video.release()
        # Closes all the frames
        # cv2.destroyAllWindows()	


    def calibrate_JSON(self):
        '''
            adding calibration data to be saved later on
        '''
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        # trf_mtx = self._trf_mtx.tolist()
        data = (
            {
                "time": current_time,
                # "camera_matrix": self._trf_mtx
            }
        )
        return data

    def ArUco_main(self):
        ''' 
            checking area bound by checking ArUco markers
            mtx - camera matrix model: logitec

        '''

        self.record_calibration_video()

        ######################################
        run = self.area_calibration_run(self._cal_frames)
        trf_mtx, success = run
        # print('check point')
        if success:
            self._trf_mtx = trf_mtx
            return self._trf_mtx
            
        if not success:
            self.camera_check()
            self.ArUco_main()
        