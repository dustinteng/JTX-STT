''' This file is 2/2 of 
    the Soccer Training Technology's developed at UCSD for 
    1. MAE 199 - Spring 2021 with prof Jack Silberman and prof Mauricio de Oliveira
    2. Triton AI - STT 

    arucoChecker check area play boundary using
    camera: Logitec HD Pro Webcam C920 (mtx and dist are calibrated)
    device: Jetson Xavier NX

    creator: Jan Dustin Tengdyantono
'''

import cv2 as cv2
import numpy as np

a_width, a_height = 360,360

class arucoChecker(object):
    def __init__(self, mtx, dist, side_length):
        self._mtx = mtx
        self._dist = dist
        self._side_length = side_length
        self._dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    def detecting_markers_location(self, image):
        ''' checking roam bound using ArUco markers 
            args = image
            returns = boolean'''
        print(type(image))
        # image = Image.fromarray(np.array(image))
        
        markers, ids, rejected = cv2.aruco.detectMarkers(image, dictionary = self._dict, cameraMatrix = self._mtx, distCoeff = self._dict)
        if not np.any(markers):
            return False, [], [], []
        
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(markers, side_length, mtx, dist)

        if not np.any(tvecs):
            return False, [], [], []

        assert (len(markers) == len(ids) == len(tvecs)), "Must have same number of markers, ids and tvecs"

        image_copy = np.copy(image)
        cv2.aruco.drawDetectedMarkers(image_copy, markers)
        for i in range(len(ids)):
            cv2.aruco.drawAxis(image_copy, mtx, dist, rvecs[i], tvecs[i], 0.05)
            marker = markers[i][0]
            x, y, z = marker[0][0], marker[0][1], tvecs[i][0][2]
            image_copy=cv2.putText(image_copy, "id = {0:d}".format(ids[i][0]), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return True, tvecs, ids, image_copy 

    def area_bound_locations(image,ids,images_count):
        ''' 
            checking detected markers 
            args: 	image = current image
                    ids = what are the id's visible
                    images_count = non-faulty images
            returns: location of each inner verteces - markers are not included to the frame
            
        '''
        X = False
        try:
            markers, ids, rejected = cv2.aruco.detectMarkers(image=image, dictionary = dic, cameraMatrix=mtx, distCoeff=dist)
            mark1 = np.where(ids == [1])[0][0]
            mark2 = np.where(ids == [2])[0][0]
            mark3 = np.where(ids == [3])[0][0]
            mark4 = np.where(ids == [4])[0][0]
            mark5 = np.where(ids == [5])[0][0]
            X = True

        except IndexError:
            print("one or more markers are not detected")
            time.sleep(0.2)
        
        if X == True:
            one = np.float32(markers[mark1][0][2])
            two = np.float32(markers[mark2][0][3])
            three = np.float32(markers[mark3][0][1])
            four = np.float32(markers[mark4][0][0])
            xarray = []
            yarray = []
            for i in range(4):
                five = np.float32(markers[mark5][0][i])
                xarray.append(five[0])
                yarray.append(five[1])
            midx = np.average(xarray)
            midy = np.average(yarray)
            images_count += 1
            data = [one,two,three,four,images_count]
            mid = np.asarray([midx,midy])
            return X, data, mid
        else:
            data = [0,0,0,0,0]
            mid = 0
            return X, data , mid


    def area_bound_calibration(self, frames):
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
        five = []
        images_count = 0
        # checking detected markers and averaging the locations
        for n in range(N):
            if images_count < 10:
                frame = frames[n]
                image = frame
                # image = cv2.flip(image,1) #do not use this, aruco can't be read and stuffs will then get hard.
                #detecting area bound
                naeloob, tvecs, ids, image_copy = detecting_markers_location(image)
                
                success, data, mid = area_bound_locations(image,ids,images_count)
                if success == True:
                    one.append(data[0])
                    two.append(data[1])
                    three.append(data[2])
                    four.append(data[3])
                    five.append(mid)
                    images_count = data[4] + 1

        # averaging the location of each markers
        loc_one = np.float32(sum(one)/len(one))
        loc_two = np.float32(sum(two)/len(two))
        loc_three = np.float32(sum(three)/len(three))
        loc_four = np.float32(sum(four)/len(four))
        loc_mid = np.float32(sum(five)/len(five))
        # loc_mid = np.float32(sum(four)/len(four))
        point2 = np.float32([[0,0],[a_width,0],[0,a_height],[a_width,a_height]])
        point1 = np.float32([loc_one,loc_two,loc_three,loc_four])
        matrix = cv2.getPerspectiveTransform(point1,point2)
        image_warped = cv2.warpPerspective(image,matrix,(a_width,a_height))
        homography = cv2.findHomography(point1,point2,cv2.RANSAC)
        # showing warped images
        # cv2.imshow("Warped", image_warped)
        return matrix, loc_mid,homography