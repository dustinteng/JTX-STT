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

# initial conditions for video processing
video_pixels = (640,360) # the logitec and asus camera that is being used are using those resolutions
resolution_percentage = 100 # how many percent out of screen_pixels
camera_path = '/dev/video0'
fps = 30
count = 0
network = "ssd-mobilenet-v2"
overlay = "box,labels,conf"
threshold = 0.5
frames=[]

# initial conditions for the ArUco Markers
mtx = np.mat([[1.35007461e+03, 0.00000000e+00, 9.64381787e+02],[0.00000000e+00, 1.34859409e+03, 6.10803328e+02],[0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist = np.mat([ 0.10485946, -0.24625137, 0.00903166, -0.00257263, -0.09894589])
side_length = 1.2
dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
a_width, a_height = 360,360 #top view width and hight pixels

# string - directories
#please stick to naming convension ex: run_[experiment number]_[percentage]_[ description]
folder = './beta_run_1_'+str(resolution_percentage) + 'p'
video_URI = folder + '/video.mp4'
video_json = folder + '/video_data.json'
# lowRes_video_URI = folder + '/lowResVid.mp4'
bounded_video = folder + '/boundedVid.mp4'
# preframes = folder + '/preprocess_frames'
# postframes = folder + '/postprocess_frames'


#helper functions:
def frame_string_maker(folder, count, post_bool):
	''' to create strings for frame titles'''
	if post_bool == False:
		string = preframes  + "/%#05d.jpg" % (count+1)
	else:
		string = postframes  + "/%#05d.jpg" % (count+1)
	return string

def rescale_frame(frame, percent=75):
	''' to help rescale the frame '''
	width = int(frame.shape[1] * percent/ 100)
	height = int(frame.shape[0] * percent/ 100)
	dim = (width, height)
	return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def resolution_converter(inputPixel,percentage):
	''' resolution converter'''
	w = inputPixel[0] * percentage/100
	h = inputPixel[1] * percentage/100
	return (w,h)

def video_information_to_json(total_frames,video_duration):
	data = {
		'total_frames' : total_frames , 
		'video_duration' : video_duration
		}
	with open(video_json, 'w') as json_file:
		json.dump(data,json_file)

def reading_video_json(video_json):
	with open(video_json) as json_file:
		data = json.load(json_file)
	total_frames = data['total_frames']
	video_duration = data['video_duration']
	print('time steps = ' + str(round(video_duration,2)/total_frames))
	return total_frames,video_duration


#functions:
def save_video(input_URI, video_URI):
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
	print('frame size is: ' + str(size))
	# Below VideoWriter object will create a frame of above defined The output 
	# is stored in output_URI file.
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	result = cv2.VideoWriter(video_URI, fourcc , fps , size)
	
	# create start time and frame counter
	start = time.time()
	fcounter = 0
	while(video.isOpened):
		ret, frame = video.read()
		if ret == True: 
			#flip frame y axis = 1, x axis = 0 , -1 for flipping axis.
			# frame = cv2.flip(frame,1) # can't use this because we want to check the markers
			#write frame
			result.write(frame)
			fcounter += 1
			# Display the frame saved in the file
			cv2.imshow('Frame', frame)
			# Press q on keyboard to stop the process
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	
		# Break the loop
		else:
			break

	# check total time for all frames
	end = time.time()
	video_length = round(end-start,2)
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

def lower_resolution_video (video_URI, output_URI, target_resolution):
	''' this functionn uses ffmpeg to lower down the resolution you want 
		params:
		video_URI -> location of the video of interest
		target_resolution -> a list containing width and height of our interest.

		returns:
		none
		'''
	
	os.system('ffmpeg -i ' + video_URI + ' -s ' + target_resolution[0]+ 'x' + target_resolution[1] + ' ' + lowRes_video_URI)

def video_to_frames(video_URI):
	""" video_to_frames is a function to convert video to frames
		it convert the video into an object to be passed in the detection.
		Args:
			video_URI - video source - should be 'video_filename.etc'
		Returns:
			preprocess frames in preprocess folder """
			
	#Log the time
	time_start = time.time()
	old_timestamp = time_start
	# Start capturing the feed
	cap = cv2.VideoCapture(video_URI)
	# Find the number of frames
	video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print ("vidd length: ", video_length)
	count = 0
	print ("Converting video..\n")

	# Start converting the video
	while cap.isOpened():
		if count < video_length:
			count = count + 1
		# print(count)
		# Extract the frame
		ret, frame = cap.read()
		# saving the frames back to an output location with ordered names.
		# cv2.imwrite(frames_dir + "/%#05d.jpg" % (count+1), frame)
		frames.append(frame)
		# If there are no more frames left
		if (count >= (video_length)):
			# Log the time again
			time_end = time.time()
			#delta time
			delta_time = time_end-time_start
			# Release the feed
			cap.release()
			# Print stats
			print ("Done extracting frames.\n%d frames extracted to a list" % count)
			print ("It took %d seconds forconversion." % delta_time)
			break
			
	return count, frames


def detecting_markers_location(image):
	''' checking roam bound using ArUco markers 
		args = image
		returns = boolean'''
	print(type(image))
	# image = Image.fromarray(np.array(image))
	
	markers, ids, rejected = cv2.aruco.detectMarkers(image, dictionary = dic, cameraMatrix = mtx, distCoeff = dist)
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


def area_bound_calibration(frames):
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

def whole_detection(network, threshold, frames, total_frames, video_duration):
	""" whole_detection is a function that takes in images in a folder then recognize the bounding
		box and at the same time will store important data to a json file which will be saved in the same directory
		Args:
			network = "ssd-mobilenet-v2" # pre-trained model to load
			sys.argv  # is a list in Python, which contains the command-line arguments passed to the script.
			default-threshold = 0.5 # default detection threshold to use
			input_URI #input directory 
			output_URI #output directory
		Return:
			detected frames in postprocess folder 
			"""

	# load the object detection network
	net = jetson.inference.detectNet(network, threshold)
	# detection intialization 
	count = 0
	db = []
	#saving a video to a location
	display = jetson.utils.videoOutput(bounded_video)
	# processing each frame
	for frame in frames:
		count += 1
		# detect objects in the image (with overlay)
		img = jetson.utils.cudaFromNumpy(frame)
		detections = net.Detect(img)
		display.Render(img)
		# print the detections
		# print("detected {:d} objects in image".format(len(detections)))
		person_bool = False
		for detection in detections:
			if detection.ClassID == 1 and person_bool == False:
				left = int(detection.Left)
				right = int(detection.Right)
				top = int(detection.Top)
				bot = int(detection.Bottom)
				center = (int(detection.Center[0]),int(detection.Center[0]))
				area = int(detection.Area)
				time = round(round(video_duration,2) / total_frames * count,2)
				print(str(time) + ',' + str(left) + ',' + str(bot) + ',' + str(right) + ',' + str(top)  )
				data = {
					"time" : time, 
					"left" : left,
					"bot" : bot,
					"right" : right,
					"top" : top,
					"center" : center,
					"area" : area
				}
				db.append(data)
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
		if person_bool == False:
			print('Bounding box not detected')
			#store in json file at once
		
		# render the image
		# output.Render(img)

		# update the title bar
		#output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))

		# print out performance info
		# net.PrintProfilerTimes()

		# exit on input/output EOS
		# if not input.IsStreaming() or not output.IsStreaming():
		# 	break
	return db

def main():
	v = False
	j = False
	f = False
	#checking if we're using an existing video / camera
	if os.path.isfile(video_URI):
		v = True 
	if os.path.isfile(video_json):
		j = True
	if os.path.isdir(folder):
		f = True

	if f == False:
		print("Creating folder")
		os.mkdir(folder)
	# eq

	# saving video from camera feed
	total_frames, video_duration = save_video(camera_path, video_URI)
	# fps = round(float(total_frames)/video_duration,2)
	# print('total_frames : ' + str(total_frames))
	video_information_to_json(total_frames,video_duration)
	total_frames, video_duration = reading_video_json(video_json)
	count, frames = video_to_frames(video_URI)
	matrix, loc_mid, homography= area_bound_calibration(frames)
	print ('matrix = ' +str(matrix))
	print ('homography = ' +str(homography))
	print (loc_mid)
	print 'bbbb'
	# print np.array([[loc_mid[0]],[loc_mid[1]],[1]])
	print matrix
	lalala = np.matmul(matrix, np.array([[loc_mid[0]],[loc_mid[1]],[1]]))
	lalala /= lalala[2]
	print lalala
	# locmidnow = matrix * np.array
	# db = whole_detection(network,threshold,frames,total_frames,video_duration)
	# print db
	# A = np.array([[207,265]], dtype=np.float32)
	# location = np.matmul(matrix,loc_mid)
	# locc = []
	# locc =cv2.perspectiveTransform(loc_mid,matrix)
	# print locc
	# locs = cv2.getAffineTransform([[207],[265],[1]],matrix)
	
	# print locs

	# 	data = []
	# 	for x in range(3):
	# 		for y in range(3):
	# 			data.append(matrix[x][y])
	# 	matrix_db.append(data) #9x9

	
	# # averaging each constant
	# m_index = 9
	# count = 0
	# to_matrix = []
	# for x in range(m_index):
	# 	value = 0
	# 	for y in range(m_index):
	# 		value = value + matrix_db[y][x]
	# 	value = value / m_index
	# 	to_matrix.append(value)
	# print(str(len(to_matrix)))

	# mtx = []
	# for x in range(3):
	# 	alist = []
	# 	for y in range(3):
	# 		alist.append(matrix[x][y])
	# 	mtx.append(alist)
	
	# human detection


	# #saving lowered res video
	# target_resolution = resolution_converter(video_pixels,percentage):
	# lower_resolution_video (video_URI, lowRes_video_URI, target_resolution)

	# converting video to frames
	# frame_number = video_to_frames(fps,video_URI,preframes)
	# human_bounding_box_video(network, threshold, video_URI)

if __name__ == "__main__":
	main()

