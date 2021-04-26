# Python program to save a 
# video using OpenCV
  
   
import cv2
import time
  
   
# Create an object to read 
# from camera
video = cv2.VideoCapture('/dev/video0')
fps = 10
period = 1.0/fps
# We need to check if camera
# is opened previously or not
if (video.isOpened() == False): 
    print("Error reading video file")
  
# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))
   
size = (frame_width, frame_height)
   
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
result = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
start = time.time()
old_timestamp = start
print('starting recording')
count = 0
while(True):
    ret, frame = video.read()
  
    if ret == True: 
        currtime = time.time()
        if (currtime - old_timestamp) > period:
            count +=1
            print(currtime - old_timestamp, "period")
            old_timestamp = currtime
            # Write the frame into the
            # file 'filename.avi'
            result.write(frame)
    
            # Display the frame
            # saved in the file
            #cv2.imshow('Frame', frame)
    
            # Press q on keyboard 
            # to stop the process
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
  
    # Break the loop
    else:
        break
    end = time.time()
    video_length = round(end-start,2)
    print(count)
print('total length : ' + str(video_length) + 's')
print('finished recording')
# When everything done, release 
# the video capture and video 
# write objects
video.release()
result.release()
    
# Closes all the frames
cv2.destroyAllWindows()
   
print("The video was successfully saved")