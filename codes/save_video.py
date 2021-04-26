# Python program to save a 
# video using OpenCV
  
   
import cv2
import time
   
# Create an object to read from camera
cap = cv2.VideoCapture('/dev/video1')
# Source pixels
spixels = (1280,720)
# Adjusting the quality
percentage = 100
# We need to check if camera is opened previously or not
if (cap.isOpened() == False): 
    print("Error reading video file")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)


# File name
filename = 'cam_video.avi'

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def pixelhelper(inputPixel,percentage):
    w = inputPixel[0] * percentage/100
    h = inputPixel[1] * percentage/100
    return (w,h)

##change here

pixels = pixelhelper(spixels,percentage)
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
result = cv2.VideoWriter(filename, fourcc , 30 , size)

# check start time
start = time.time()
while(True):
    ret, frame = cap.read()
  
    if ret == True: 
        frame = rescale_frame(frame, percent=percentage)
        # Write the frame into the filename
        result.write(frame)
        # Display the frame saved in the file
        cv2.imshow('Frame', frame)
        # Press q on keyboard to stop the process
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  
    # Break the loop
    else:
        break
end = time.time()
delta_time = end-start
print('time = ' + str(delta_time))
# When everything done, release the video capture and video write objects
cap.release()
result.release()   
# Closes all the frames
cv2.destroyAllWindows()
   
print("The video was successfully saved")

