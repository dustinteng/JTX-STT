import cv2
import time

#Capture video from webcam
vid_capture = cv2.VideoCapture('/dev/video0')
vid_cod = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter("cam_video.mp4", vid_cod, 20.0, (640,480))

start = time.time()
count = 0
while(True):
    count += 1
    # Capture each frame of webcam video
    ret,frame = vid_capture.read()
    cv2.imshow("My cam video", frame)
    output.write(frame)
    # Close and break the loop after pressing "x" key
    if cv2.waitKey(1) &0XFF == ord('x'):
        break
end = time.time()
totaltime = end - start
print(totaltime)
print(count)
# close the already opened camera
vid_capture.release()
# close the already opened file
output.release()
# close the window and de-allocate any associated memory usage
cv2.destroyAllWindows()