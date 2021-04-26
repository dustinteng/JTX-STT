import numpy as np
import cv2
import time
import os
import random
import sys


fps=10
period = 1.0/fps
video_codec=cv2.VideoWriter_fourcc('D','I','V','X')
name = './filename/'


cap = cv2.VideoCapture('/dev/video0)
ret=cap.set(3, cap.get(3))
ret=cap.set(4, cap.get(4))
# cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))


start = time.time()
video_file_count = 1
video_file = os.path.join(name, str(video_file_count) + ".avi")
print('Capture video saved location : {}'.format(video_file))


while(cap.isOpened()):
    start_time = time.time()
    ret, frame = cap.read()
    if ret==True:
        cv2.imshow('frame',frame)
        if (time.time() - start > ):
            start = time.time()
            video_file_count += 1
            video_file = os.path.join(name, str(video_file_count) + ".avi")
            video_writer = cv2.VideoWriter(video_file,video_codec, fps,(int(cap.get(3)),int(cap.get(4))))
            time.sleep(10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            print(video_file_count)

    else:
        break
    print(time.time()-start)
cap.release()
cv2.destroyAllWindows()