import jetson.inference
import jetson.utils
import time

fps = 10
period = 1.0/ fps
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = jetson.utils.videoSource("/dev/video1")      # '/dev/video0' for V4L2
display = jetson.utils.videoOutput("display://0") # 'my_video.mp4' for file

timestamp = []
lleft = []
lright = []
ltop = []
lbot = []

start = time.time()

count = 0
while display.IsStreaming():
    currtime = time.time()
    img = camera.Capture()
    detections = net.Detect(img)
    display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    if currtime >= period:
        count = count + 1
        timestamp.append(round(count * period,2))

        for detection in detections:
            index = 0
            if detection.ClassID == 1 and index == 0:
                left = int(detection.Left)
                right = int(detection.Right)
                top = int(detection.Top)
                bot = int(detection.Bottom)
                center = (int(detection.Center[0]),int(detection.Center[0]))
                area = int(detection.Area)
                lleft.append(left) 
                lright.append(right) 
                ltop.append(top) 
                lbot.append(bot) 
                #print(detection)
                # print('Left-Bottom (X,Y) =  ('+ str(left) + ',' + str(bot) + ')')
                # print('Right-Bottom (X,Y) =  ('+ str(right) + ',' + str(bot) + ')')
                # print('Left-Top (X,Y) =  ('+ str(left) + ',' + str(top) + ')')
                # print('Right-Top (X,Y) =  ('+ str(right) + ',' + str(top) + ')')
                # print('Center (X,Y) = ('+ str(center[0]) + ',' + str(center[1]) + ')')
                # print('Area = ' + str(area) )
                index += 1

            # elif detection.ClassID = 1
    display.Render(img)
    print(timestamp)
