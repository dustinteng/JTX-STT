import time
import cv2
import os

fps = 10
period = 1.0/fps
def video_to_frames(input_loc, output_loc):
    """Function to extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
    """
    


    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    count = 0
    print ("Converting video..\n")
    # Start converting the video

        # Log the time
    time_start = time.time()
    old_timestamp = time_start
    imageArray = []
    while cap.isOpened():
        ret, frame = cap.read()

        if ((time.time() - old_timestamp) > period)):
        # Extract the frame
            old_timestamp = time.time()
            ret, frame = cap.read()
            # Write the results back to output location.
            imageArray.append (frame) 
            print(str(time.time()-time_start))
            #cv2.imwrite(output_loc + "/%#05d.jpg" % (count), frame)
            count = count + 1
            print(count)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            # Release the feed
    cap.release()
    time_end = time.time()
    print(str(time_end-time_start))
    # Print stats
    print ("Done extracting frames.\n%d frames extracted" % count)
    print ("It took %d seconds forconversion." % (time_end-time_start))
    

if __name__=="__main__":

    input_loc = '/dev/video0'
    output_loc = './data/output/prefolder'
    if not os.path.isdir('./data/output/prefolder'):
        print("Creating folder")
        os.mkdir('data')
        os.mkdir('data/output')
        os.mkdir('data/output/prefolder')
    video_to_frames(input_loc, output_loc) 