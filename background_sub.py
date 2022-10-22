import cv2
import numpy as np


# Create a VideoCapture object
video_cap = cv2.VideoCapture("cars.mp4")
# initialize the background subtractor object
background_sub = cv2.createBackgroundSubtractorMOG2()

# loop through the video frames
while True:
    # read the video frame
    success, frame = video_cap.read()
  
    # if there is no more frames to show, break the loop
    if not success:
        break
        
    # apply the background subtractor to the frame
    mask = background_sub.apply(frame)
    # apply the opening morphological operation to the mask to remove the noise
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # get the foreground frame by applying the mask to the original frame
    new_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # get the contours of the moving objects
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # loop over the contours
    for contour in contours:
        # if the contour is too small, ignore it
        if cv2.contourArea(contour) > 1000:
            # get the bounding rectangle of the contour
            (x, y, w, h) = cv2.boundingRect(contour)
            # draw the bounding rectangle on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # convert the mask to 3 channels
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    opening = cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR)
    # stack the frame horizontally
    hstacked_frames = np.hstack((frame, new_frame))
    hstacked_frames1 = np.hstack((mask, opening))
    # stack the frame vertically
    vstacked_frames = np.vstack((hstacked_frames, hstacked_frames1))
    cv2.imshow("Frame + New frame + Mask + Opening operation", vstacked_frames)
        
    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(30) == ord("q"): 
        break
    
# release the video capture object
video_cap.release()
cv2.destroyAllWindows()
