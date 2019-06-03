# This code uses dlib + pretrained model for facial landmark detection

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import os
from pynput.mouse import Listener,Controller,Button
from collections import deque


def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# take the mean of centers of 3 cosecutive frames (frames in deque)
def mean(face_centers,n):  
  fcx=int(sum([i[0] for i in face_centers ])/n)        # face center x-coordinate & y-coordinate after taking mean of face center of each frame
  fcy=int(sum([i[1] for i in face_centers ])/n)        
  return (fcx,fcy)

# compute the distance b/w     
def distance(face_centers1,face_centers2,n):      
  (fcx1,fcy1)=mean(face_centers1,n)       
  (fcx2,fcy2)=mean(face_centers2,n)
  # calculating the relative distance b/w two continuous face centers
  (rel_x,rel_y) = fcx2-fcx1,fcy2-fcy1
  return (rel_x,rel_y)

# creating a controller for mouse
mouse = Controller()

def on_move(x, y):
  mouse_event.append(1)
  #print("on_move")
  

def on_click(x, y, button, pressed):
  mouse_event.append(1)
  #print("On_click")

# obtaining face center by taking mean of nose coordinates
def nose_cord_mean(nose_cords, no_points):
    f_c_x = int(sum([i[0] for i in nose_cords])/no_points)
    f_c_y = int(sum([i[1] for i in nose_cords])/no_points)
    return [f_c_x, f_c_y]

# find the relative distance between consecutive frames    
def relative_distance(face_centers_past,face_centers_present):    
  (fc_x_past,fc_y_past)=mean_deque_points(face_centers_past)
  (fc_x_present,fc_y_present)=mean_deque_points(face_centers_present)
  (rel_x,rel_y) = fc_x_present-fc_x_past, fc_y_present-fc_y_past
  return (rel_x,rel_y)

# take the mean of points in deque
def mean_deque_points(face_centers):
    fc_x = int(sum([i[0] for i in face_centers])/no_frame_in_deque)
    fc_y = int(sum([i[1] for i in face_centers])/no_frame_in_deque)
    return (fc_x, fc_y)


# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
# for right click, there must be two blinks of 'per_r_blink_frame' frames each, within ----  consecutive frames 
# for eg. if there are two clicks of 2 frame each in 8-continuous frames then there will be a right click 
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES_single_click = 5
EYE_AR_CONSEC_FRAMES_right_click = 30
per_r_blink_frame = 3

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
CLICK = 0
d_counter = 0   # frame counter for (a blink of per_r_blink_frame) right click
continuous_frame = 0   # counter for continuous frames
d_blink = 0  # blink counter(according to per_r_blink_frame) for right_click

# constants for mouse movement
speed_x=30       # move the mouse speed_x and speed_y times faster w.r.t. change in x & y-coordinate of the face center point.
speed_y=30
no_frame_in_deque = 15  # no. of frames to store the face center of previous frames
# we will take mean of all points in deque to move the mouse smoothly

# give path for facial detector
curnt_dir = os.getcwd()
path_detector = curnt_dir + '/shape_predictor_68_face_landmarks.dat'

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_detector)

# grab the indexes of the facial landmarks for the left and
# right eye, and nose, respectively 
# nose coordinate for determining a point as face center( by taking the mean of nose coordinate)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(nstart, nend) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

# initialises the deque for storing the coordinates of face center of past few frames
face_center_presnt_frm = deque()
face_center_past_frm = deque()

# start the webcam
cam = cv2.VideoCapture(0)
# loop over frames from the video stream
while True:
	# creating a list for recording mouse event and creating a listener for mouse_events
    mouse_event = list([])
    listener = Listener(on_move=on_move, on_click=on_click)
	# starting the listener
    listener.start()        

    ret, frame = cam.read()
    # flip the frame to insure that your movement's direction matches the cursor direction. if you don't flip the frame then if you move in the right,
    # cursor will move in left direction
    frame = cv2.flip(frame, 1)
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
    rects = detector(gray, 0)
	# loop over the face detections
    for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # extract the nose coordinates
        nose_cord = shape[nstart:nend]
        # change or assign the value to face_center_past_frm and face_center_presnt_frm
        face_center_past_frm = face_center_presnt_frm.copy()
        
        face_center_presnt_frm.append(nose_cord_mean(nose_cord, len(nose_cord)))
        # find the distance to move the mouse
        if len(face_center_past_frm) == no_frame_in_deque:
            ret = face_center_presnt_frm.popleft()

            (move_rel_x, move_rel_y) = relative_distance(face_center_past_frm, face_center_presnt_frm) 
            # finally move the mouse
            if len(mouse_event) == 0:
                mouse.move( speed_x*move_rel_x, speed_y*move_rel_y)
                           
		# average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # increment the continuous_frame
        continuous_frame += 1

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            d_counter += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
        else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES_single_click:
                mouse.press(Button.left)
                mouse.release(Button.left)
                TOTAL+=1
            
            if d_counter >= per_r_blink_frame:
                d_blink += 1
            
            if d_blink >= 2 and continuous_frame <= EYE_AR_CONSEC_FRAMES_right_click:
                mouse.press(Button.right)
                mouse.release(Button.right)
                continuous_frame = 0
                d_blink =0
                CLICK += 1
            
            # reset the continuous_frame if there is no blink (according to per_r_blink_frame)(i.e. d_blink == 0) and d_counter == 0
            if d_counter == 0 and d_blink == 0:
                continuous_frame = 0
                print("d_counter becomes zero")
			# reset the eye frame counter
            COUNTER = 0
            d_counter = 0

        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "right click: {}".format(CLICK), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
	# show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
    if key == ord("q"):
        listener.stop()
        break

# do a bit of cleanup
cam.release()        
cv2.destroyAllWindows()