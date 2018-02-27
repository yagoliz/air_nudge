#!/usr/bin/env python

# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import rospy
from std_msgs.msg import Int8


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

def eye_direction_ratio(ear):

	# compute the eye aspect ratio
	edr = 1.0 / ear

	return edr

rospy.init_node("distraction_detection")
state_publisher = rospy.Publisher("/state", Int8, queue_size=1)
state = Int8()

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-w", "--webcam", type=int, default=0,
	help="index of webcam on system")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.3
DISTRACTION_THRES = 0.4
EYE_AR_CONSEC_FRAMES = 48
DISTRACTION_CONSEC_FRAMES = 48

# initialize the frame COUNTER_DROWSINESS as well as a boolean used to
# indicate if the alarm is going off
COUNTER_DROWSINESS = 0
COUNTER_DISTRACTION_RIGHT = 0
COUNTER_DISTRACTION_LEFT  = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

state.data = 0
# loop over frames from the video stream
while True:
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
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

		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		leftEDR = eye_direction_ratio(leftEAR)
		rightEAR = eye_aspect_ratio(rightEye)
		rightEDR = eye_direction_ratio(rightEAR)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame COUNTER_DROWSINESS
		if ear < EYE_AR_THRESH:
			COUNTER_DROWSINESS += 1
			COUNTER_DISTRACTION_LEFT  = 0
			COUNTER_DISTRACTION_RIGHT = 0
			# if the eyes were closed for a sufficient number of
			# then sound the alarm
			if COUNTER_DROWSINESS >= EYE_AR_CONSEC_FRAMES:
				# if the alarm is not on, turn it on

				# draw an alarm on the frame
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

				state.data = 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold, so reset the COUNTER_DROWSINESS and alarm
		else:
			COUNTER_DROWSINESS = 0

			if (rightEDR-leftEDR) > DISTRACTION_THRES: # Looking at left direction
				COUNTER_DISTRACTION_LEFT  += 1
				COUNTER_DISTRACTION_RIGHT  = 0

				if COUNTER_DISTRACTION_LEFT >= DISTRACTION_CONSEC_FRAMES:
					cv2.putText(frame, " LEFT DISTRACTION ALERT!", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

					state.data = 2

			elif (leftEDR-rightEDR) > DISTRACTION_THRES: # Looking at right direction
				COUNTER_DISTRACTION_RIGHT += 1
				COUNTER_DISTRACTION_LEFT   = 0

				if COUNTER_DISTRACTION_RIGHT >= DISTRACTION_CONSEC_FRAMES:
					cv2.putText(frame, "RIGHT DISTRACTION ALERT!", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

					state.data = 3

			else:
				state.data = 0

		# draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame COUNTER_DROWSINESS

		cv2.putText(frame, "Left: {:1.2f}".format(leftEDR), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "Right: {:1.2f}".format(rightEDR), (300, 60),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 90),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	state_publisher.publish(state)
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
