import imutils
import cv2

# print(cv2.__version__)

# # load the input image and show its dimensions, keeping in mind that
# # images are represented as a multi-dimensional NumPy array with
# # shape no. rows (height) x no. columns (width) x no. channels (depth)
image = cv2.imread("static/images/boca2000.jpg")
(h, w, d) = image.shape



# # Depth is the number of channels — in our case this is three since we’re working with 3 color channels: Blue, Green, and Red.

# print("width={}, height={}, depth={}".format(w, h, d))
# # display the image to our screen -- we will need to click the window
# # open by OpenCV and press a key on our keyboard to continue execution
cv2.imshow("Title", image)
cv2.waitKey(0)

# # access the RGB pixel located at x=50, y=100, keepind in mind that
# # OpenCV stores images in BGR order rather than RGB

# image[y,x]
(B, G, R) = image[100, 50]
print(B)
print(G)
print(R)
# print("R={}, G={}, B={}".format(R, G, B))
#
#
# # extract a 100x100 pixel square ROI (Region of Interest) from the
# # input image starting at x=320,y=60 at ending at x=420,y=160
# roi = image[100:200, 350:450]
# # image[startY:endY, startX:endX]
# cv2.imshow("Cara", roi)
# cv2.waitKey(0)
#
# # in a near future we will be able to detect faces
#
# resize the image to 200x200px, ignoring aspect ratio
# resized = cv2.resize(image, (200, 200))
# cv2.imshow("Fixed Resizing", resized)
# cv2.waitKey(0)
#
# # fixed resizing and distort aspect ratio so let's resize the width
# # to be 300px but compute the new height based on the aspect ratio
# r = 300.0 / w
# dim = (300, int(h * r))
# resized = cv2.resize(image, dim)
# cv2.imshow("Aspect Ratio Resize", resized)
# cv2.waitKey(0)
#
#
# # manually computing the aspect ratio can be a pain so let's use the
# # imutils library instead

resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)


# # let's rotate an image 45 degrees clockwise using OpenCV by first
# # computing the image center, then constructing the rotation matrix,
# # and then finally applying the affine warp
# center = (w // 2, h // 2)
# # use //  to perform integer math
# M = cv2.getRotationMatrix2D(center, -45, 1.0)
# # -45 == 315
# rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("OpenCV Rotation", rotated)
# cv2.waitKey(0)
#
# # rotation can also be easily accomplished via imutils with less code
# rotated = imutils.rotate(image, -45)
# cv2.imshow("Imutils Rotation", rotated)
# cv2.waitKey(0)
#
# # OpenCV doesn't "care" if our rotated image is clipped after rotation
# # so we can instead use another imutils convenience function to help
# # us out
# rotated = imutils.rotate_bound(image, 45)
# cv2.imshow("Imutils Bound Rotation", rotated)
# cv2.waitKey(0)
#
# # apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# # useful when reducing high frequency noise
# blurred = cv2.GaussianBlur(image, (11, 11), 0)
# cv2.imshow("Blurred", blurred)
# cv2.waitKey(0)
#
# blurred_2 = cv2.GaussianBlur(image, (5, 5), 0)
# cv2.imshow("Blurred2", blurred_2)
# cv2.waitKey(0)
#
# # drawing operations on images are performed in-place.
# # Therefore at the beginning of each code block,
# # we make a copy of the original image storing the copy as output.
# # We then proceed to draw on the image called output in-place
# # so we do not destroy our original image.
#
# # draw a 2px thick red rectangle surrounding the face
# output = image.copy()
# # cv2.rectangle(img, pt1, pt2, color, thickness)
# cv2.rectangle(output, (350, 100), (450, 200), (0, 0, 255), 2)
# cv2.imshow("Rectangle", output)
# cv2.waitKey(0)
#
# cv2.rectangle(output, (350, 100), (450, 200), (0, 0, 255), 8)
# cv2.imshow("Rectangle2", output)
# cv2.waitKey(0)
#
# output = image.copy()
# # cv2.line(img, pt1, pt2, color, thickness)
# cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
# cv2.imshow("Line", output)
# cv2.waitKey(0)
#
# output = image.copy()
# # cv2.circle(img, center, radius, color, thickness)
# cv2.circle(output, (400, 150), 50, (0, 255, 255), 2)
# cv2.imshow("Circle", output)
# cv2.waitKey(0)
#
# output = image.copy()
# # cv2.putText(img, text, pt, font, scale, color, thickness)
# cv2.putText(output, "OpenCV + Vision Artificial", (350, 500),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# cv2.imshow("Text", output)
# cv2.waitKey(0)








# # USAGE
# # python opencv_object_tracking.py
# # python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt
#
# # import the necessary packages
# from imutils.video import VideoStream
# from imutils.video import FPS
# import argparse
# import imutils
# import time
# import cv2
#
# # construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-v", "--video", type=str,
# 	help="path to input video file")
# ap.add_argument("-t", "--tracker", type=str, default="kcf",
# 	help="OpenCV object tracker type")
# args = vars(ap.parse_args())
#
# # extract the OpenCV version info
# (major, minor) = cv2.__version__.split(".")[:2]
#
# # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# # function to create our object tracker
# if int(major) == 3 and int(minor) < 3:
# 	tracker = cv2.Tracker_create(args["tracker"].upper())
#
# # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# # approrpiate object tracker constructor:
# else:
# 	# initialize a dictionary that maps strings to their corresponding
# 	# OpenCV object tracker implementations
# 	OPENCV_OBJECT_TRACKERS = {
# 		"csrt": cv2.TrackerCSRT_create,
# 		"kcf": cv2.TrackerKCF_create,
# 		"boosting": cv2.TrackerBoosting_create,
# 		"mil": cv2.TrackerMIL_create,
# 		"tld": cv2.TrackerTLD_create,
# 		"medianflow": cv2.TrackerMedianFlow_create,
# 		"mosse": cv2.TrackerMOSSE_create
# 	}
#
# 	# grab the appropriate object tracker using our dictionary of
# 	# OpenCV object tracker objects
# 	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
#
# # initialize the bounding box coordinates of the object we are going
# # to track
# initBB = None
#
# # if a video path was not supplied, grab the reference to the web cam
# if not args.get("video", False):
# 	print("[INFO] starting video stream...")
# 	vs = VideoStream(src=0).start()
# 	time.sleep(1.0)
#
# # otherwise, grab a reference to the video file
# else:
# 	vs = cv2.VideoCapture(args["video"])
#
# # initialize the FPS throughput estimator
# fps = None
#
# # loop over frames from the video stream
# while True:
# 	# grab the current frame, then handle if we are using a
# 	# VideoStream or VideoCapture object
# 	frame = vs.read()
# 	frame = frame[1] if args.get("video", False) else frame
#
# 	# check to see if we have reached the end of the stream
# 	if frame is None:
# 		break
#
# 	# resize the frame (so we can process it faster) and grab the
# 	# frame dimensions
# 	frame = imutils.resize(frame, width=500)
# 	(H, W) = frame.shape[:2]
#
# 	# check to see if we are currently tracking an object
# 	if initBB is not None:
# 		# grab the new bounding box coordinates of the object
# 		(success, box) = tracker.update(frame)
#
# 		# check to see if the tracking was a success
# 		if success:
# 			(x, y, w, h) = [int(v) for v in box]
# 			cv2.rectangle(frame, (x, y), (x + w, y + h),
# 				(0, 255, 0), 2)
#
# 		# update the FPS counter
# 		fps.update()
# 		fps.stop()
#
# 		# initialize the set of information we'll be displaying on
# 		# the frame
# 		info = [
# 			("Tracker", args["tracker"]),
# 			("Success", "Yes" if success else "No"),
# 			("FPS", "{:.2f}".format(fps.fps())),
# 		]
#
# 		# loop over the info tuples and draw them on our frame
# 		for (i, (k, v)) in enumerate(info):
# 			text = "{}: {}".format(k, v)
# 			cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
# 				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#
# 	# show the output frame
# 	cv2.imshow("Frame", frame)
# 	key = cv2.waitKey(1) & 0xFF
#
# 	# if the 's' key is selected, we are going to "select" a bounding
# 	# box to track
# 	if key == ord("s"):
# 		# select the bounding box of the object we want to track (make
# 		# sure you press ENTER or SPACE after selecting the ROI)
# 		initBB = cv2.selectROI("Frame", frame, fromCenter=False,
# 			showCrosshair=True)
#
# 		# start OpenCV object tracker using the supplied bounding box
# 		# coordinates, then start the FPS throughput estimator as well
# 		tracker.init(frame, initBB)
# 		fps = FPS().start()
#
# 	# if the `q` key was pressed, break from the loop
# 	elif key == ord("q"):
# 		break
#
# # if we are using a webcam, release the pointer
# if not args.get("video", False):
# 	vs.stop()
#
# # otherwise, release the file pointer
# else:
# 	vs.release()
#
# # close all windows
# cv2.destroyAllWindows()
