import cv2 as cv


def on_trackbar(val):
    pass


backSub = cv.createBackgroundSubtractorMOG2()

window_name = 'Background'
trackbar_name = 'Trackbar'
cv.namedWindow(window_name)
slider_max = 100
cv.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)

capture = cv.VideoCapture('../static/vtest.avi')
while True:
    ret, frame = capture.read()
    if frame is None:
        break

    trackbar_val = cv.getTrackbarPos(trackbar_name, window_name)/100

    subtracted = backSub.apply(frame, learningRate=trackbar_val)

    cv.imshow('Frame', frame)
    cv.imshow('Background', subtracted)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
