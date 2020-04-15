import cv2


def create_trackbar(trackbar_name, window_name, slider_max):
    cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)


def on_trackbar(val):
    pass


def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name) / 2) * 2 + 3
