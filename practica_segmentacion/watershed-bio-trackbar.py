import cv2
import numpy as np


def create_trackbar(trackbar_name, window_name, slider_max):
    cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)


def on_trackbar(val):
    pass


def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))

loop = True

trackbar_window_name = 'Trackbars'
cv2.namedWindow('1')
thresh_name = 'OG Threshold'
thresh_slider_max = 255
create_trackbar(thresh_name, trackbar_window_name, thresh_slider_max)

opening_name = 'opening iterations'
thresh_slider_max = 25
create_trackbar(opening_name, trackbar_window_name, thresh_slider_max)

dilate_name = 'dilate iterations'
thresh_slider_max = 25
create_trackbar(dilate_name, trackbar_window_name, thresh_slider_max)

erode_name = 'erode iterations'
thresh_slider_max = 25
create_trackbar(erode_name, trackbar_window_name, thresh_slider_max)

while loop:
    thresh_val = get_trackbar_value(trackbar_name=thresh_name, window_name=trackbar_window_name)
    opening_val = get_trackbar_value(trackbar_name=opening_name, window_name=trackbar_window_name)
    dilate_val = get_trackbar_value(trackbar_name=dilate_name, window_name=trackbar_window_name)
    erode_val = get_trackbar_value(trackbar_name=erode_name, window_name=trackbar_window_name)

    img = cv2.imread("../static/images/levadura.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray_img, thresh_val, 255, cv2.THRESH_BINARY)
    cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=opening_val)
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

    cv2.imshow('opening', opening)
    # cv2.waitKey(0)

    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=dilate_val)
    cv2.imshow('Sure bg', sure_bg)
    # cv2.waitKey(0)

    # Sure foreground
    sure_fg = cv2.erode(opening, kernel, iterations=erode_val)
    cv2.imshow('Sure fg2', sure_fg)
    # cv2.waitKey(0)

    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_C, 5)
    # _, sure_fg = cv2.threshold(dist_transform, 20, 255, 0)
    # sure_fg = np.uint8(sure_fg)
    # sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=3)
    # cv2.imshow('Sure fg', sure_fg)
    # cv2.waitKey(0)

    # Unknown
    unknown = cv2.subtract(sure_bg, sure_fg)
    cv2.imshow('Unknown', unknown)
    # cv2.waitKey(0)

    # Markers
    _, markers = cv2.connectedComponents(sure_fg)
    # para que no categorize el background como unknown
    markers = markers + 10

    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # Coloreamos los bordes de rojo
    img[markers == -1] = [0, 0, 255]

    cv2.imshow('Resultado', img)
    # cv2.waitKey(0)
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

