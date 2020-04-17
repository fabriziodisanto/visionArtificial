import cv2


def get_contours(frame, mode, method):
    contours, hierarchy = cv2.findContours(frame, mode, method)
    return contours


def get_biggest_contour(contours):
    max_cnt = contours[0]
    for cnt in contours:
        if cv2.contourArea(cnt) > cv2.contourArea(max_cnt):
            max_cnt = cnt
    return max_cnt
