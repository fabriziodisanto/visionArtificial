import cv2


def denoise(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


def apply_color_convertion(frame, color):
    return cv2.cvtColor(frame, color)


def adaptive_threshold(frame, slider_max, adaptative, binary, trackbar_value):
    return cv2.adaptiveThreshold(frame, slider_max, adaptative, binary, trackbar_value, 1)

