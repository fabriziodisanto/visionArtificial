import cv2


def denoise(frame, method, radius):
    kernel = cv2.getStructuringElement(method, (radius, radius))
    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


def apply_color_convertion(frame, color):
    return cv2.cvtColor(frame, color)


def adaptive_threshold(frame, slider_max, adaptative, binary, trackbar_value):
    return cv2.adaptiveThreshold(frame, slider_max, adaptative, binary, trackbar_value, 0)


def draw_contours(frame, contours, color, thickness):
    # -1 for all contours
    cv2.drawContours(frame, contours, -1, color, thickness)
    return frame

