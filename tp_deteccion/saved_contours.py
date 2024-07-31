import cv2
from contour import get_contours


def get_saved_contour(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    return get_contours(frame=image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[1]
