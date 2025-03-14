import cv2

def create_trackbar(trackbar_name, window_name, slider_max):
    cv2.createTrackbar(trackbar_name, window_name, 1, slider_max, on_trackbar)

def on_trackbar(val):
    pass

def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))


WINDOW_NAME = 'WINDOW'

TRACKBAR_THRESH_NAME = 'Threshold'
TRACKBAR_THRESH_SLIDER_MAX = 255

TRACKBAR_KERNEL_NAME = 'Kernel size'
TRACKBAR_KERNEL_SLIDER_MAX = 10


def main():
    cv2.namedWindow(WINDOW_NAME)
    create_trackbar(TRACKBAR_THRESH_NAME, WINDOW_NAME, TRACKBAR_THRESH_SLIDER_MAX)
    create_trackbar(TRACKBAR_KERNEL_NAME, WINDOW_NAME, TRACKBAR_KERNEL_SLIDER_MAX)
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        cv2.imshow(WINDOW_NAME, frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('gray', gray)

        trackbar_thresh_value = get_trackbar_value(TRACKBAR_THRESH_NAME, WINDOW_NAME)
        _, thresh = cv2.threshold(gray, trackbar_thresh_value, 255, cv2.THRESH_BINARY)
        cv2.imshow('binary', thresh)

        kernel_size_value = get_trackbar_value(TRACKBAR_KERNEL_NAME, WINDOW_NAME)
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size_value, kernel_size_value))
        
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # cv2.imshow('opening', opening)

        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('closing', closing)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()


main()
