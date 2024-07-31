import cv2


def create_trackbar(trackbar_name, window_name, slider_max):
    cv2.createTrackbar(trackbar_name, window_name, 0, slider_max, on_trackbar)


def on_trackbar(val):
    pass


def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))


WINDOW_NAME = 'WINDOW'
COLOR_RED = (0, 0, 255)

TRACKBAR_X_NAME = 'X length'
TRACKBAR_X_SLIDER_MAX = 1200

TRACKBAR_Y_NAME = 'Y length'
TRACKBAR_Y_SLIDER_MAX = 900

RECTANGLE_STARTING_POINT = (50, 50)


def main():
    cv2.namedWindow(WINDOW_NAME)
    create_trackbar(TRACKBAR_X_NAME, WINDOW_NAME, TRACKBAR_X_SLIDER_MAX)
    create_trackbar(TRACKBAR_Y_NAME, WINDOW_NAME, TRACKBAR_Y_SLIDER_MAX)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        trackbar_x_value = get_trackbar_value(TRACKBAR_X_NAME, WINDOW_NAME)
        trackbar_y_value = get_trackbar_value(TRACKBAR_Y_NAME, WINDOW_NAME)

        rectangle_end_point = (RECTANGLE_STARTING_POINT[0] + trackbar_x_value,
                               RECTANGLE_STARTING_POINT[1] + trackbar_y_value)

        cv2.rectangle(frame,
                      RECTANGLE_STARTING_POINT,
                      rectangle_end_point,
                      COLOR_RED,
                      2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow(WINDOW_NAME, frame)

    cap.release()
    cv2.destroyAllWindows()


main()