import cv2


def get_video():
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, 1)

        # Display the resulting frame
        cv2.imshow('frame', cv2.flip(frame, 1))



def end_video():
    cap = cv2.VideoCapture(0)
    cap.release()
    cv2.destroyAllWindows()