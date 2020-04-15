import cv2
import time


def main():
    cap = cv2.VideoCapture(0)
    time.sleep(2)

    print(cap)

    ret, frame = cap.read()
    print(ret)
    # while True:
    #     # Capture frame-by-frame
    #     ret, frame = cap.read()
    #
    #     # Our operations on the frame come here
    #     # gray = cv2.cvtColor(frame, 1)
    #
    #     if ret == True:
    #         # Display the resulting frame
    #         cv2.imshow('frame', frame)
    #         if cv2.waitKey(0) & 0xFF == ord('q'):
    #             break

    cap.release()


main()
