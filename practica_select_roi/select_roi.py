import cv2

image = cv2.imread('../static/images/patente.jpeg')


def select_roi():
    show_image = True
    while show_image:
        cv2.imshow('image', image)
        key = cv2.waitKey()
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            box = cv2.selectROI("Frame", image, fromCenter=False,
                                showCrosshair=True)
            print(box)
        if key == ord("q"):
            show_image = False


select_roi()