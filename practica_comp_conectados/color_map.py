import cv2 as cv2

# image = cv.imread('../static/images/messi.jpg')
# gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#
#
# def binary():
#     # ret1, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
#     # map = cv.applyColorMap(thresh1, cv.COLORMAP_JET)
#     # cv.imshow("messi", cv.connectedComponentsWithStats(thresh1))
#     cv.imshow("messi", cv.connectedComponents(gray))
#     cv.waitKey()
#
#
# def normal():
#     map = cv.applyColorMap(image, cv.COLORMAP_JET)
#     cv.imshow("messi", map)
#     cv.waitKey()

# ACA DESCOMENTEN EL METODO QUE QUIEREN CORRER

# binary()

# normal()


def main():
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame_flipped = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', gray)

        if cv2.waitKey(1) == ord("q"):
            ret, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow("Negreado", thresh1)
            dilation = cv2.dilate(thresh1, kernel)
            cv2.imshow("dilatado", dilation)

            [retval, labels, stats, centroids] = cv2.connectedComponentsWithStats(dilation)
            cv2.imshow("conectado", labels)
            print("w")

        if cv2.waitKey(1) == ord("w"):
            ret1, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            map = cv2.applyColorMap(thresh1, cv2.COLORMAP_JET)
            cv2.imshow("mapeado", map)

        if cv2.waitKey(1) == ord('z'):
            break
    cap.release()
    cv2.destroyAllWindows()


main()