import cv2 as cv

image = cv.imread('../static/images/messi.jpg')
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)


def binary():
    ret1, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    map = cv.applyColorMap(thresh1, cv.COLORMAP_JET)
    cv.imshow("messi", map)
    cv.waitKey()


def normal():
    map = cv.applyColorMap(image, cv.COLORMAP_JET)
    cv.imshow("messi", map)
    cv.waitKey()

# ACA DESCOMENTEN EL METODO QUE QUIEREN CORRER

# binary()

# normal()
