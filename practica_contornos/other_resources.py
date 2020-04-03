import cv2 as cv
import numpy as np

image = cv.imread('../static/images/rayo.jpg')
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
ret1, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)


def rectangle():
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv.imshow("rayo", image)


def convex_hull():
    hull = []
    for cnt in contours:
        hull.append(cv.convexHull(cnt, False))
    cv.drawContours(image, hull, -1, (255, 0, 0), 3)
    cv.imshow("rayo", image)


def min_area_rect():
    for cnt in contours:
        rect = cv.minAreaRect(cnt)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, (0, 0, 255), 2)    # cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv.imshow("rayo", image)


def min_circle():
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv.circle(image, center, radius, (0, 255, 0), 2)
    cv.imshow("rayo", image)

# ACA DESCOMENTEN EL METODO QUE QUIEREN QUE CORRA

# rectangle()

# convex_hull()

# min_area_rect()

# min_circle()


cv.waitKey()
