import cv2 as cv
import numpy as np

# ejercicio 4
# Generar una imagen binaria con dos umbrales con inRange, para segmentar un objeto por su color en el espacio HSV
image = cv.imread('static/images/messi.jpg')
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])
# Threshold the HSV image to get only blue colors

cv.imshow("inRange", cv.inRange(hsv, lower_blue, upper_blue))
cv.waitKey()