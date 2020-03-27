import cv2 as cv
import numpy as np

alpha_slider_max = 100
betha_slider_max = 100


# def on_trackbar(val):
#     image = cv.imread('boca2000.jpg')
#     gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
#     ret2, thresh1 = cv.threshold(gray, val, 255, cv.THRESH_BINARY)
#     cv.imshow("Boca", thresh1)
#
#
# def on_trackbar2(val):
#     image2 = cv.imread('boca2000.jpg')
#     gray = cv.cvtColor(image2, cv.COLOR_RGB2GRAY)
#     ret2, thresh2 = cv.threshold(gray, val, 255, cv.THRESH_BINARY_INV)
#     cv.imshow("Boca2", thresh2)
#
#
# cv.namedWindow('Boca')
# cv.namedWindow('Boca2')
# cv.createTrackbar('Trackbar', 'Boca', 0, alpha_slider_max, on_trackbar)
# cv.createTrackbar('Trackbar2', 'Boca2', 0, betha_slider_max, on_trackbar2)
# # Show some stuff
# on_trackbar(0)
# on_trackbar2(0)
# # Wait until user press some key
# cv.waitKey()

# ejercicio 2
# gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
# ret1, thresh1 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
# cv.imshow("OTSU", thresh1)
# ret2, thresh2 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_TRIANGLE)
# cv.imshow("TRIANGLE", thresh2)
# cv.waitKey()


alpha_slider_max = 125

def adaptive_mean(val):
    image = cv.imread('boca2000.jpg')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    adapt = cv.adaptiveThreshold(gray, alpha_slider_max, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, val, 0)
    cv.imshow("Mean", adapt)


def adaptive_gaussian(val):
    image = cv.imread('boca2000.jpg')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    adapt = cv.adaptiveThreshold(gray, alpha_slider_max, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, val, 0)
    cv.imshow("Gaussian", adapt)


cv.namedWindow('Mean')
cv.createTrackbar('Trackbar', 'Mean', 0, alpha_slider_max, adaptive_mean)
cv.namedWindow('Gaussian')
cv.createTrackbar('Trackbar', 'Gaussian', 0, alpha_slider_max, adaptive_gaussian)

# Show some stuff
adaptive_mean(3)
adaptive_gaussian(3)
# Wait until user press some key
cv.waitKey()



# ejercicio 4
# image = cv.imread('boca2000.jpg')
# hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
# # define range of blue color in HSV
# lower_blue = np.array([110, 50, 50])
# upper_blue = np.array([130, 255, 255])
# # Threshold the HSV image to get only blue colors
#
# cv.imshow("inRange", cv.inRange(hsv, lower_blue, upper_blue))
# cv.waitKey()