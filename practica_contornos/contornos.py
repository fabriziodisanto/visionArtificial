import cv2 as cv

image = cv.imread('../static/images/rayo.jpg')
gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
ret1, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
contours, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
cv.drawContours(image, contours, -1, (0, 0, 255), 3)
cv.imshow("Rayo", image)

cv.waitKey()



