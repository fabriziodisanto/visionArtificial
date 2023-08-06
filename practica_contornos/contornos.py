import cv2 as cv

image = cv.imread('../static/images/patente.jpeg')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
ret1, thresh1 = cv.threshold(gray, 80, 255, cv.THRESH_BINARY)
cv.imshow("bin", thresh1)

cv.waitKey()



contours, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# contours = [cnt1, cnt2, cnt3]
cv.drawContours(image, contours, -1, (100, 0, 100), 2)
cv.imshow("Contornos patente", image)

cv.waitKey()



