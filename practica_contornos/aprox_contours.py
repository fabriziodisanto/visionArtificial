import cv2 as cv


def aprox_contours(val):
    image = cv.imread('../static/images/rayo.jpg')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh1, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        aprox = cv.approxPolyDP(cnt, val, True)
        cv.drawContours(image, [aprox], -1, (0, 0, 255), 3)
    cv.imshow("Rayo", image)


cv.namedWindow('Rayo')
cv.createTrackbar('Trackbar', 'Rayo', 0, 50, aprox_contours)

aprox_contours(0)

cv.waitKey()
