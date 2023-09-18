import cv2
import numpy as np


img = cv2.imread("../static/images/levadura.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray_img, 48, 255, cv2.THRESH_BINARY)
cv2.imshow('thresh', thresh)
cv2.waitKey(0)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
# closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

cv2.imshow('opening', opening)
cv2.waitKey(0)

# Sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imshow('Sure bg', sure_bg)
cv2.waitKey(0)

# Sure foreground
kernel = np.ones((5, 5), np.uint8)
sure_fg = cv2.erode(opening, kernel, iterations=10)
# sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imshow('Sure fg2', sure_fg)
cv2.waitKey(0)

# dist_transform = cv2.distanceTransform(opening, cv2.DIST_C, 5)
# _, sure_fg = cv2.threshold(dist_transform, 20, 255, 0)
# sure_fg = np.uint8(sure_fg)
# sure_fg = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, kernel, iterations=3)
# cv2.imshow('Sure fg', sure_fg)
# cv2.waitKey(0)

# Unknown
unknown = cv2.subtract(sure_bg, sure_fg)
cv2.imshow('Unknown', unknown)
cv2.waitKey(0)

# Markers
_, markers = cv2.connectedComponents(sure_fg)
# para que no categorize el background como unknown
markers = markers + 10

markers[unknown == 255] = 0
markers = cv2.watershed(img, markers)

# Coloreamos los bordes de amarillo
img[markers == -1] = [0, 255, 255]

cv2.imshow('Resultado', img)
cv2.waitKey(0)
