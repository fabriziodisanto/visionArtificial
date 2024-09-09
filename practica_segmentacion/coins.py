import numpy as np
import cv2 as cv
 

def draw_labels(labels, img):
    normalized_labels = cv.normalize(labels, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Apply a colormap
    colored_labels = cv.applyColorMap(normalized_labels, cv.COLORMAP_JET)

    # Display the result
    cv.imshow('Colored Labels', colored_labels)
    cv.waitKey(0)


img = cv.imread("../static/images/water_coins.jpeg")

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

cv.imshow('thresh', thresh)
cv.waitKey(0)

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
 
cv.imshow('opening', opening)
cv.waitKey(0)

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=10)

cv.imshow('sure_bg', sure_bg)
cv.waitKey(0)

# Finding sure foreground area
# sure_fg = cv.erode(thresh, kernel, iterations=12)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
_, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)

cv.imshow('sure_fg', sure_fg)
cv.waitKey(0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)


cv.imshow('unknown', unknown)
cv.waitKey(0)

# Marker labelling
_, markers = cv.connectedComponents(sure_fg)
 
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
 
# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

draw_labels(markers, img)

cv.imshow('img', img)
cv.waitKey(0)