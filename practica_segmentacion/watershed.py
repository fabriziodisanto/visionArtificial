import numpy as np
import cv2 as cv

img = cv.imread('/Users/juancruzpisani/projects/Austral/visionArtificial/static/images/water_coins.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

cv.imshow("img", thresh)

cv.waitKey()
# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=2)

# sure background area
sure_bg = cv.dilate(closing, kernel, iterations=3)

cv.imshow("sure_bg", sure_bg)

cv.waitKey()
# Finding sure foreground area
sure_fg = cv.erode(closing, kernel, iterations=3)

# SE DEBERIA USAR EL DISTANCE TRANSFORM ANTES Q EL ERODE
# PQ LOS OBJETOS ESTAN TODOS PEGADOS PERO DE BAJA PQ NOSE SI HAY Q EXPLCIAR ESO
# dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
# ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)


cv.imshow("sure_fg", sure_fg)

cv.waitKey()
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labelling
_, markers = cv.connectedComponents(sure_fg)
print(markers)
print("------------------")
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)

print(img)
print(markers)

img[markers == -1] = [255, 0, 0]

cv.imshow("img", img)

cv.waitKey()
