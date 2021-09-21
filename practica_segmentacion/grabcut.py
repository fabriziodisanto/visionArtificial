import numpy as np
import cv2 as cv

img = cv.imread('/Users/juancruzpisani/projects/Austral/visionArtificial/static/images/messi.jpg')

# si usamos el metodo de GC_INIT_WITH_RECT no es necesario camara por eso hacemos una matriz de 0
mask = np.zeros(img.shape[:2], np.uint8)

# These are arrays used by the algorithm internally. You just create two np.float64 type zero arrays
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# usamos roi para agarrar el rect
rect = cv.selectROI("img", img, fromCenter=False, showCrosshair=True)
print(mask)

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 10, cv.GC_INIT_WITH_RECT)

print(mask)
# ????????????
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

print(mask2)

# ?????????????
img = img*mask2[:, :, np.newaxis]

cv.imshow("img", img)
cv.waitKey()

