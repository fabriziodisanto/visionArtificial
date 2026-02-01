import cv2

image = cv2.imread(r"C:\Users\User\Documents\ORNELLA\procesamiento\visionArtificial\machine_tp\shapes.tp\estrella_tp\WhatsApp Image 2025-09-10 at 11.21.23.jpeg")

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2)


# Invert the image so the area of the UAV is filled with 1's. This is necessary since
# cv::findContours describes the boundary of areas consisting of 1's.
inv = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

cv2.imshow('image', bin)
cv2.imshow('inv', inv)
cv2.waitKey(0)