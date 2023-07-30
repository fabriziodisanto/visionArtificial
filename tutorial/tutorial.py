import imutils
import cv2

# print(cv2.__version__)

# # load the input image and show its dimensions, keeping in mind that
# # images are represented as a multi-dimensional NumPy array with
# # shape no. rows (height) x no. columns (width) x no. channels (depth)
image = cv2.imread("../static/images/boca2000.jpg")
(h, w, d) = image.shape

# # Depth is the number of channels — in our case this is three since we’re working with 3 color channels: Blue, Green, and Red.

print("width={}, height={}, depth={}".format(w, h, d))
# # display the image to our screen -- we will need to click the window
# # open by OpenCV and press a key on our keyboard to continue execution
cv2.imshow("Title", image)
cv2.waitKey(0)

# # access the RGB pixel located at x=50, y=100, keepind in mind that
# # OpenCV stores images in BGR order rather than RGB

# image[y,x]
(B, G, R) = image[100, 50]
print("R={}, G={}, B={}".format(R, G, B))


# extract a 100x100 pixel square ROI (Region of Interest) from the
# input image starting at x=320,y=60 at ending at x=420,y=160
roi = image[100:200, 350:450]
# image[startY:endY, startX:endX]
cv2.imshow("Cara", roi)
cv2.waitKey(0)

# resize the image to 200x200px, ignoring aspect ratio
resized = cv2.resize(image, (200, 200))
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)

# # fixed resizing and distort aspect ratio so let's resize the width
# # to be 300px but compute the new height based on the aspect ratio
r = 300.0 / w
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resize", resized)
cv2.waitKey(0)

# # manually computing the aspect ratio can be a pain so let's use the
# # imutils library instead
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)

# let's rotate an image 45 degrees clockwise
rotated = imutils.rotate(image, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

# OpenCV doesn't "care" if our rotated image is clipped after rotation
# so we can instead use another imutils convenience function to help
# us out
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# apply a Gaussian blur with a 11x11 kernel to the image to smooth it,
# useful when reducing high frequency noise
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

blurred_2 = cv2.GaussianBlur(image, (5, 5), 0)
cv2.imshow("Blurred2", blurred_2)
cv2.waitKey(0)

# drawing operations on images are performed in-place.
# Therefore at the beginning of each code block,
# we make a copy of the original image storing the copy as output.
# We then proceed to draw on the image called output in-place
# so we do not destroy our original image.

# # draw a 2px thick red rectangle surrounding the face
output = image.copy()
# cv2.rectangle(img, pt1, pt2, color, thickness)
cv2.rectangle(output, (350, 100), (450, 200), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

cv2.rectangle(output, (350, 100), (450, 200), (0, 0, 255), 8)
cv2.imshow("Rectangle2", output)
cv2.waitKey(0)

output = image.copy()
# cv2.line(img, pt1, pt2, color, thickness)
cv2.line(output, (60, 20), (400, 200), (0, 0, 255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

output = image.copy()
# cv2.circle(img, center, radius, color, thickness)
cv2.circle(output, (400, 150), 50, (0, 255, 255), 2)
cv2.imshow("Circle", output)
cv2.waitKey(0)

output = image.copy()
# cv2.putText(img, text, pt, font, scale, color, thickness)
cv2.putText(output, "OpenCV + Vision Artificial", (350, 500),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)
