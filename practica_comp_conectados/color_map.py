import cv2
from utils import get_connected_components

image = cv2.imread('../static/images/patente.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def binary():
    map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    cv2.imshow("map", map)
    _, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    get_connected_components(thresh1, 8, image)
    cv2.waitKey()


def normal():
    cv2.imshow("normal", image)
    map = cv2.applyColorMap(image, cv2.COLORMAP_JET)
    cv2.imshow("messi", map)
    cv2.waitKey()


binary()

# normal()


# The most popular method is cv2.connectedComponentsWithStats which returns the following information:
# - The bounding box of the connected component
# - The area (in pixels) of the component
# - The centroid/center (x, y)-coordinates of the component

# cv2.connectedComponents, is the same as the second, only it does not return the above statistical information

# def main():
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     cap = cv2.VideoCapture(0)
#     while True:
#         _, frame = cap.read()
#         frame_flipped = cv2.flip(frame, 1)
#         gray = cv2.cv2tColor(frame_flipped, cv2.COLOR_BGR2GRAY)
#         cv2.imshow('frame', gray)
#
#         if cv2.waitKey(1) == ord("q"):
#             _, thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#             cv2.imshow("Negreado", thresh1)
#             dilation = cv2.dilate(thresh1, kernel)
#             cv2.imshow("dilatado", dilation)
#             get_connected_components(dilation, 4)
#             print("w")
#
#         if cv2.waitKey(1) == ord("w"):
#             ret1, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#             map = cv2.applyColorMap(thresh1, cv2.COLORMAP_JET)
#             cv2.imshow("mapeado", map)
#
#         if cv2.waitKey(1) == ord('z'):
#             break
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# main()
