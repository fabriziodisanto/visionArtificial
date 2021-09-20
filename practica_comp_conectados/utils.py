import cv2
import numpy as np


def analyze_components(num_labels, labels, stats, centroids, image):
    grey = image.copy()
    mask = np.zeros(cv2.cvtColor(grey, cv2.COLOR_RGB2GRAY).shape, dtype="uint8")
    # loop over the number of unique connected component labels
    for i in range(1, num_labels):
        # extract the connected component statistics for the current
        # label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        print("x: {x}\ny: {y}\nw: {w}\nh: {h}\narea: {area}\ncentroid: {cX},{cY}".format(x=x, y=y, w=w, h=h, area=area, cX=cX, cY=cY))
        # draw_component(cX, cY, h, i, image, labels, w, x, y)
        mask = add_mask(w, h, area, labels, i, mask)
    cv2.imshow("image", image)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)


def add_mask(w, h, area, labels, i, mask):
    keepWidth = w > 5 and w < 100
    keepHeight = h > 5 and h < 100
    keepArea = area > 200 and area < 1500
    if all((keepWidth, keepHeight, keepArea)):
        # construct a mask for the current connected component and
        # then take the bitwise OR with the mask
        print("[INFO] keeping connected component '{}'".format(i))
        componentMask = (labels == i).astype("uint8") * 255
        return cv2.bitwise_or(mask, componentMask)
    return mask


def draw_component(cX, cY, h, i, image, labels, w, x, y):
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
    # construct a mask for the current connected component by
    # finding a pixels in the labels array that have the current
    # connected component ID
    componentMask = (labels == i).astype("uint8") * 255
    # show our output image and connected component mask
    cv2.imshow("Output", output)
    cv2.imshow("Connected Component", componentMask)
    cv2.waitKey(0)


def get_connected_components(thresh, connectivity, image):
    (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(image=thresh,
                                                                              connectivity=connectivity,
                                                                              ltype=cv2.CV_32S)
    analyze_components(num_labels, labels, stats, centroids, image)
