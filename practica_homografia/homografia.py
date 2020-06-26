import numpy as np
import cv2


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


points = []
# points = [(0, 240), (50, 150), (590, 150), (640, 240)]

#
# def main():
#     cv2.namedWindow('frame')
#     cv2.setMouseCallback('frame', on_click)
#     frame = cv2.imread('../static/images/tenis.jpg')
#     cv2.imshow('frame', frame)
#     global points
#     while len(points) <= 4:
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             if len(points) == 4:
#                 pts = np.array(points, dtype="float32")
#                 cv2.imshow('imagen', four_point_transform(frame, pts))
#                 cv2.waitKey(0)
#                 break
#     cv2.waitKey(0)
#

def calibrate_camera():
    CHECKBOARD = (4, 7)

    cap = cv2.VideoCapture(0)

    objp = np.zeros((CHECKBOARD[0] * CHECKBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKBOARD[0], 0:CHECKBOARD[1]].T.reshape(-1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    image_points = []
    object_points = []
    camera_matrix = None
    distortion_coeff = None

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('binary', cv2.flip(gray, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            ret2, corners = cv2.findChessboardCorners(gray, (CHECKBOARD[0], CHECKBOARD[1]))
            if ret2:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners2)
                object_points.append(objp)
                ret, camera_matrix, distortion_coeff, rotationvecs, translationvecs = \
                    cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)
                cv2.imshow('drawedCorners',
                           cv2.drawChessboardCorners(frame, (CHECKBOARD[0], CHECKBOARD[1]), corners2, ret2))
                print("--------------CAMERA MATRIX------------")
                print(camera_matrix)
                print("--------------DISTORTION COEFF------------")
                print(distortion_coeff)
                cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            (h, w, d) = frame.shape
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))
            return camera_matrix, distortion_coeff, new_camera_matrix


def mainVideo():
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('../static/videos/carsRt9_3.avi')
    camera_matrix = None
    distortion_coeff = None
    new_camera_matrix = None
    cv2.namedWindow('imagen normal')
    cv2.namedWindow('imagen distorsioned_frame')
    cv2.setMouseCallback('imagen distorsioned_frame', on_click)
    while True:
        while camera_matrix is None:
            camera_matrix, distortion_coeff, new_camera_matrix = calibrate_camera()
        global points
        ret, frame = cap.read()
        cv2.imshow('imagen normal', frame)
        distorsioned_frame = cv2.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)
        cv2.imshow('imagen distorsioned_frame', distorsioned_frame)
        if len(points) == 4:
            pts = np.array(points, dtype="float32")
            cv2.imshow('imagen transformada', four_point_transform(distorsioned_frame, pts))
        if cv2.waitKey(1) & 0xFF == ord('w'):
            break


def on_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global points
        points.append((x, y))


mainVideo()
