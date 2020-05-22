import numpy as np
import cv2

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
            cv2.imshow('drawedCorners', cv2.drawChessboardCorners(frame, (CHECKBOARD[0], CHECKBOARD[1]), corners2, ret2))
            print("--------------CAMERA MATRIX------------")
            print(camera_matrix)
            print("--------------DISTORTION COEFF------------")
            print(distortion_coeff)
            cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        (h, w, d) = frame.shape
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeff, (w, h), 1, (w, h))
        print("--------------NEW CAMERA MATRIX------------")
        print(new_camera_matrix)
        print("--------------ROI------------")
        print(roi)
        cv2.imwrite('original.jpg', frame)
        distorsioned_frame = cv2.undistort(frame, camera_matrix, distortion_coeff, None, new_camera_matrix)
        cv2.imwrite('distorsioned.jpg', distorsioned_frame)
        break
