import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture(0)


def find_homography():

    objp = np.zeros((4 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:4, 0:7].T.reshape(-1, 2)

    image_points = []
    object_points = []
    camera_matrix = None

    reference = None

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ret1, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, 1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            ret2, corners = cv2.findChessboardCorners(gray, (4, 7))
            if ret2:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                image_points.append(corners2)
                object_points.append(objp)
                ret, mtx, distortion_coeff, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points,
                                                                                         gray.shape[::-1], None, None)
                camera_matrix = mtx
                cv2.imshow('drawedCorners', cv2.drawChessboardCorners(frame, (4, 7), corners2, ret2))
                cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('w'):
            rt, reference = cap.read()
            cv2.imshow('reference', reference)

        # Display the resulting frame
        cv2.imshow('binary', cv2.flip(frame, 1))
        if cv2.waitKey(1) & 0xFF == ord('e'):
            image = reference
            image2 = imutils.rotate(frame, 90)

            # sift = cv2.xfeatures2d_SIFT().create()
            sift = cv2.xfeatures2d.SIFT_create()

            kp1, des1 = sift.detectAndCompute(image, None)
            kp2, des2 = sift.detectAndCompute(image2, None)

            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts)
                print("-----------homographyMatrix-----------")
                print(M)

                retval, rotations, translations, normals = cv2.decomposeHomographyMat(M, camera_matrix)
                print("-----------retval-----------")
                print(retval)
                print("-----------rotations-----------")
                print(rotations)
                print("-----------translations-----------")
                print(translations)
                print("-----------normals-----------")
                print(normals)
                break


find_homography()
