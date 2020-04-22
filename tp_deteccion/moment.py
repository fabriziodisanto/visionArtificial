from math import copysign, log10
import cv2


def get_hu_moments(contour):
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    for i in range(len(hu_moments)):
        hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
    return hu_moments


# def compare_hu_moments(hu_moments, saved_hu_moments, max_diff):
#     for moments in saved_hu_moments:
#         if cv2.matchShapes(hu_moments, moments, cv2.CONTOURS_MATCH_I2, 0) < max_diff:
#             return True
#     return False


# def save_moment(hu_moments, file_name):
#     file = open(file_name, "a+")
#     hu_moments_str = generate_hu_moments_str(hu_moments=hu_moments)
#     file.write(hu_moments_str)
#     file.close()
#
#
# def generate_hu_moments_str(hu_moments):
#     for i in range(len(hu_moments) - 1):
#         string = string + str(hu_moments[i]) + ", "
#     return string + str(hu_moments[len(hu_moments) - 1])
#
#
# def generate_hu_moments_array(string):
#     # hu_moments = []
#     # for moment in string.split(', '):
#     #     hu_moments.append(float(moment))
#     # return hu_moments
#
#
# def load_hu_moments(file_name):
#     file = open(file_name, "r+")
#     saved_hu_moments_str = file.read()
#     saved_hu_moments = generate_hu_moments_array(string=saved_hu_moments_str)
#     print(saved_hu_moments)
#     # file.close()
