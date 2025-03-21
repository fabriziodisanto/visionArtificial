import cv2
import numpy as np
import math

from contour import get_contours, filter_contours_by_area, get_bounding_rect
from frame_editor import apply_color_convertion, threshold, denoise, draw_contours
from saved_contours import get_saved_contour
from trackbar import create_trackbar, get_trackbar_value
from ml_model import train_model, int_to_label

# BGR
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)

TRIANGLE_CONTOUR = get_saved_contour('./figures/triangle.png')
SQUARE_CONTOUR = get_saved_contour('./figures/square.png')
STAR_CONTOUR = get_saved_contour('./figures/star.jpeg')

SCORE_LIMIT = 0.15


def main():
    window_name = 'Window'
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(0)
    model = train_model()

    trackbar_thresh_name = 'Threshold'
    thresh_slider_max = 255
    create_trackbar(trackbar_thresh_name, window_name, thresh_slider_max)

    trackbar_kernel_name = 'Kernel denoise'
    contour_kernel_max = 10
    create_trackbar(trackbar_kernel_name, window_name, contour_kernel_max)

    trackbar_min_area_name = 'Min Area'
    contour_min_area_max = 10000
    create_trackbar(trackbar_min_area_name, window_name, contour_min_area_max)

    trackbar_max_area_name = 'Max Area'
    contour_max_area_max = 99999
    create_trackbar(trackbar_max_area_name, window_name, contour_max_area_max)


    while True:
        ret, frame = cap.read()
        
        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_BGR2GRAY)
        
        trackbar_thresh_val = get_trackbar_value(trackbar_name=trackbar_thresh_name, window_name=window_name)
        trackbar_min_area_val = get_trackbar_value(trackbar_name=trackbar_min_area_name, window_name=window_name)
        trackbar_max_area_val = get_trackbar_value(trackbar_name=trackbar_max_area_name, window_name=window_name)

        thresh_frame = threshold(frame=gray_frame, slider_max=thresh_slider_max,
                                 binary=cv2.THRESH_BINARY,
                                 trackbar_value=trackbar_thresh_val)

        frame_denoised = denoise(frame=thresh_frame, method=cv2.MORPH_ELLIPSE, radius=5)

        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        filtered_contours = filter_contours_by_area(contours=contours, min_area=trackbar_min_area_val,
                                                    max_area=trackbar_max_area_val)

        for cont in filtered_contours:
            triangle_score = cv2.matchShapes(cont, TRIANGLE_CONTOUR, cv2.CONTOURS_MATCH_I1, 0)
            square_score = cv2.matchShapes(cont, SQUARE_CONTOUR, cv2.CONTOURS_MATCH_I1, 0)
            star_score = cv2.matchShapes(cont, STAR_CONTOUR, cv2.CONTOURS_MATCH_I1, 0)
            min_score = min(triangle_score, square_score, star_score)

            x, y, w, h = get_bounding_rect(cont)

            if min_score > SCORE_LIMIT:
                draw_contours(frame=frame, contours=[cont], color=COLOR_RED, thickness=3)
                cv2.putText(frame, "Unknown", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)
            else:
                draw_contours(frame=frame, contours=[cont], color=COLOR_GREEN, thickness=3)
                if min_score == triangle_score:
                    cv2.putText(frame, "Triangle", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)
                elif min_score == square_score:
                    cv2.putText(frame, "Square", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)
                else:
                    # MIN SCORE == STAR
                    cv2.putText(frame, "Star", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)

        cv2.imshow('Window', frame)
        cv2.imshow('Window debug', frame_denoised)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


main()
