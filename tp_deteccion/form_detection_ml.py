import cv2
import numpy as np
import math

from contour import get_contours, filter_contours_by_area, get_bounding_rect
from frame_editor import apply_color_convertion, threshold, denoise, draw_contours
from trackbar import create_trackbar, get_trackbar_value
from ml_model import train_model, int_to_label

# BGR
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

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
        _, frame = cap.read()
        
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
            
            mom = cv2.moments(cont)
            hu_moments = cv2.HuMoments(mom)
            
            for i in range(0, 7):
                if (hu_moments[i] != 0):
                    hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))
            
            sample = np.array(hu_moments, dtype=np.float32).reshape(1, -1)
            predict = model.predict(sample)[0]
            
            label = int_to_label(predict)

            color = COLOR_BLUE if label == '5-point-star' else COLOR_RED if label == 'rectangle' else COLOR_GREEN
            draw_contours(frame=frame, contours=[cont], color=color, thickness=3)
            
            x, y, _, __ = get_bounding_rect(cont)
            cv2.putText(frame, label, (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Window', frame)
        cv2.imshow('Window debug', frame_denoised)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


main()
