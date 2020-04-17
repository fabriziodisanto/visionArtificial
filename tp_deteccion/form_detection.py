import cv2

from tp_deteccion.contour import get_contours, get_biggest_contour
from tp_deteccion.frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours
from tp_deteccion.moment import get_hu_moments, compare_hu_moments
from tp_deteccion.trackbar import create_trackbar, get_trackbar_value


def main():

    window_name = 'Window'
    trackbar_name = 'Trackbar'
    slider_max = 151
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(0)
    hu_moments = None
    color_white = (255, 255, 255)
    create_trackbar(trackbar_name, window_name, slider_max)
    # saved_hu_moments = load_hu_moments(file_name="hu_moments.txt")
    saved_hu_moments = []

    while True:
        ret, frame = cap.read()
        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_RGB2GRAY)
        trackbar_val = get_trackbar_value(trackbar_name=trackbar_name, window_name=window_name)
        adapt_frame = adaptive_threshold(frame=gray_frame, slider_max=slider_max,
                                         adaptative=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         binary=cv2.THRESH_BINARY,
                                         trackbar_value=trackbar_val)
        frame_denoised = denoise(frame=adapt_frame, method=cv2.MORPH_ELLIPSE, radius=10)
        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            biggest_contour = get_biggest_contour(contours=contours)
            hu_moments = get_hu_moments(contour=biggest_contour)
            if compare_hu_moments(hu_moments=hu_moments, saved_hu_moments=saved_hu_moments, max_diff=1):
                draw_contours(frame=frame_denoised, contours=biggest_contour, color=color_white, thickness=20)
            draw_contours(frame=frame_denoised, contours=biggest_contour, color=color_white, thickness=3)

        cv2.imshow('Window', frame_denoised)
        if cv2.waitKey(1) & 0xFF == ord('k'):
            if hu_moments is not None and len(hu_moments) > 0:
                # save_moment(hu_moments=hu_moments, file_name="hu_moments.txt")
                saved_hu_moments.append(hu_moments)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()
