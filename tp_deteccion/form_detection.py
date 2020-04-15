import cv2
from tp_deteccion.frame_editor import apply_color_convertion, adaptive_threshold, denoise
from tp_deteccion.trackbar import create_trackbar, get_trackbar_value


def main():

    window_name = 'Window'
    trackbar_name = 'Trackbar'
    slider_max = 125
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(0)

    create_trackbar(trackbar_name, window_name, slider_max)
    while True:
        ret, frame = cap.read()
        gray_frame = apply_color_convertion(frame, cv2.COLOR_RGB2GRAY)
        trackbar_val = get_trackbar_value(trackbar_name, window_name)
        adapt_frame = adaptive_threshold(gray_frame, slider_max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                         trackbar_val)
        frame_denoised = denoise(adapt_frame)

        cv2.imshow('Window', frame_denoised)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


main()