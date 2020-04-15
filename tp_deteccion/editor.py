import cv2

alpha_slider_max = 100

def get_binary(frame, color, val=0):
    gray = cv2.cvtColor(frame, color)
    ret1, thresh1 = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)
    cv2.imshow("Binary", thresh1)


cv2.namedWindow('Binary')
cv2.createTrackbar('Trackbar', 'Binary', 0, alpha_slider_max, get_binary)