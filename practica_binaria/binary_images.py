import cv2 as cv

# ejercicio 1
# Generar una imagen binaria normal y otra invertida sobre la cámara, controlando el umbral con una barra deslizante
# maximo valor del slider
alpha_slider_max = 100

# muestra una imagen binaria normal y controla umbral con trackbar
def binary(val):
    image = cv.imread('../static/images/messi.jpg')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret1, thresh1 = cv.threshold(gray, val, 255, cv.THRESH_BINARY)
    cv.imshow("Binary", thresh1)

# muestra una imagen binaria invertida y controla umbral con trackbar
def binary_inv(val):
    image = cv.imread('../static/images/messi.jpg')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    ret2, thresh2 = cv.threshold(gray, val, 255, cv.THRESH_BINARY_INV)
    cv.imshow("BinaryInv", thresh2)

# crea el trackbar y lo asocia a la imagen Binary
cv.namedWindow('Binary')
cv.createTrackbar('Trackbar', 'Binary', 0, alpha_slider_max, binary)

# crea el trackbar y lo asocia a la imagen BinaryInv
cv.namedWindow('BinaryInv')
cv.createTrackbar('Trackbar', 'BinaryInv', 0, alpha_slider_max, binary_inv)

# Show some stuff
binary(0)
binary_inv(0)
# Wait until user press some key
cv.waitKey()
