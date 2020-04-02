import cv2 as cv

alpha_slider_max = 125

# ejercicio 3
# Generar una imagen binaria normal sobre la cámara, usando ambos métodos de umbral adaptativo, controlando el tamaño de bloque con una barra deslizante

def adaptive_mean(val):
    image = cv.imread('../static/images/messi.jpg')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # primer metodo adaptativo MEAN
    adapt = cv.adaptiveThreshold(gray, alpha_slider_max, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, val, 0)
    cv.imshow("Mean", adapt)


def adaptive_gaussian(val):
    image = cv.imread('../static/images/messi.jpg')
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    # segundo metodo adaptativo GAUSSIAN
    adapt = cv.adaptiveThreshold(gray, alpha_slider_max, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, val, 0)
    cv.imshow("Gaussian", adapt)


cv.namedWindow('Mean')
cv.createTrackbar('Trackbar', 'Mean', 0, alpha_slider_max, adaptive_mean)
cv.namedWindow('Gaussian')
cv.createTrackbar('Trackbar', 'Gaussian', 0, alpha_slider_max, adaptive_gaussian)

# Show some stuff
adaptive_mean(3)
adaptive_gaussian(3)
# Wait until user press some key
cv.waitKey()
