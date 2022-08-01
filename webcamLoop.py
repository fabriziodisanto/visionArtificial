import numpy as np
import cv2 as cv

# Instrucciones en consola
print("Pulse ESC para terminar.")

webcam = cv.VideoCapture(0) # webcam 0, podría ser 1, 2... para abrir otra si hay más de una
f = cv.getTickFrequency()   # tics del reloj por segundo

# Aquí se escribe el código de setup
#

while True:
    ret, imWebcam = webcam.read()
    cv.imshow('webcam', imWebcam)

    tInicial = cv.getTickCount()
    # Aquí se escribe el código para procesar la imagen imWebcam
    imGris = cv.cvtColor(imWebcam, cv.COLOR_BGR2GRAY)

    # Tiempo de procesamiento
    tFinal = cv.getTickCount()
    duracion = (tFinal - tInicial) / f  # Este valor se puede mostrar en consola o sobre la imagen

    # Aquí se escribe el código de visualización
    cv.imshow('blancoYNegro', imGris)
    #
    #

    # Lee el teclado y decide qué hacer con cada tecla
    tecla = cv.waitKey(30)  # espera 30 ms. El mínimo es 1 ms.
    # tecla == 0 si no se pulsó ninguna

    # tecla ESC para salir
    # ESC == 27 en ASCII
    if tecla == 27:
        break
    # aquí se pueden agregar else if y procesar otras teclas



cv.destroyAllWindows()
