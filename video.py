import cv2

# iniciamos la capturadora con el nombre cap
cap = cv2.VideoCapture(0)

# nos metemos en este while loop, para que no corte la capturadora
while True:
    # metodo para que la camara "lea"
    # frame son las "imagenes" (frames) de cada milisegundo
    ret, frame = cap.read()
    # aca las espejamos para que se vean bien
    frame = cv2.flip(frame, 1)
    # aca las mostramos en una ventana
    cv2.imshow('img1', frame)
    # al presionar h (sin el bloc mayus activado) guardamos una foto
    # tomando el tick count como nombre para evitar repeticiones
    if cv2.waitKey(1) == ord('h'):
        ticks = str(cv2.getTickCount())
        cv2.imwrite(ticks + '.png', frame)
    # al presionar z salimos del loop
    if cv2.waitKey(1) == ord('z'):
        break

# apagamos la capturadora y cerramos las ventanas que se nos abrieron
cap.release()
cv2.destroyAllWindows()
