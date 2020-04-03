# Ejercicios con la cámara, en tiempo real,
# mostrando siempre en una ventana la imagen de la cámara.

# Los ejercicios fueron hechos con una foto precargada
# debido a que tuve problemas con mi camara

# Ejercicio 1
# Al pulsar una tecla, obtener la plantilla recortando
# un cuadrado de 100 x 100 píxeles en el centro de la imagen
# y mostrarla en una ventana aparte

# Ejercicio 2
# Aplicar matchTemplate con la plantilla
# sobre la imagen de la cámara y mostrar la
# imagen generada.  Usar teclas para cambiar el método.

# Ejercicio 3
# Buscar la posición del macheo con minMaxLoc() y dibujar
# un recuadro de 100 x 100 sobre la imagen original, señalando la detección

import cv2

# si van a copiar este codigo para probarlo,
# cambien la foto que levantan con el cv2,
# pongan una que ustedes tengan
img = cv2.imread("static/images/boca2000.jpg")
(h, w, d) = img.shape
roi = img[int(h/2-50):int(h/2+50), int(w/2-50):int(w/2+50)]


def template_matching(img, roi, method, name):
    template = cv2.matchTemplate(img, roi, method)
    cv2.imshow(name, template)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template)
    if method == cv2.TM_SQDIFF_NORMED:
        center = min_loc
    else:
        center = max_loc
    cv2.rectangle(img, (center[0], center[1]),
                  (center[0] + 100, center[1] + 100), (0, 0, 255), 2)


while True:
    if roi is not None and img is not None:
        cv2.imshow("roi", roi)
        cv2.imshow("img", img)
        if cv2.waitKey(1) == ord('q'):
            template_matching(img, roi, cv2.TM_CCOEFF_NORMED, "TM_CCOEFF_NORMED")
        if cv2.waitKey(1) == ord('w'):
            template_matching(img, roi, cv2.TM_CCORR_NORMED, "TM_CCORR_NORMED")
        if cv2.waitKey(1) == ord('e'):
            template_matching(img, roi, cv2.TM_SQDIFF_NORMED, "TM_SQDIFF_NORMED")
        if cv2.waitKey(1) == ord('m'):
            break

cv2.destroyAllWindows()