from utiles.entrenando_tp import train_model
from utiles.labels_tp import int_to_label  
import numpy as np
import math
import cv2

model = train_model()

capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
else:
    print("Cámara iniciada. Presiona 'q' para salir.")


cv2.namedWindow('1. Imagen Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('2. Escala de Grises', cv2.WINDOW_NORMAL)
cv2.namedWindow('3. Imagen Binarizada', cv2.WINDOW_NORMAL)
cv2.namedWindow('4. Morfologia', cv2.WINDOW_NORMAL)
cv2.namedWindow('5. Contornos Detectados', cv2.WINDOW_NORMAL)
cv2.namedWindow('6. Resultado Final', cv2.WINDOW_NORMAL)


cv2.moveWindow('1. Imagen Original', 50, 50)
cv2.moveWindow('2. Escala de Grises', 350, 50)
cv2.moveWindow('3. Imagen Binarizada', 650, 50)
cv2.moveWindow('4. Morfologia', 50, 400)
cv2.moveWindow('5. Contornos Detectados', 350, 400)
cv2.moveWindow('6. Resultado Final', 650, 400)


def nothing(x):
    pass

cv2.createTrackbar('Int.', '3. Imagen Binarizada', 2, 20, nothing)  # valor que se resta al promedio local

while True:
    
    ret, frame = capture.read()
    if not ret:
        print("No se pudo capturar el fotograma. Reintentando...")
        continue
    
    frame = cv2.flip(frame, 1)

    cv2.imshow('1. Imagen Original', frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('2. Escala de Grises', gray)

    C = cv2.getTrackbarPos('Int.', '3. Imagen Binarizada')
    bin_img = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        67,  # tamaño del bloque (ventana)
        C
    )
    cv2.imshow('3. Imagen Binarizada', bin_img)

    kernel = np.ones((3,3), np.uint8)
    morphed = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('4. Morfologia', morphed)

    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = frame.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('5. Contornos Detectados', contour_image)

    result_frame = frame.copy()
    if contours:
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)

                # Momentos de Hu
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments)
                for i in range(0, 7):
                    val = hu_moments[i][0]
                    if val != 0:
                        hu_moments[i][0] = -1 * math.copysign(1.0, val) * math.log10(abs(val))
                hu_moments_reshaped = hu_moments.reshape(1, -1).astype(np.float32)

                # Predicción
                prediction = model.predict(hu_moments_reshaped)
                label_text = int_to_label(int(prediction[0]))  

                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(result_frame, f"{label_text}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(result_frame, "No se detectaron formas", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('6. Resultado Final', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


capture.release()
cv2.destroyAllWindows()
print("Programa terminado correctamente.")

