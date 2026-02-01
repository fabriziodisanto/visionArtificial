import cv2
import numpy as np
import math
from utiles.entrenando_tp import train_model

# --- 1. ENTRENAR EL MODELO UNA SOLA VEZ ---
print("Entrenando el modelo, por favor espera...")
model = train_model()
print("¡Modelo entrenado y listo!")

# Diccionario para traducir la predicción del modelo a texto
labels = {1: 'estrella_tp', 2: 'rectangulo_tp', 3: 'triangulo_tp'}

# --- 2. INICIAR LA CÁMARA ---
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
else:
    print("Cámara iniciada. Presiona 'q' para salir.")

# Crear ventanas
cv2.namedWindow('1. Imagen Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('2. Escala de Grises', cv2.WINDOW_NORMAL)
cv2.namedWindow('3. Imagen Binarizada', cv2.WINDOW_NORMAL)
cv2.namedWindow('4. Contornos Detectados', cv2.WINDOW_NORMAL)
cv2.namedWindow('5. Morfologia', cv2.WINDOW_NORMAL)
cv2.namedWindow('6. Resultado Final', cv2.WINDOW_NORMAL)

# Posicionar ventanas
cv2.moveWindow('1. Imagen Original', 50, 50)
cv2.moveWindow('2. Escala de Grises', 350, 50)
cv2.moveWindow('3. Imagen Binarizada', 650, 50)
cv2.moveWindow('4. Contornos Detectados', 50, 400)
cv2.moveWindow('5. Morfologia', 350, 400)
cv2.moveWindow('6. Resultado Final', 650, 400)

# --- TRACKBAR PARA AJUSTAR ILUMINACIÓN (parámetro C de adaptiveThreshold) ---
def nothing(x):
    pass

cv2.createTrackbar('C', '3. Imagen Binarizada', 2, 20, nothing)  # valor que se resta al promedio local

while True:
    # Capturamos un fotograma
    ret, frame = capture.read()
    if not ret:
        print("No se pudo capturar el fotograma. Reintentando...")
        continue
    
    # Volteamos horizontalmente
    frame = cv2.flip(frame, 1)

    # --- ETAPA 1: Imagen original
    cv2.imshow('1. Imagen Original', frame)

    # --- ETAPA 2: Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('2. Escala de Grises', gray)

    # --- ETAPA 3: Adaptive Threshold
    C = cv2.getTrackbarPos('C', '3. Imagen Binarizada')
    bin_img = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        67,  # tamaño del bloque (ventana)
        C
    )
    cv2.imshow('3. Imagen Binarizada', bin_img)

    # --- ETAPA 4: Contornos
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = frame.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('4. Contornos Detectados', contour_image)

    # --- ETAPA 5: Operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('5. Morfologia', morphed)

    # --- ETAPA 6: Predicción de formas
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
                label_text = labels.get(int(prediction[0]), "Desconocido")

                # Rectángulo para texto
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(result_frame, f"{label_text}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(result_frame, "No se detectaron formas", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('6. Resultado Final', result_frame)

    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
capture.release()
cv2.destroyAllWindows()
print("Programa terminado correctamente.")

