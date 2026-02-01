import cv2
import numpy as np
import math

# Importamos la función para entrenar desde tu propio script
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
    print("Asegúrate de que no esté siendo usada por otro programa.")
    exit()
else:
    print("Cámara iniciada. Presiona 'q' para salir.")

while True:
    # Capturamos un fotograma (frame) de la cámara
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)

    # --- MODIFICACIÓN CLAVE ---
    if not ret:
        print("Advertencia: No se pudo capturar el fotograma. Reintentando...")
        continue

    # --- 3. PROCESAR LA IMAGEN PARA ENCONTRAR FORMAS ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 4. ANALIZAR LA FORMA MÁS GRANDE ENCONTRADA ---
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 500:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 3)

            # CORRECCIÓN: extracción de valores escalares de Hu Moments
            moments = cv2.moments(cnt)
            hu_moments = cv2.HuMoments(moments)
            for i in range(0, 7):
                val = hu_moments[i][0]  # extraemos el escalar
                if val != 0:
                    hu_moments[i][0] = -1 * math.copysign(1.0, val) * math.log10(abs(val))

            # CORRECCIÓN: conversión a float32
            hu_moments_reshaped = hu_moments.reshape(1, -1).astype(np.float32)

            # Predicción del modelo
            prediction = model.predict(hu_moments_reshaped)
            label_text = labels.get(int(prediction[0]), "Desconocido")

            cv2.putText(frame, label_text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

    # --- 6. MOSTRAR EL VIDEO ---
    cv2.imshow('Reconocimiento de Figuras', frame)

    # Salimos del bucle con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberamos la cámara y cerramos las ventanas
capture.release()
cv2.destroyAllWindows()

