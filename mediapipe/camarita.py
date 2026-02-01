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
    exit()
else:
    print("Cámara iniciada. Presiona 'q' para salir.")

# Crear ventanas con posiciones específicas
cv2.namedWindow('1. Imagen Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('2. Escala de Grises', cv2.WINDOW_NORMAL)
cv2.namedWindow('3. Imagen Binarizada', cv2.WINDOW_NORMAL)
cv2.namedWindow('4. Contornos Detectados', cv2.WINDOW_NORMAL)
cv2.namedWindow('5. Resultado Final', cv2.WINDOW_NORMAL)

# Posicionar las ventanas
cv2.moveWindow('1. Imagen Original', 50, 50)
cv2.moveWindow('2. Escala de Grises', 350, 50)
cv2.moveWindow('3. Imagen Binarizada', 650, 50)
cv2.moveWindow('4. Contornos Detectados', 50, 400)
cv2.moveWindow('5. Resultado Final', 350, 400)

while True:
    # Capturamos un fotograma de la cámara
    ret, frame = capture.read()
    
    if not ret:
        print("No se pudo capturar el fotograma. Reintentando...")
        continue
    
    # Volteamos la imagen horizontalmente para efecto espejo
    frame = cv2.flip(frame, 1)
    
    # ETAPA 1: Mostrar imagen original
    cv2.imshow('1. Imagen Original', frame)
    
    # ETAPA 2: Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('2. Escala de Grises', gray)
    
    # ETAPA 3: Aplicar desenfoque y binarización
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('3. Imagen Binarizada', thresh)
    
    # ETAPA 4: Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una copia para mostrar solo los contornos
    contour_image = frame.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow('4. Contornos Detectados', contour_image)
    
    # ETAPA 5: Análisis y predicción (para muchos contornos)
    result_frame = frame.copy()
    
    if contours:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filtro para evitar ruido
                # Dibujar el contorno
                cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)

                # Calcular momentos de Hu
                moments = cv2.moments(contour)
                hu_moments = cv2.HuMoments(moments)

                # Procesar momentos de Hu
                for i in range(0, 7):
                    val = hu_moments[i][0]
                    if val != 0:
                        hu_moments[i][0] = -1 * math.copysign(1.0, val) * math.log10(abs(val))

                # Preparar datos para predicción
                hu_moments_reshaped = hu_moments.reshape(1, -1).astype(np.float32)

                # Realizar predicción
                prediction = model.predict(hu_moments_reshaped)
                label_text = labels.get(int(prediction[0]), "Desconocido")

                # Obtener el rectángulo delimitador para ubicar el texto
                x, y, w, h = cv2.boundingRect(contour)

                # Mostrar información al lado de la forma
                cv2.putText(result_frame, f"{label_text}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

                cv2.putText(result_frame, f"Area: {int(area)}", (x, y + h + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(result_frame, "No se detectaron formas", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow('5. Resultado Final', result_frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
capture.release()
cv2.destroyAllWindows()
print("Programa terminado correctamente.")

