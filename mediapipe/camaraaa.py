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

# --- 2. CONFIGURACIÓN MEJORADA ---
# Parámetros de filtrado adaptativos
MIN_AREA = 200  # Área mínima más pequeña
MAX_AREA = 50000  # Área máxima para evitar detectar toda la imagen
MIN_ASPECT_RATIO = 0.2  # Relación de aspecto mínima
MAX_ASPECT_RATIO = 5.0  # Relación de aspecto máxima

# --- 3. INICIAR LA CÁMARA ---
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()
else:
    print("Cámara iniciada. Presiona 'q' para salir.")

# Configurar resolución de cámara para mejor rendimiento
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Crear ventanas
cv2.namedWindow('Original + Detecciones', cv2.WINDOW_NORMAL)
cv2.namedWindow('Procesamiento', cv2.WINDOW_NORMAL)
cv2.moveWindow('Original + Detecciones', 50, 50)
cv2.moveWindow('Procesamiento', 700, 50)

# Variables para estabilización
last_predictions = []
PREDICTION_BUFFER = 5  # Número de predicciones a promediar

def preprocess_image(frame):
    """Preprocesamiento mejorado de la imagen"""
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar filtro bilateral para reducir ruido manteniendo bordes
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Ecualización de histograma adaptativo para mejor contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(filtered)
    
    # Desenfoque gaussiano suave
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    return blurred

def adaptive_threshold(image):
    """Binarización adaptativa mejorada"""
    # Usar umbralización adaptativa que se ajusta a las condiciones locales
    thresh1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Combinar con umbralización de Otsu
    _, thresh2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Combinar ambos métodos
    combined = cv2.bitwise_and(thresh1, thresh2)
    
    # Operaciones morfológicas para limpiar la imagen
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def filter_contours(contours):
    """Filtrar contornos basándose en criterios mejorados"""
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtrar por área
        if area < MIN_AREA or area > MAX_AREA:
            continue
        
        # Calcular rectángulo delimitador
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Filtrar por relación de aspecto
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            continue
        
        # Calcular la solidez (área del contorno / área del rectángulo delimitador)
        rect_area = w * h
        solidity = float(area) / rect_area
        
        # Filtrar formas demasiado irregulares o demasiado simples
        if solidity < 0.3 or solidity > 0.95:
            continue
        
        # Calcular el perímetro y la compacidad
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = (perimeter * perimeter) / (4 * np.pi * area)
            # Filtrar formas extremadamente irregulares
            if compactness > 10:
                continue
        
        valid_contours.append(contour)
    
    return valid_contours

def get_stable_prediction(prediction):
    """Estabilizar predicciones usando un buffer"""
    global last_predictions
    
    last_predictions.append(prediction)
    if len(last_predictions) > PREDICTION_BUFFER:
        last_predictions.pop(0)
    
    if len(last_predictions) >= 3:
        # Usar la predicción más frecuente
        from collections import Counter
        most_common = Counter(last_predictions).most_common(1)
        return most_common[0][0]
    
    return prediction

# --- BUCLE PRINCIPAL MEJORADO ---
while True:
    ret, frame = capture.read()
    
    if not ret:
        print("No se pudo capturar el fotograma. Reintentando...")
        continue
    
    # Voltear imagen para efecto espejo
    frame = cv2.flip(frame, 1)
    
    # Preprocesamiento mejorado
    processed = preprocess_image(frame)
    
    # Binarización adaptativa
    thresh = adaptive_threshold(processed)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos
    valid_contours = filter_contours(contours)
    
    # Crear imagen de resultado
    result_frame = frame.copy()
    
    # Mostrar imagen de procesamiento
    processing_display = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    if valid_contours:
        # Ordenar contornos por área (el más grande primero)
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        # Procesar hasta 3 contornos más grandes
        for i, contour in enumerate(valid_contours[:3]):
            area = cv2.contourArea(contour)
            
            # Dibujar contorno
            color = [(0, 255, 0), (255, 0, 0), (0, 0, 255)][i]  # Verde, Azul, Rojo
            cv2.drawContours(result_frame, [contour], -1, color, 2)
            cv2.drawContours(processing_display, [contour], -1, color, 2)
            
            # Calcular momentos de Hu
            moments = cv2.moments(contour)
            if moments['m00'] != 0:  # Evitar división por cero
                hu_moments = cv2.HuMoments(moments)
                
                # Procesar momentos de Hu
                for j in range(0, 7):
                    val = hu_moments[j][0]
                    if val != 0:
                        hu_moments[j][0] = -1 * math.copysign(1.0, val) * math.log10(abs(val))
                
                # Preparar datos para predicción
                hu_moments_reshaped = hu_moments.reshape(1, -1).astype(np.float32)
                
                # Realizar predicción
                prediction = model.predict(hu_moments_reshaped)
                stable_prediction = get_stable_prediction(int(prediction[0]))
                label_text = labels.get(stable_prediction, "Desconocido")
                
                # Calcular posición del texto
                x, y, w, h = cv2.boundingRect(contour)
                
                # Mostrar información
                cv2.putText(result_frame, f"{label_text}", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                cv2.putText(result_frame, f"Area: {int(area)}", (x, y+h+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
                # Dibujar rectángulo delimitador
                cv2.rectangle(result_frame, (x, y), (x+w, y+h), color, 1)
    
    else:
        cv2.putText(result_frame, "Acerca una forma a la camara", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result_frame, "Mejores resultados con fondo contrastante", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # Mostrar FPS
    cv2.putText(result_frame, "Presiona 'q' para salir", (20, frame.shape[0]-20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Mostrar resultados
    cv2.imshow('Original + Detecciones', result_frame)
    cv2.imshow('Procesamiento', processing_display)
    
    # Salir con 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset buffer de predicciones
        last_predictions = []
        print("Buffer de predicciones reiniciado")

# Liberar recursos
capture.release()
cv2.destroyAllWindows()
print("Programa terminado correctamente.")