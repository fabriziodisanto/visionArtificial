# pip install mediapipe 
import cv2
import mediapipe as mp
import numpy as np

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    # static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def get_hand_gesture(hand_landmarks):
    if not hand_landmarks:
        return False
    
    # Get landmarks
    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]

    index_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_MCP]
    middle_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_MCP]
    ring_mcp = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_MCP]
    pinky_mcp= hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_MCP]

    finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
    finger_mcps = [index_mcp, middle_mcp, ring_mcp, pinky_mcp]

    extended_fingers = 0
    landmarks = hand_landmarks.landmark
    if landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y:
        extended_fingers += 1
    if landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y:
        extended_fingers += 1
    if landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.RING_FINGER_PIP].y:
        extended_fingers += 1
    if landmarks[mp_holistic.HandLandmark.PINKY_TIP].y < landmarks[mp_holistic.HandLandmark.PINKY_PIP].y:
        extended_fingers += 1

    if extended_fingers == 0:
        return "Piedra"
    elif extended_fingers == 4:
        return "Papel"
    
    index_extended = landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.INDEX_FINGER_PIP].y
    middle_extended = landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_holistic.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_contracted = landmarks[mp_holistic.HandLandmark.RING_FINGER_TIP].y > landmarks[mp_holistic.HandLandmark.RING_FINGER_PIP].y
    pinky_contracted = landmarks[mp_holistic.HandLandmark.PINKY_TIP].y > landmarks[mp_holistic.HandLandmark.PINKY_PIP].y
    
    if index_extended and middle_extended and ring_contracted and pinky_contracted:
        return "Tijera"
    
    return "Desconocido"

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
 
    # Converting the from BGR to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Making predictions using holistic model
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
 
    # Converting back the RGB image to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    # Variables para guardar la jugada de cada mano
    left_hand_gesture = "Desconocido"
    right_hand_gesture = "Desconocido"

    # Obtener la jugada de la mano izquierda si está presente
    if results.left_hand_landmarks:
        left_hand_gesture = get_hand_gesture(results.left_hand_landmarks)
            
    # Obtener la jugada de la mano derecha si está presente
    if results.right_hand_landmarks:
        right_hand_gesture = get_hand_gesture(results.right_hand_landmarks)

    # Lógica para determinar el ganador y mostrar el resultado
    # Solo se ejecuta si ambas manos están presentes y sus gestos son reconocidos
    if left_hand_gesture != "desconocido" and right_hand_gesture != "desconocido":
        
        # Caso de Empate
        if left_hand_gesture == "Desconocido" or right_hand_gesture == "Desconocido":
            cv2.putText(image, f"Invalido", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2, cv2.LINE_AA)

        elif left_hand_gesture == right_hand_gesture and left_hand_gesture != "Desconocido":
            cv2.putText(image, f"Empate! Ambos eligieron {left_hand_gesture}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, GREEN_COLOR, 2, cv2.LINE_AA)
        # Casos donde gana la mano izquierda
        elif (left_hand_gesture == "Piedra" and right_hand_gesture == "Tijera") or \
             (left_hand_gesture == "Papel" and right_hand_gesture == "Piedra") or \
             (left_hand_gesture == "Tijera" and right_hand_gesture == "Papel"):
            cv2.putText(image, "Gana la mano Derecha!", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE_COLOR, 2, cv2.LINE_AA)
        
        # Si no es empate ni gana la izquierda, gana la derecha
        elif (left_hand_gesture == "Tijera" and right_hand_gesture == "Piedra") or \
             (left_hand_gesture == "Piedra" and right_hand_gesture == "Papel") or \
             (left_hand_gesture == "Papel" and right_hand_gesture == "Tijera"):
            cv2.putText(image, "Gana la mano Izquierda!", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Invalido", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "Empiece a jugar!", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE_COLOR, 2, cv2.LINE_AA)

    cv2.imshow("Piedra, Papel o Tijera - Dos Jugadores", image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
 
# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()