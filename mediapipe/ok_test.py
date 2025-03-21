import cv2
import time
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
mp_drawing = mp.solutions.drawing_utils

RED_COLOR = (0, 0, 255)
capture = cv2.VideoCapture(0)

def calculate_distance(p1, p2):
    """Calculate distance between two points"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_ok_sign(hand_landmarks):
    if not hand_landmarks:
        return False
        
    # Get landmarks
    thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]
    
    # Check if thumb and index are close (forming a circle)
    thumb_index_distance = calculate_distance(thumb_tip, index_tip)
    
    # Get distances for other fingers to palm base for checking if they're extended
    palm_base = hand_landmarks.landmark[mp_holistic.HandLandmark.WRIST]
    middle_distance = calculate_distance(middle_tip, palm_base)
    ring_distance = calculate_distance(ring_tip, palm_base)
    pinky_distance = calculate_distance(pinky_tip, palm_base)
    
    # Thresholds
    circle_threshold = 0.1
    extension_threshold = 0.15
    
    # Check if thumb and index form a small circle and other fingers are extended
    is_circle_formed = thumb_index_distance < circle_threshold
    are_others_extended = all(d > extension_threshold for d in [middle_distance, ring_distance, pinky_distance])
    
    return is_circle_formed and are_others_extended

while capture.isOpened():
    ret, frame = capture.read()
 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True
 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
 
    right_hand_ok = False
    left_hand_ok = False

    # Check for OK sign in right hand
    if results.right_hand_landmarks:
        right_hand_ok = is_ok_sign(results.right_hand_landmarks)
            
    # Check for OK sign in left hand
    if results.left_hand_landmarks:
        left_hand_ok = is_ok_sign(results.left_hand_landmarks)

    if right_hand_ok and left_hand_ok:
        cv2.putText(image, "Both hands OK Sign Detected!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2)
    elif right_hand_ok:
        cv2.putText(image, "Right hand OK Sign Detected!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2)
    elif left_hand_ok:       
        cv2.putText(image, "Left hand OK Sign Detected!", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, RED_COLOR, 2)
 
    cv2.imshow("Image", image)
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
capture.release()
cv2.destroyAllWindows()
