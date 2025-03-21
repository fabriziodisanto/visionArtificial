# pip install mediapipe 
import cv2
import mediapipe as mp

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    # static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
 
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
 
    # Drawing Right hand Land Marks
    mp_drawing.draw_landmarks(
      image, 
      results.right_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
 
    # Drawing Left hand Land Marks
    mp_drawing.draw_landmarks(
      image, 
      results.left_hand_landmarks, 
      mp_holistic.HAND_CONNECTIONS
    )
      
    # Display the resulting image
    cv2.imshow("Hand Landmarks", image)
 
    # Enter key 'q' to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()