import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib

# Load the trained model
model = tf.keras.models.load_model('gesture_recognition_model.h5')

# Load label mapping
label_mapping = joblib.load('label_mapping.pkl')

# Define gesture meanings (based on your labels)
gesture_meanings = {
    0: "Friend",
    1: "A",
    # Add more gestures if needed
}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Preprocess function for real-time keypoint extraction
def preprocess_keypoints(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    keypoints = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            image_height, image_width, _ = frame.shape
            for landmark in hand_landmarks.landmark:
                x = landmark.x * image_width
                y = landmark.y * image_height
                z = landmark.z * image_width
                keypoints.extend([x, y, z])

    # Ensure keypoints list is complete for both hands
    while len(keypoints) < 21 * 3 * 2:  # 21 landmarks * 3 (x, y, z) for two hands
        keypoints.extend([0, 0, 0])

    # Reshape for LSTM input
    keypoints = np.array(keypoints).reshape((1, 1, len(keypoints)))
    return keypoints, result

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Preprocess frame to get keypoints and the result object
    keypoints, result = preprocess_keypoints(frame)

    # Make prediction
    prediction = model.predict(keypoints)
    predicted_class = np.argmax(prediction, axis=1)[0]

    # Get the gesture meaning
    gesture_meaning = gesture_meanings.get(predicted_class, "Unknown Gesture")

    # Display the result on the frame
    cv2.putText(frame, f"Prediction: {gesture_meaning}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw landmarks on the frame
    if result and result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame
    cv2.imshow("Gesture Recognition", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
