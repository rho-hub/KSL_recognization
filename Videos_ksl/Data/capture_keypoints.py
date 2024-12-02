import cv2
import mediapipe as mp
import csv
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# File to save keypoints
data_file = 'ksl_keypoints.csv'

# Prepare CSV file with headers only if the file does not exist
if not os.path.exists(data_file):
    with open(data_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # 21 landmarks * 3 (x, y, z) for 2 hands + 1 label column
        headers = [f'hand_{hand}_{i}_{axis}' for hand in range(2) for i in range(21) for axis in ('x', 'y', 'z')] + ['label']
        writer.writerow(headers)


def capture_keypoints(label):
    cap = cv2.VideoCapture(0)
    print(f"Capturing data for gesture: {label}. Press 'q' to quit.")

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result and result.multi_hand_landmarks:
            keypoints = []
            image_height, image_width, _ = frame.shape

            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                for landmark in hand_landmarks.landmark:
                    x = landmark.x * image_width
                    y = landmark.y * image_height
                    z = landmark.z * image_width  # Depth scaling
                    keypoints.extend([x, y, z])

            # Ensure the keypoints list has 2 hands (21 landmarks each, 3 coordinates per landmark)
            while len(keypoints) < 21 * 3 * 2:
                keypoints.extend([0, 0, 0])

            # Append keypoints to the CSV file
            with open(data_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(keypoints + [label])

        else:
            print("No hand detected in this frame.")

        cv2.imshow('Capture Keypoints', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main function to collect data
def main():
    while True:
        label = input("Enter the label for the gesture (or 'exit' to quit): ")
        if label.lower() == 'exit':
            print("Exiting data collection.")
            break
        capture_keypoints(label)


if __name__ == "__main__":
    main()
