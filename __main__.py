import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

# Gives us drawing utilities
mp_drawing = mp.solutions.drawing_utils
# Imports the pose estimation models
# Mediapipe has a lot of models such as iris detection, face, etc...
mp_pose = mp.solutions.pose

# Sets up webcam from cv2
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Reads video input from cap
        ret, frame = cap.read()

        # Recolor the image before passing it to the pose estimation model. This reduces noise and memory usage as color is not that important for this task
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)
        image.flags.writeable = True
        # Rerenders image back to color BGR (opencv requires colors in BGR format)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Extarcting the landmark
        try:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            # Calculate the angle
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            # Visualize the angle
            cv2.putText(image, str(angle), tuple(np.multiply(left_elbow, [640, 480]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                                2, cv2.LINE_AA)
            if angle > 160.0:
                stage = "down"
            if angle < 30 and stage == 'down':
                stage = 'up'
                counter += 1
                print(counter)
        except:
            pass

        cv2.rectangle(image, (0,0), (255,73), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        # Render detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
        

        # Renders to the display
        cv2.imshow("Mediapipe Feed", image)
        # Waits for key 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()