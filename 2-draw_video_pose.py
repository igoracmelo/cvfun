import cv2
import mediapipe as mp

pose = mp.solutions.pose
drawing = mp.solutions.drawing_utils

cam = cv2.VideoCapture(0)
pose_detector = pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    ok, frame = cam.read()

    if not ok:
        print("failed to read frame")
        continue

    result = pose_detector.process(frame)

    drawing.draw_landmarks(
        frame,
        result.pose_landmarks,
        pose.POSE_CONNECTIONS,
    )

    cv2.imshow("window", cv2.flip(frame, 1))
    cv2.waitKey(5)
