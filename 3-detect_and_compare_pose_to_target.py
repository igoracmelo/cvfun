import cv2
import mediapipe as mp
import math

pl = mp.solutions.pose.PoseLandmark
drawing_utils = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

def main():
    angle_names = [
        'left elbow',
        'left shoulder',
        'left thigh',
        'left knee',
        'right elbow',
        'right shoulder',
        'right thigh',
        'right knee',
    ]

    pose_detector = mp.solutions.pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    target = cv2.imread('onefoot.png')

    target_result = pose_detector.process(target)
    if not target_result.pose_landmarks:
        print("failed to process reference image")
        return

    drawing_utils.draw_landmarks(
        target,
        target_result.pose_landmarks,
        mp.solutions.pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
    )
    cv2.imshow("", cv2.flip(target, 1))

    target_angles = extract_pose_angles(target_result.pose_landmarks.landmark)

    cam = cv2.VideoCapture(0)
    while True:
        ok, frame = cam.read()
        if not ok:
            continue

        result = pose_detector.process(frame)
        if not result.pose_landmarks:
            continue

        drawing_utils.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style()
        )

        landmarks = result.pose_landmarks.landmark
        angle = get_angle(
            landmarks[pl.LEFT_SHOULDER],
            landmarks[pl.LEFT_ELBOW],
            landmarks[pl.LEFT_WRIST],
        )

        user_angles = extract_pose_angles(result.pose_landmarks.landmark)

        bad = False
        for i in range(len(target_angles)):
            t = target_angles[i]
            u = user_angles[i]
            if not is_angle_close(t, u):
                print(f'BAD: {angle_names[i]} ({u} vs {t})')
                bad = True

        if not bad:
            print("ALL GOOD!")

        cv2.imshow("", cv2.flip(frame, 1))
        cv2.waitKey(16)

def get_angle(p1, p2, p3):
    angle_rad = math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p1.y - p2.y, p1.x - p2.x)
    angle_deg = angle_rad * 180 / math.pi
    angle_deg = abs(angle_deg)
    if angle_deg > 180:
        angle_deg = 360 - angle_deg
    return int(angle_deg)

def is_angle_close(a, b, e=30):
    d = abs(a - b)
    return d < e

def extract_pose_angles(landmarks):
    angles = []

    # left
    angles.append(get_angle(
        landmarks[pl.LEFT_SHOULDER],
        landmarks[pl.LEFT_ELBOW],
        landmarks[pl.LEFT_WRIST],
    ))

    angles.append(get_angle(
        landmarks[pl.LEFT_ELBOW],
        landmarks[pl.LEFT_SHOULDER],
        landmarks[pl.LEFT_HIP],
    ))

    angles.append(get_angle(
        landmarks[pl.LEFT_SHOULDER],
        landmarks[pl.LEFT_HIP],
        landmarks[pl.LEFT_KNEE],
    ))

    angles.append(get_angle(
        landmarks[pl.LEFT_HIP],
        landmarks[pl.LEFT_KNEE],
        landmarks[pl.LEFT_ANKLE],
    ))

    # right
    angles.append(get_angle(
        landmarks[pl.RIGHT_SHOULDER],
        landmarks[pl.RIGHT_ELBOW],
        landmarks[pl.RIGHT_WRIST],
    ))

    angles.append(get_angle(
        landmarks[pl.RIGHT_ELBOW],
        landmarks[pl.RIGHT_SHOULDER],
        landmarks[pl.RIGHT_HIP],
    ))

    angles.append(get_angle(
        landmarks[pl.RIGHT_SHOULDER],
        landmarks[pl.RIGHT_HIP],
        landmarks[pl.RIGHT_KNEE],
    ))

    angles.append(get_angle(
        landmarks[pl.RIGHT_HIP],
        landmarks[pl.RIGHT_KNEE],
        landmarks[pl.RIGHT_ANKLE],
    ))

    return angles

if __name__ == '__main__':
    main()
