#!/usr/bin/env python3

import cv2
import mediapipe as mp

cam = cv2.VideoCapture(0)
face_detector = mp.solutions.face_detection.FaceDetection()

while True:
    ok, frame = cam.read()

    if not ok:
        print("failed to read frame")
        break

    faces = face_detector.process(frame)

    if faces.detections:
        for face in faces.detections:
            mp.solutions.drawing_utils.draw_detection(frame, face)

    cv2.imshow("eita", frame)

    key = cv2.waitKey(5)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
