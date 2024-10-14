import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.services.camera_service import CameraService
from src.services.hand_gesture_service import HandGestureService
from src.handlers.hand_detect_handler import HandDetectHandler

def detect_gesture():
    camera_service = CameraService()
    hand_handler = HandDetectHandler(num_hands=2)

    static_model_path = 'trained_data/static_gesture_model.pth'
    dynamic_model_path = 'trained_data/dynamic_gesture_model.pth'
    static_label_path = 'data/hand_gesture/static_labels.csv'
    dynamic_label_path = 'data/hand_gesture/dynamic_labels.csv'

    hand_gesture_service = HandGestureService(static_model_path, dynamic_model_path, static_label_path, dynamic_label_path)

    while True:
        frame = camera_service.capture_frame()
        if frame is None:
            break

        hands, _ = hand_handler.find_hands(frame, isDraw=True)
        
        landmarks_list = [hand["lmList"] for hand in hands]
        for landmarks in landmarks_list:
            # Static gesture prediction
            _, gesture_name_static = hand_gesture_service.predict_static(landmarks)
            cv2.putText(frame, f'Static Gesture: {gesture_name_static}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Dynamic gesture processing
            _, gesture_name_dynamic = hand_gesture_service.predict_dynamic(landmarks)
            if gesture_name_dynamic:
                cv2.putText(frame, f'Dynamic Gesture: {gesture_name_dynamic}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Hiển thị khung hình lên cửa sổ camera
        cv2.imshow('Capture Hand Gesture', frame)

        # Nhấn phím để điều khiển
        key = cv2.waitKey(1) & 0xFF

        # Nhấn 'q' để thoát
        if key == ord('q'):
            break

    camera_service.release()

if __name__ == "__main__":
    detect_gesture()