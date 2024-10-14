import cv2
from src.services.camera_service import CameraService
from src.services.hand_gesture_service import HandGestureService
from src.utils.visualization_utils import draw_label

def main():
    static_model_path = 'trained_data/static_gesture_model.pth'
    dynamic_model_path = 'trained_data/dynamic_gesture_model.pth'
    label_path = 'data/hand_gesture/labels.csv'

    recognition_service = HandGestureService(static_model_path, dynamic_model_path, label_path)
    camera_service = CameraService(num_hands=2)

    while True:
        frame = camera_service.capture_frame()
        if frame is None:
            break
        
        landmarks_list = camera_service.get_landmarks(frame)
        if landmarks_list:
            for i, landmarks in enumerate(landmarks_list):
                hand_position = 'left' if i == 0 else 'right'

                static_gesture_index = recognition_service.predict_static(landmarks)
                static_gesture_name = recognition_service.get_gesture_name(static_gesture_index)

                # dynamic_gesture_index = recognition_service.predict_dynamic(landmarks)
                # dynamic_gesture_name = recognition_service.get_gesture_name(dynamic_gesture_index)

                # Vẽ nhãn cử chỉ lên khung hình
                draw_label(frame, static_gesture_name, hand_position + ' (Static)')
                # draw_label(frame, dynamic_gesture_name, hand_position + ' (Dynamic)')

        cv2.imshow('Hand Gesture Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_service.release()

if __name__ == "__main__":
    main()