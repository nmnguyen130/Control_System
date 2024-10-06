import cv2
from services.camera_service import CameraService
from handlers.label_handler import LabelHandler

def main():
    camera_service = CameraService(num_hands=2)
    label_handler = LabelHandler('data/hand_gesture/labels.csv')

    while True:
        frame = camera_service.capture_frame()
        if frame is None:
            print("Error: No frame captured.")
            break

        landmarks_list = camera_service.get_landmarks(frame)
        cv2.imshow('Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_service.release()

if __name__ == '__main__':
    main()
