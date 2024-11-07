import cv2
import time
from src.services.camera_service import CameraService
from src.services.hand_gesture_service import HandGestureService
from src.handlers.hand_detect_handler import HandDetectHandler
from src.handlers.gesture_label_handler import GestureLabelHandler
from src.modules.system_control import GestureController

class GestureDetector:
    def __init__(self, add_message_callback):
        self.add_message = add_message_callback
        self.camera_service = CameraService()
        self.hand_handler = HandDetectHandler(num_hands=2)

        static_model_path = 'trained_data/static_gesture_model.pth'
        dynamic_model_path = 'trained_data/dynamic_gesture_model.pth'
        static_label_path = 'data/hand_gesture/static_labels.csv'
        dynamic_label_path = 'data/hand_gesture/dynamic_labels.csv'

        self.label_handler = GestureLabelHandler(static_label_path, dynamic_label_path)
        self.hand_gesture_service = HandGestureService(static_model_path, dynamic_model_path, self.label_handler)
        self.gesture_controller = GestureController()

        self.last_detection_time = 0
        self.detection_interval = 0.5  # 500ms between detections (2 FPS for gesture detection)

    def start_detection(self):
        prev_x, prev_y = None, None
        
        try:
            while True:
                frame = self.camera_service.capture_frame()
                if frame is None:
                    break

                current_time = time.time()
                hands, _ = self.hand_handler.find_hands(frame, isDraw=True)
                landmarks_list = [hand["lmList"] for hand in hands]
                
                for landmarks in landmarks_list:
                    thumb = landmarks[4*3:4*3+3]  # x,y,z of thumb tip
                    curr_x, curr_y = thumb[0], thumb[1]
                    
                    # Static gesture prediction
                    _, gesture_name_static = self.hand_gesture_service.predict_static(landmarks)
                    if gesture_name_static is not None:
                        if gesture_name_static == "fist":
                            if prev_x is not None and prev_y is not None:
                                dx = curr_x - prev_x
                                dy = curr_y - prev_y
                                self.handle_gesture(gesture_name_static, dx, dy)
                            prev_x, prev_y = curr_x, curr_y
                        else:
                            if current_time - self.last_detection_time >= self.detection_interval:
                                prev_x, prev_y = None, None
                                self.handle_gesture(gesture_name_static)
                                self.last_detection_time = current_time
                    cv2.putText(frame, f"Static: {gesture_name_static}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Dynamic gesture processing
                    # _, gesture_name_dynamic = self.hand_gesture_service.predict_dynamic(landmarks)
                    # if gesture_name_dynamic:
                    #     # self.handle_gesture(gesture_name_dynamic)
                    #     cv2.putText(frame, f"Dynamic: {gesture_name_dynamic}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Hiển thị khung hình lên cửa sổ camera
                cv2.imshow('Capture Hand Gesture', frame)

                # Nhấn phím để điều khiển
                key = cv2.waitKey(1) & 0xFF

                # Nhấn 'q' để thoát
                if key == ord('q'):
                    break
        finally:
            self.camera_service.release()

    def handle_gesture(self, gesture_name, dx=None, dy=None):
        action_control = self.label_handler.get_static_action_by_gesture(gesture_name)

        if action_control:
            message = f"Detected Gesture: {gesture_name}, Control Action: {action_control}"
            self.add_message("Gesture Detection", message)  # Send message to UI
            self.gesture_controller.handle_gesture(action_control, dx, dy)

# Testing the GestureDetector
def add_message(speaker, message):
    print(f"{speaker}: {message}")

if __name__ == "__main__":
    gesture_detector = GestureDetector(add_message)
    gesture_detector.start_detection()