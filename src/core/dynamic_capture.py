import os
import cv2
import pandas as pd
import threading
import queue
import time
from tqdm import tqdm
from src.utils.file_utils import create_directory_if_not_exists
from src.utils.data_utils import initialize_dataframe, save_dataframe_to_csv
from src.services.camera_service import CameraService
from src.handlers.hand_detect_handler import HandDetectHandler
from src.handlers.gesture_label_handler import GestureLabelHandler

class DataCollector(threading.Thread):
    def __init__(self, camera_service, hand_handler, data_queue, current_label, sequence_length, no_sequences):
        super().__init__()
        self.camera_service = camera_service
        self.hand_handler = hand_handler
        self.data_queue = data_queue
        self.current_label = current_label
        self.sequence_length = sequence_length
        self.no_sequences = no_sequences
        self.running = True
        self.fps = 30  # Target FPS for consistent capture

    def run(self):
        for sequence in range(self.no_sequences):
            print(f'\nGesture sequence: {sequence + 1}/{self.no_sequences}')
            self._countdown()
            
            with tqdm(total=self.sequence_length, desc=f"Collecting frames") as pbar:
                for frame_num in range(self.sequence_length):
                    frame_start_time = time.time()
                    
                    frame = self.camera_service.capture_frame()
                    if frame is None or not self.running:
                        return

                    hands, _ = self.hand_handler.find_hands(frame, isDraw=True)

                    if hands:
                        landmarks = hands[0]["lmList"].tolist()
                        landmarks.append(self.current_label)
                        landmarks.insert(0, frame_num)
                        self.data_queue.put(landmarks)
                    else:
                        print("\nWarning: No hand detected in frame")

                    # Improved visual feedback
                    self._draw_capture_info(frame, sequence, frame_num)
                    cv2.imshow('Capture Dynamic Gesture', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        return

                    # Control frame rate
                    elapsed = time.time() - frame_start_time
                    if elapsed < 1./self.fps:
                        time.sleep(1./self.fps - elapsed)
                    
                    pbar.update(1)

            # Pause between sequences
            if sequence < self.no_sequences - 1:
                time.sleep(1)  # 1 second pause between sequences

        print(f"\nData collection for gesture '{self.current_label}' completed successfully.")

    def _countdown(self, count=3):
        for i in range(count, 0, -1):
            print(f"\rStarting capture in {i}...", end="")
            time.sleep(1)
        print("\rCapturing...            ")

    def _draw_capture_info(self, frame, sequence, frame_num):
        info_text = f"Gesture: {self.current_label} | Sequence: {sequence + 1}/{self.no_sequences}"
        cv2.putText(frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        progress = f"Frame: {frame_num + 1}/{self.sequence_length}"
        cv2.putText(frame, progress, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    def stop(self):
        self.running = False

def capture_dynamic_gesture(csv_output_path):
    """
    Capture data for dynamic gestures, extract landmarks, and save to a CSV file.
    - Press 's' to start capturing dynamic gesture.
    - Press 'q' to quit.
    """

    create_directory_if_not_exists(os.path.dirname(csv_output_path))

    camera_service = CameraService()
    hand_handler = HandDetectHandler(num_hands=1)

    # Tạo dataframe để lưu điểm đặc trưng
    columns = ['frame'] + [f'point_{i}_{axis}' for i in range(1, 22) for axis in ['x', 'y', 'z']] + ['label']  # Tổng cộng 65 cột
    df = initialize_dataframe(columns, csv_output_path)
    current_label = None

    label_handler = GestureLabelHandler(dynamic_label_path='data/hand_gesture/dynamic_labels.csv')
    label_map = {str(i + 1): gesture for i, gesture in enumerate(label_handler.get_all_dynamic_gestures())}

    no_sequences = 5
    sequence_length = 30
    data_queue = queue.Queue()  # Initialize queue
    data_collector = None   
    
    try:
        while True:
            frame = camera_service.capture_frame()
            if frame is None:
                print("Error: Cannot capture frame")
                break

            hands, _ = hand_handler.find_hands(frame, isDraw=True)

            if current_label:
                cv2.putText(frame, f"Current Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Capture Dynamic Gesture', frame)

            # Nhấn phím để điều khiển
            key = cv2.waitKey(1) & 0xFF

            # Nhấn phím số (1-...) để chọn nhãn cho dữ liệu
            if chr(key) in label_map:
                current_label = label_map[chr(key)]
                print(f"Nhãn hiện tại: {current_label}")

            # Nhấn 's' để bắt đầu ghi dữ liệu cho động
            elif key == ord('s') and current_label and (data_collector is None or not data_collector.is_alive()):
                print(f"Starting data collection for gesture: {current_label}...")
                data_collector = DataCollector(camera_service, hand_handler, data_queue, current_label, sequence_length, no_sequences)
                data_collector.start()
                    
            # Nhấn 'q' để thoát
            elif key == ord('q'):
                break

            # Cập nhật DataFrame từ hàng đợi
            while not data_queue.empty():
                landmarks = data_queue.get()
                df = pd.concat([df, pd.DataFrame([landmarks], columns=df.columns)], ignore_index=True)

    except Exception as e:
        print(f"Error during capture: {e}")
    finally:
        if data_collector is not None:
            data_collector.stop()
            data_collector.join()
        camera_service.release()
        cv2.destroyAllWindows()
        save_dataframe_to_csv(df, csv_output_path)

if __name__ == "__main__":
    capture_dynamic_gesture('data/hand_gesture/dynamic_data.csv')
