import os
import cv2
import pandas as pd
import threading
import queue
from src.utils.file_utils import create_directory_if_not_exists
from src.utils.data_utils import initialize_dataframe, save_dataframe_to_csv
from src.services.camera_service import CameraService
from src.handlers.hand_detect_handler import HandDetectHandler
from src.handlers.label_handler import LabelHandler

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

    def run(self):
        for sequence in range(self.no_sequences):
            print(f'Gesture sequence: {sequence + 1}/{self.no_sequences}')
            for frame_num in range(self.sequence_length):
                frame = self.camera_service.capture_frame()
                if frame is None or not self.running:
                    return

                hands, _ = self.hand_handler.find_hands(frame, isDraw=True)

                if hands:
                    landmarks = hands[0]["lmList"].tolist()
                    landmarks.append(self.current_label)
                    landmarks.insert(0, frame_num)
                    # Đưa dữ liệu vào hàng đợi
                    self.data_queue.put(landmarks)

                if frame_num == 29:
                    cv2.putText(frame, 'STOP COLLECTION', (120, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting frames for {self.current_label} Video Number {sequence + 1}', (15, 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Capture Dynamic Gesture', frame)
                    
                    print(f"Completed sequence {sequence + 1}. Pausing for 0.5 seconds...")
                    cv2.waitKey(500)
                else:
                    cv2.putText(frame, f'Collecting frames for {self.current_label} Video Number {sequence + 1}', (15, 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Capture Dynamic Gesture', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    return
                
        print(f"Data collection for gesture '{self.current_label}' completed.")

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

    label_handler = LabelHandler(dynamic_label_path='data/hand_gesture/dynamic_labels.csv')
    label_map = {str(i + 1): gesture for i, gesture in enumerate(label_handler.dynamic_labels.keys())}

    no_sequences = 1
    sequence_length = 30
    data_queue = queue.Queue()  # Khởi tạo hàng đợi
    data_collector = None   
    
    while True:
        frame = camera_service.capture_frame()
        if frame is None:
            break

        hands, _ = hand_handler.find_hands(frame, isDraw=True)

        if current_label:
            cv2.putText(frame, f"Current Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị khung hình lên cửa sổ camera
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

    if data_collector is not None:
        data_collector.stop()
        data_collector.join()

    save_dataframe_to_csv(df, csv_output_path)

    camera_service.release()

if __name__ == "__main__":
    capture_dynamic_gesture('data/hand_gesture/dynamic_data.csv')
