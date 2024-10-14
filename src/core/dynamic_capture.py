import os
import cv2
import pandas as pd
from src.utils.file_utils import create_directory_if_not_exists
from src.utils.data_utils import initialize_dataframe, save_dataframe_to_csv
from src.services.camera_service import CameraService
from src.handlers.hand_detect_handler import HandDetectHandler
from src.handlers.label_handler import LabelHandler

def capture_dynamic_gesture(csv_output_path):
    """
    Capture data for dynamic gestures, extract landmarks, and save to a CSV file.
    - Press 's' to start capturing dynamic gesture.
    - Press 'e' to stop capturing.
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

    capturing = False
    frame_counter = 0
    
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

        # Nhấn phím số (1-4) để chọn nhãn cho dữ liệu
        if chr(key) in label_map:
            current_label = label_map[chr(key)]
            print(f"Nhãn hiện tại: {current_label}")

        # Nhấn 's' để bắt đầu ghi dữ liệu cho động
        elif key == ord('s'):
            if current_label and not capturing:
                capturing = True  # Set the capturing flag
                print(f"Capturing data for gesture: {current_label}...")
                frame_counter = 0  # Reset frame counter at the start of capturing

        # Nhấn 'e' để dừng
        elif key == ord('e'):
            if capturing:
                capturing = False  # Reset the capturing flag
                print(f"Stopped capturing data for gesture: {current_label}")

        # Nhấn 'q' để thoát
        elif key == ord('q'):
            break

        if capturing:
            if hands:
                landmarks_list = [hand["lmList"] for hand in hands]
                for landmarks in landmarks_list:
                    landmarks = landmarks.tolist()
                    landmarks.append(current_label)
                    landmarks.insert(0, frame_counter)
                    # Append data to DataFrame
                    df = pd.concat([df, pd.DataFrame([landmarks], columns=df.columns)], ignore_index=True)

                    # Display the current frame
                    cv2.imshow('Capture Dynamic Gesture', frame)
                
                frame_counter += 1

    save_dataframe_to_csv(df, csv_output_path)

    camera_service.release()

if __name__ == "__main__":
    capture_dynamic_gesture('data/hand_gesture/dynamic_data.csv')
