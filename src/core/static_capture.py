import os
import cv2
import pandas as pd
from src.utils.file_utils import create_directory_if_not_exists
from src.utils.data_utils import initialize_dataframe, save_dataframe_to_csv
from src.services.camera_service import CameraService
from src.handlers.label_handler import LabelHandler

def capture_static_gesture(csv_output_path):
    """
    Capture data from camera, extract landmarks, and save to a CSV file.
    - Press 's' to capture data.
    - Press 'q' to quit.
    """

    create_directory_if_not_exists(os.path.dirname(csv_output_path))

    camera_service = CameraService()

    # Tạo dataframe để lưu điểm đặc trưng
    columns = [f'point_{i}_{axis}' for i in range(1, 22) for axis in ['x', 'y', 'z']] + ['label']  # Tổng cộng 64 cột
    df = initialize_dataframe(columns, csv_output_path)
    current_label = None

    label_handler = LabelHandler('data/hand_gesture/labels.csv')

    label_map = {str(i + 1): gesture for i, gesture in enumerate(label_handler.get_static_gestures().keys())}
    
    while True:
        frame = camera_service.capture_frame()
        if frame is None:
            break

        landmarks_list = camera_service.get_landmarks(frame)

        if current_label:
            cv2.putText(frame, f"Current Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Hiển thị khung hình lên cửa sổ camera
        cv2.imshow('Capture Hand Gesture', frame)

        # Nhấn phím để điều khiển
        key = cv2.waitKey(1) & 0xFF

        # Nhấn phím số (1-7) để chọn nhãn cho dữ liệu
        if chr(key) in label_map:
            current_label = label_map[chr(key)]
            print(f"Nhãn hiện tại: {current_label}")

        # Nhấn 's' để chụp ảnh và ghi dữ liệu
        elif key == ord('s'):
            if current_label and landmarks_list:
                for landmarks in landmarks_list:
                    landmarks = landmarks.tolist()
                    landmarks.append(current_label)
                    # Lưu vào DataFrame
                    df = pd.concat([df, pd.DataFrame([landmarks], columns=df.columns)], ignore_index=True)
                    print(f"Đã ghi dữ liệu cho nhãn: {current_label}")

        # Nhấn 'q' để thoát
        elif key == ord('q'):
            break

    save_dataframe_to_csv(df, csv_output_path)

    camera_service.release()

if __name__ == "__main__":
    capture_static_gesture('data/hand_gesture/data_static.csv')