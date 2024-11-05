import os
import cv2
import pandas as pd
from src.utils.file_utils import create_directory_if_not_exists
from src.utils.data_utils import initialize_dataframe, save_dataframe_to_csv
from src.services.camera_service import CameraService
from src.handlers.hand_detect_handler import HandDetectHandler
from src.handlers.gesture_label_handler import GestureLabelHandler

def capture_static_gesture(csv_output_path):
    """
    Capture static hand gesture data from camera and save landmarks to CSV.
    Controls:
    - Number keys (1-...): Select gesture label
    - 's': Save current hand pose
    - 'q': Quit capture
    """

    create_directory_if_not_exists(os.path.dirname(csv_output_path))

    camera_service = CameraService()
    hand_handler = HandDetectHandler(num_hands=2)
    label_handler = GestureLabelHandler(static_label_path='data/hand_gesture/static_labels.csv')

    # Setup data storage
    columns = [f'point_{i}_{axis}' for i in range(1, 22) for axis in ['x', 'y', 'z']] + ['label']
    df = initialize_dataframe(columns, csv_output_path)
    label_map = {str(i + 1): gesture for i, gesture in enumerate(label_handler.get_all_static_gestures())}
    
    current_label = None
    save_count = 0

    try:
        while True:
            frame = camera_service.capture_frame()
            if frame is None:
                break

            hands, _ = hand_handler.find_hands(frame, isDraw=True)

            if current_label:
                cv2.putText(frame, f"Current Label: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Capture Hand Gesture', frame)
            key = cv2.waitKey(1) & 0xFF

            # Handle key inputs
            if chr(key) in label_map:
                current_label = label_map[chr(key)]
                save_count = 0
                print(f"Selected label: {current_label}")

            # Nhấn 's' để chụp ảnh và ghi dữ liệu
            elif key == ord('s'):
                if current_label and hands:
                    landmarks_list = [hand["lmList"] for hand in hands]  # Get landmarks from detected hands
                    for landmarks in landmarks_list:
                        landmarks = landmarks.tolist()
                        landmarks.append(current_label)

                        save_count += 1

                        # Save to DataFrame
                        df = pd.concat([df, pd.DataFrame([landmarks], columns=df.columns)], ignore_index=True)
                        print(f"Data saved for label: {current_label} (Count: {save_count})")

            elif key == ord('q'):
                break
            
    except Exception as e:
        print(f"Error during capture: {e}")
    finally:
        save_dataframe_to_csv(df, csv_output_path)
        camera_service.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_static_gesture('data/hand_gesture/static_data.csv')