import numpy as np
import torch
import torch.nn.functional as F
from src.modules.gesture_detection import StaticGestureModel, DynamicGestureModel
from src.handlers.label_handler import LabelHandler

# Dynamic gesture manager
class GestureManager:
    def __init__(self, lstm_model, sequence_length=30, threshold=0.05):
        self.lstm_model = lstm_model
        self.threshold = threshold  # Ngưỡng xác định cử chỉ động
        self.sequence_length = sequence_length  # Kích thước chuỗi động tối đa

        # Mảng để phát hiện động và lưu chuỗi
        self.dynamic_gesture = []
        self.gesture_active = False  # Trạng thái chuỗi động

    def is_dynamic_gesture(self, prev_points, curr_points):
        """Kiểm tra cử chỉ động bằng cách so sánh khoảng cách"""
        prev_points_reshaped = prev_points.reshape(-1, 3)
        curr_points_reshaped = curr_points.reshape(-1, 3)
        
        euclidean_dist = np.linalg.norm(curr_points_reshaped - prev_points_reshaped, axis=1)
        return np.any(euclidean_dist >= self.threshold)

    def process_frame(self, curr_points):
        if not self.dynamic_gesture:
            self.dynamic_gesture.append(curr_points)
            return None

        if self.is_dynamic_gesture(self.dynamic_gesture[-1], curr_points):
            if not self.gesture_active:
                self.start_gesture()
            self.dynamic_gesture.append(curr_points)
            
            if len(self.dynamic_gesture) > self.sequence_length:
                self.dynamic_gesture.pop(0)  # Loại khung cũ nhất

            if len(self.dynamic_gesture) >= self.sequence_length // 2:
                return self.process_dynamic_gesture()

        else:  # Nếu không còn động
            if self.gesture_active:
                return self.end_gesture()  # Kết thúc khi không còn chuyển động
        return None

    def start_gesture(self):
        print("Cử chỉ động bắt đầu.")
        self.gesture_active = True

    def end_gesture(self):
        print("Cử chỉ động kết thúc.")
        self.gesture_active = False

        gesture_index = self.process_dynamic_gesture()
        self.dynamic_gesture.clear()  # Xóa chuỗi sau khi xử lý
        return gesture_index

    def process_dynamic_gesture(self):
        input_tensor = torch.cat(self.dynamic_gesture, dim=0).unsqueeze(0)  # Shape: (1, seq_len, 63)
        lengths = torch.tensor([input_tensor.size(1)])  # Length of the sequence

        with torch.no_grad():
            output = self.lstm_model(input_tensor, lengths)
            gesture_index = torch.argmax(output.data, dim=1).item()
            return gesture_index

class HandGestureService:
    def __init__(self, static_model_path, dynamic_model_path, static_label_path, dynamic_label_path, device='cpu'):
        self.device = device

        self.label_handler = LabelHandler(static_label_path, dynamic_label_path)
        self.static_model = self.load_model(static_model_path, StaticGestureModel(len(self.label_handler.static_labels)))
        self.dynamic_model = self.load_model(dynamic_model_path, DynamicGestureModel(len(self.label_handler.dynamic_labels)))

        self.gesture_manager = GestureManager(self.dynamic_model)

    def load_model(self, model_path, model):
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        model.eval()
        return model.to(self.device)
    
    def predict_static(self, landmarks):
        points_tensor = torch.tensor(landmarks, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.static_model(points_tensor)
            probabilities = F.softmax(output, dim=1)
            gesture_index = torch.argmax(probabilities, dim=1).item()
            gesture_name = self.get_static_gesture_name(gesture_index)

            if probabilities[0][gesture_index] < 0.9:
                return None, "No Gesture"
            
            return gesture_index, gesture_name
    
    def predict_dynamic(self, landmarks):
        points_tensor = torch.tensor(np.array(landmarks), dtype=torch.float32).unsqueeze(0)  # Shape: (1, 63)
        gesture_index = self.gesture_manager.process_frame(points_tensor)

        if gesture_index is not None:
            gesture_name = self.get_dynamic_gesture_name(gesture_index)
            return gesture_index, gesture_name
        return None, None  # Return None if no gesture is detected
    
    def get_static_gesture_name(self, index):
        return self.label_handler.get_static_gesture_by_value(index)
    
    def get_dynamic_gesture_name(self, index):
        return self.label_handler.get_dynamic_gesture_by_value(index)