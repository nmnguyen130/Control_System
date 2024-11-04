import numpy as np
import torch
import torch.nn.functional as F
from src.modules.gesture_detection import StaticGestureModel, DynamicGestureModel

# Dynamic gesture manager
class GestureManager:
    def __init__(self, lstm_model, sequence_length=30, threshold=0.05, cooldown_frames=5):
        self.lstm_model = lstm_model
        self.gesture_active = False
        self.dynamic_gesture = []
        self.sequence_length = sequence_length

        self.threshold = threshold
        self.cooldown_frames = cooldown_frames
        self.inactive_frames = 0

    def is_dynamic_gesture(self, prev_points, curr_points):
        points_diff = curr_points.reshape(-1, 3) - prev_points.reshape(-1, 3)
        euclidean_dist = np.linalg.norm(points_diff, axis=1)
        velocity = euclidean_dist / 1.0  # Assuming time between frames is 1.0
        return np.any(euclidean_dist >= self.threshold) or np.any(velocity >= self.threshold * 2)

    def process_frame(self, curr_points):
        if not self.dynamic_gesture:
            self.dynamic_gesture.append(curr_points)
            return None

        if self.is_dynamic_gesture(self.dynamic_gesture[-1], curr_points):
            self.inactive_frames = 0
            if not self.gesture_active:
                self.start_gesture()
            self.dynamic_gesture.append(curr_points)
            
            if len(self.dynamic_gesture) > self.sequence_length:
                self.dynamic_gesture.pop(0)

            return self.process_dynamic_gesture()
        else:
            self.inactive_frames += 1
            if self.gesture_active and self.inactive_frames > self.cooldown_frames:
                return self.end_gesture()
            elif len(self.dynamic_gesture) < self.sequence_length:
                self.dynamic_gesture.append(curr_points)

        if self.gesture_active:
            return self.process_dynamic_gesture()
        return None

    def start_gesture(self):
        print("Dynamic gesture started.")
        self.gesture_active = True

    def end_gesture(self):
        print("Dynamic gesture ended.")
        self.gesture_active = False

        gesture_index = self.process_dynamic_gesture()
        self.dynamic_gesture.clear()
        return gesture_index

    def process_dynamic_gesture(self):
        if len(self.dynamic_gesture) < 2:
            return None
        
        # Pad the sequence if it's shorter than sequence_length
        padded_sequence = self.dynamic_gesture + [self.dynamic_gesture[-1]] * (self.sequence_length - len(self.dynamic_gesture))
        
        input_tensor = torch.cat(padded_sequence, dim=0).unsqueeze(0)  # Shape: (1, seq_len, 63)

        with torch.no_grad():
            output = self.lstm_model(input_tensor)
            gesture_index = torch.argmax(output.data, dim=1).item()
            return gesture_index

class HandGestureService:
    def __init__(self, static_model_path, dynamic_model_path, label_handler, device='cpu'):
        self.device = device
        self.label_handler = label_handler

        self.static_model = self.load_model(static_model_path, StaticGestureModel(len(self.label_handler.get_all_static_gestures())))
        self.dynamic_model = self.load_model(dynamic_model_path, DynamicGestureModel(len(self.label_handler.get_all_dynamic_gestures())))

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
            if probabilities[0][gesture_index] < 0.9:
                return None, "No Gesture"
            return gesture_index, self.get_static_gesture_name(gesture_index)
    
    def predict_dynamic(self, landmarks):
        points_tensor = torch.tensor(np.array(landmarks), dtype=torch.float32).unsqueeze(0)  # Shape: (1, 63)
        gesture_index = self.gesture_manager.process_frame(points_tensor)

        if gesture_index is not None:
            return gesture_index, self.get_dynamic_gesture_name(gesture_index)
        return None, None

    def get_static_gesture_name(self, index):
        return self.label_handler.get_static_gesture_by_value(index)
    
    def get_dynamic_gesture_name(self, index):
        return self.label_handler.get_dynamic_gesture_by_value(index)
