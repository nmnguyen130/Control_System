import pandas as pd

class LabelHandler:
    def __init__(self, label_file):
        self.labels = self.load_labels(label_file)
        self.reverse_labels = {value['value']: key for key, value in self.labels.items()}

    def load_labels(self, label_file):
        """
        Đọc file CSV và tải label vào dictionary.
        """
        df = pd.read_csv(label_file)
        return {
            row['gesture']: {
                'value': row['value'],
                'control': row['control'],
                'type': row['type']
            } for _, row in df.iterrows()
        }

    def get_control_by_gesture(self, gesture):
        """
        Lấy hành động điều khiển dựa vào cử chỉ.
        """
        return self.labels.get(gesture, {}).get('control', None)
    
    def get_type_by_gesture(self, gesture):
        """
        Lấy loại cử chỉ (static/dynamic) dựa vào cử chỉ.
        """
        return self.labels.get(gesture, {}).get('type', None)

    def get_value_by_gesture(self, gesture):
        """
        Lấy giá trị dựa vào cử chỉ.
        """
        return self.labels.get(gesture, {}).get('value', None)

    def get_gesture_by_value(self, value):
        """
        Lấy cử chỉ dựa vào giá trị.
        """
        return self.reverse_labels.get(value, None)
    
    def get_static_gestures(self):
        """
        Lấy danh sách các cử chỉ tĩnh.
        """
        return {gesture: details for gesture, details in self.labels.items() if details['type'] == 'static'}

    def get_dynamic_gestures(self):
        """
        Lấy danh sách các cử chỉ động.
        """
        return {gesture: details for gesture, details in self.labels.items() if details['type'] == 'dynamic'}