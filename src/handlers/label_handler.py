import pandas as pd

class LabelHandler:
    def __init__(self, static_label_path=None, dynamic_label_path=None):
        self.static_labels = self.load_labels(static_label_path) if static_label_path else {}
        self.dynamic_labels = self.load_labels(dynamic_label_path) if dynamic_label_path else {}
        self.reverse_static_labels = {value['value']: key for key, value in self.static_labels.items()}
        self.reverse_dynamic_labels = {value['value']: key for key, value in self.dynamic_labels.items()}

    def load_labels(self, label_file):
        """
        Đọc file CSV và tải label vào dictionary.
        """
        df = pd.read_csv(label_file)
        return {
            row['gesture']: {
                'value': row['value'],
                'control': row['control']
            } for _, row in df.iterrows()
        }
    
    def get_static_value_by_gesture(self, gesture):
        """
        Lấy giá trị (value) dựa vào cử chỉ tĩnh.
        """
        return self.static_labels.get(gesture, {}).get('value', None)

    def get_dynamic_value_by_gesture(self, gesture):
        """
        Lấy giá trị (value) dựa vào cử chỉ động.
        """
        return self.dynamic_labels.get(gesture, {}).get('value', None)
    
    def get_static_control_by_gesture(self, gesture):
        """
        Lấy hành động điều khiển dựa vào cử chỉ tĩnh.
        """
        return self.static_labels.get(gesture, {}).get('control', None)

    def get_dynamic_control_by_gesture(self, gesture):
        """
        Lấy hành động điều khiển dựa vào cử chỉ động.
        """
        return self.dynamic_labels.get(gesture, {}).get('control', None)

    def get_static_gesture_by_value(self, value):
        """
        Lấy cử chỉ tĩnh dựa vào giá trị.
        """
        return self.reverse_static_labels.get(value, None)

    def get_dynamic_gesture_by_value(self, value):
        """
        Lấy cử chỉ động dựa vào giá trị.
        """
        return self.reverse_dynamic_labels.get(value, None)

    def get_all_static_gestures(self):
        """
        Lấy tất cả các cử chỉ tĩnh.
        """
        return self.static_labels

    def get_all_dynamic_gestures(self):
        """
        Lấy tất cả các cử chỉ động.
        """
        return self.dynamic_labels