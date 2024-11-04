import pandas as pd
from src.handlers.base_handler import BaseCommandHandler

class BaseGestureLabelHandler(BaseCommandHandler):
    def load_commands(self, command_file):
        """Load commands from CSV file"""
        if not command_file:
            return {}
        df = pd.read_csv(command_file)
        return {
            row['gesture']: {
                'value': row['value'],
                'action': row['control']  # renamed from 'control' to match base class
            } for _, row in df.iterrows()
        }

class GestureLabelHandler(BaseGestureLabelHandler):
    def __init__(self, static_label_path=None, dynamic_label_path=None):
        self.static_handler = BaseGestureLabelHandler(static_label_path)
        self.dynamic_handler = BaseGestureLabelHandler(dynamic_label_path)

    # Delegate static gesture methods
    def get_static_value_by_gesture(self, gesture):
        return self.static_handler.get_value_by_command(gesture)

    def get_static_action_by_gesture(self, gesture):
        return self.static_handler.get_action_by_command(gesture)

    def get_static_gesture_by_value(self, value):
        return self.static_handler.get_command_by_value(value)

    def get_all_static_gestures(self):
        return self.static_handler.get_all_commands()

    # Delegate dynamic gesture methods
    def get_dynamic_value_by_gesture(self, gesture):
        return self.dynamic_handler.get_value_by_command(gesture)

    def get_dynamic_action_by_gesture(self, gesture):
        return self.dynamic_handler.get_action_by_command(gesture)

    def get_dynamic_gesture_by_value(self, value):
        return self.dynamic_handler.get_command_by_value(value)

    def get_all_dynamic_gestures(self):
        return self.dynamic_handler.get_all_commands()