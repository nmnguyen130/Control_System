from src.modules.system_control.mouse_actions import MouseActions
from src.modules.system_control.keyboard_actions import KeyboardActions
from src.modules.system_control.system_actions import SystemActions

class GestureController:
    def __init__(self):
        self.mouse_actions = MouseActions()
        self.keyboard_actions = KeyboardActions()
        self.system_actions = SystemActions()
        self.interaction_active = False

    def handle_gesture(self, control_action):
        if control_action == "Start Interaction":
            self.interaction_active = True
            print("Interaction started")
        elif control_action == "End Interaction":
            self.interaction_active = False
            print("Interaction ended")
        elif self.interaction_active:
            if control_action in ["Confirm Selection", "Reject Selection", "Hold State"]:
                self.control_system(control_action)
            elif control_action in ["Select Object", "Drag Object"]:
                self.control_mouse(control_action)

    def control_mouse(self, control_action):
        if control_action == "Select Object":
            self.mouse_actions.move_cursor()
        elif control_action == "Drag Object":
            self.mouse_actions.drag_object()

    def control_keyboard(self, control_action):
        pass

    def control_system(self, control_action):
        if control_action == "Confirm Selection":
            self.system_actions.confirm_selection()
        elif control_action == "Reject Selection":
            self.system_actions.reject_selection()
