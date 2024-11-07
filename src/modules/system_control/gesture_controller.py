from src.modules.system_control.mouse_actions import MouseActions
from src.modules.system_control.keyboard_actions import KeyboardActions
from src.modules.system_control.system_actions import SystemActions

class GestureController:
    def __init__(self):
        self.mouse_actions = MouseActions()
        self.keyboard_actions = KeyboardActions()
        self.system_actions = SystemActions()
        self.interaction_active = True

    def handle_gesture(self, control_action, dx=None, dy=None):
        # if control_action == "Start Interaction":
        #     self.interaction_active = True
        #     print("Interaction started")
        # elif control_action == "End Interaction":
        #     self.interaction_active = False
        #     print("Interaction ended")
        if self.interaction_active:
            self.control_mouse(control_action, dx, dy)

    def control_mouse(self, control_action, dx=None, dy=None):
        if control_action == "stop_cursor":
            self.mouse_actions.stop_cursor()
        elif control_action == "move_cursor":
            self.mouse_actions.move_cursor(dx, dy)
        elif control_action == "left_click":
            self.mouse_actions.click_mouse('left')
        elif control_action == "right_click":
            self.mouse_actions.right_click_mouse()
        elif control_action == "double_click":
            self.mouse_actions.double_click_mouse()
        elif control_action == "scroll_up":
            self.mouse_actions.scroll_up()
        elif control_action == "scroll_down":
            self.mouse_actions.scroll_down()
        elif control_action == "drag_and_drop":
            self.mouse_actions.start_drag()
        elif control_action == "open_menu":
            self.mouse_actions.right_click_mouse()

    def control_keyboard(self, control_action):
        pass

    def control_system(self, control_action):
        if control_action == "Confirm Selection":
            self.system_actions.confirm_selection()
        elif control_action == "Reject Selection":
            self.system_actions.reject_selection()
