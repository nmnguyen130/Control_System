import math
import pyautogui

class MouseActions:
    def __init__(self):
        self.is_dragging = False
        pyautogui.FAILSAFE = True  # Enable fail-safe feature
        
        # Configuration settings
        self.scroll_amount = 200  # Scroll speed
        self.move_speed = 1.2    # Reduced speed for smoother movement
        self.smoothing_factor = 0.65  # Smoothing factor for movement (0-1)
        self.drag_threshold = 5    # Minimum pixels to start drag
        
        # Get screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        self.smooth_dx = 0
        self.smooth_dy = 0
        
    def move_cursor(self, dx, dy):
        current_x, current_y = pyautogui.position()
        
        # Calculate the distance to move based on hand movement and screen dimensions
        dx_scaled = dx * self.screen_width * self.move_speed
        dy_scaled = dy * self.screen_height * self.move_speed

        # Apply smoothing to the movement
        self.smooth_dx = self.smooth_dx * self.smoothing_factor + dx_scaled * (1 - self.smoothing_factor)
        self.smooth_dy = self.smooth_dy * self.smoothing_factor + dy_scaled * (1 - self.smoothing_factor)
        
        new_x = current_x + int(self.smooth_dx)
        new_y = current_y + int(self.smooth_dy)
        
        new_x = max(0, min(new_x, self.screen_width - 1))
        new_y = max(0, min(new_y, self.screen_height - 1))

        try:
            pyautogui.moveTo(new_x, new_y, _pause=False)
        except pyautogui.FailSafeException:
            print("Failsafe triggered - mouse movement cancelled")

    def stop_cursor(self):
        if self.is_dragging:
            self.end_drag()

    def click_mouse(self, button='left'):
        try:
            pyautogui.click(button=button)
        except pyautogui.FailSafeException:
            print(f"Failsafe triggered - {button} click cancelled")

    def double_click_mouse(self):
        try:
            pyautogui.doubleClick()
        except pyautogui.FailSafeException:
            print("Failsafe triggered - double click cancelled")

    def right_click_mouse(self):
        try:
            pyautogui.rightClick()
        except pyautogui.FailSafeException:
            print("Failsafe triggered - right click cancelled")

    def scroll_up(self):
        try:
            pyautogui.scroll(self.scroll_amount)
        except pyautogui.FailSafeException:
            print("Failsafe triggered - scroll up cancelled")

    def scroll_down(self):
        try:
            pyautogui.scroll(-self.scroll_amount)
        except pyautogui.FailSafeException:
            print("Failsafe triggered - scroll down cancelled")

    def start_drag(self):
        if not self.is_dragging:
            try:
                pyautogui.mouseDown()
                self.is_dragging = True
            except pyautogui.FailSafeException:
                print("Failsafe triggered - drag start cancelled")

    def end_drag(self):
        if self.is_dragging:
            try:
                pyautogui.mouseUp()
                self.is_dragging = False
            except pyautogui.FailSafeException:
                print("Failsafe triggered - drag end cancelled")

    def drag_mouse(self, dx, dy):
        if self.is_dragging:
            try:
                screen_x = int(dx * self.screen_width)
                screen_y = int(dy * self.screen_height)
                pyautogui.dragTo(screen_x, screen_y, duration=0.1, _pause=False)
            except pyautogui.FailSafeException:
                print("Failsafe triggered - drag movement cancelled")
                self.end_drag()

    def open_menu(self):
        self.right_click_mouse()

    def set_scroll_speed(self, speed):
        self.scroll_amount = speed

    def set_move_speed(self, speed):
        self.move_speed = speed
