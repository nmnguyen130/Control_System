import pyautogui

class SystemActions:
    def __init__(self):
        pass

    def volume_up(self):
        pyautogui.press('volumeup')

    def volume_down(self):
        pyautogui.press('volumedown')

    def switch_window(self):
        pyautogui.hotkey('alt', 'tab')

    def lock_screen(self):
        pyautogui.hotkey('ctrl', 'l')

    def sleep_screen(self):
        pyautogui.hotkey('ctrl', 'shift', 'power')

    def restart_system(self):
        pyautogui.hotkey('ctrl', 'shift', 'esc')

    def shutdown_system(self):
        pyautogui.hotkey('ctrl', 'shift', 'power')
