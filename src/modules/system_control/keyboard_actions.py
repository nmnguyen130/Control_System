import pyautogui

class KeyboardActions:
    def __init__(self):
        pass

    def press_key(self, key):
        pyautogui.press(key)

    def type_string(self, string):
        pyautogui.write(string)

    def copy(self):
        pyautogui.hotkey('ctrl', 'c')

    def paste(self):
        pyautogui.hotkey('ctrl', 'v')
