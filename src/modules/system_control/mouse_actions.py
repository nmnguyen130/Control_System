import pyautogui

class MouseActions:
    def __init__(self):
        pass

    def move_mouse(self, x, y):
        pyautogui.moveTo(x, y)

    def click_mouse(self, button='left'):
        pyautogui.click(button=button)

    def scroll_mouse(self, scroll_amount):
        pyautogui.scroll(scroll_amount)

    def drag_mouse(self, x, y):
        pyautogui.dragTo(x, y)

    def double_click_mouse(self):
        pyautogui.doubleClick()

    def right_click_mouse(self):
        pyautogui.rightClick()

    def middle_click_mouse(self):
        pyautogui.middleClick()
