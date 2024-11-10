from datetime import datetime
import os
import random
import webbrowser
import pyautogui
from src.modules.voice_assistant.intent_dataset import IntentDataset
from src.services.audio_service import AudioControl

class IntentMethod:
    def __init__(self):
        self.intent_dataset = IntentDataset('data/speech/intents.json')
        self.audio_control = AudioControl()

        self.method_mappings = {
            "greeting": lambda: self.get_random_response("greeting"),
            "web_search": self.web_search,
            "open_browser": lambda: self.open_application("browser"),
            "open_calculator": lambda: self.open_application("calculator"),
            "open_file_explorer": lambda: self.open_application("file_explorer"),
            "open_notepad": lambda: self.open_application("notepad"),
            "open_task_manager": lambda: self.open_application("task_manager"),
            "take_screenshot": self.take_screenshot,
            "minimize_window": lambda: self.manage_window("minimize"),
            "maximize_window": lambda: self.manage_window("maximize"),
            "close_window": lambda: self.manage_window("close"),
            "shutdown": lambda: self.system_control("shutdown"),
            "restart_computer": lambda: self.system_control("restart"),
            "increase_volume": self.audio_control.increase_volume,
            "decrease_volume": self.audio_control.decrease_volume,
            "mute_audio": self.audio_control.mute,
            "unmute_audio": self.audio_control.unmute,
            "goodbye": lambda: self.get_random_response("goodbye"),
            "help": self.show_help,
            "history": self.show_history,
            "input_text": self.input_text,
        }
        self.command_history = []
        
    def handle_intent(self, intent_tag, text=None):
        print(intent_tag)
        # Check if the intent tag has a mapped method and call it
        if intent_tag in self.method_mappings:
            if intent_tag in ["web_search", "input_text"]:
                result = self.method_mappings[intent_tag](text)
            else:
                result = self.method_mappings[intent_tag]()
            self.command_history.append({
                "timestamp": datetime.now(),
                "command": intent_tag
            })
            return result
        return "I'm not sure how to respond to that."

    def web_search(self, text):
        """Perform a web search for specified terms."""
        search_terms = text.replace('search', '').strip()
        if search_terms:
            webbrowser.open(f"https://www.google.com/search?q={search_terms}")
            return self.get_random_response("web_search", search_terms)
        return "Please specify search terms."
    
    def open_application(self, app_name):
        """Opens specified applications based on provided name."""
        commands = {
            "browser": lambda: webbrowser.open("https://www.google.com"),
            "calculator": lambda: os.system("calc"),
            "file_explorer": lambda: os.system("explorer"),
            "notepad": lambda: os.system("notepad"),
            "task_manager": lambda: os.system("taskmgr")
        }
        if app_name in commands:
            commands[app_name]()
            return self.get_random_response(f"open_{app_name}", app_name)
        return "Application not found."
        
    def take_screenshot(self, _=None):
        """Capture and save a screenshot with a timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            return self.get_random_response("take_screenshot", filename)
        except Exception as e:
            return f"Could not take screenshot: {e}"
        
    def manage_window(self, action):
        """Manages the active window, supporting minimize or close actions."""
        try:
            if action == "minimize":
                pyautogui.hotkey('win', 'down')
                return self.get_random_response(f"{action}_window", "minimize")
            if action == "maximize":
                pyautogui.hotkey('win', 'up')
                return self.get_random_response(f"{action}_window", "maximize")
            elif action == "close":
                pyautogui.hotkey('alt', 'f4')
                return self.get_random_response(f"{action}_window", "close")
        except Exception as e:
            return f"Could not perform window action: {e}"
        return "Invalid window action."
    
    def system_control(self, action):
        """Controls system power actions like shutdown and restart."""
        try:
            if action == "shutdown":
                os.system("shutdown /s /t 1")
                return self.get_random_response(f"{action}_computer", "shutdown")
            elif action == "restart":
                os.system("shutdown /r /t 1")
                return self.get_random_response(f"{action}_computer", "shutdown")
        except Exception as e:
            return f"Could not perform system action: {e}"
        return "Invalid system action."
    
    def input_text(self, text):
        """Types the specified text."""
        text_to_type = text.replace('type', '').strip()
        pyautogui.write(text_to_type)
        return self.get_random_response("input_text", text_to_type)
    
    def show_history(self):
        if not self.command_history:
            return "No command history available"
            
        recent_commands = self.command_history[-5:]  # Show last 5 commands
        history_text = "Recent commands:\n"
        for cmd in recent_commands:
            history_text += f"- {cmd['timestamp'].strftime('%H:%M:%S')}: {cmd['command']}\n"
        return history_text
    
    def show_help(self):
        help_text = """
        Available commands:
        - greeting: Greet the assistant
        - search [terms]: Performs web search
        - open [app]: Opens specified application (browser, calculator, file_explorer, notepad, task_manager)
        - volume [up/down/mute]: Controls system volume
        - window [minimize/close]: Manages window actions
        - system [shutdown/restart]: Controls power actions
        - screenshot: Takes a screenshot
        - type [text]: Types the specified text
        - history: Shows recent command history
        - help: Shows this help message
        - goodbye/exit: Ends the session
        """
        return help_text.strip()
    
    def get_random_response(self, action_type, extra_info=None):
        """Fetches a random response for the given action type."""
        for intent in self.intent_dataset.intents_data["intents"]:
            if intent["tag"] == action_type:
                response = random.choice(intent["responses"])
                # Optionally, customize the response with additional information
                if extra_info:
                    response = response.format(extra_info)
                return response
        return "I don't have a response for this action."