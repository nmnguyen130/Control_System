import speech_recognition as sr
import pyttsx3
import torch
import numpy as np
import pyautogui
import os
from datetime import datetime
from .speech_model import SpeechRecognitionModel

class VoiceAssistant:
    def __init__(self, callback, model_path='trained_data/speech_model.pth'):
        self.callback = callback
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.is_listening = False
        self.command_history = []
        
        # Configure text-to-speech
        self.setup_tts()
        
        # Load the speech recognition model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
        # Command mappings
        self.commands = {
            'open': self.open_application,
            'search': self.web_search,
            'volume': self.adjust_volume,
            'screenshot': self.take_screenshot,
            'type': self.type_text,
            'history': self.show_history,
            'help': self.show_help
        }
        
    def setup_tts(self):
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('rate', 150)    # Speed of speech
        self.engine.setProperty('volume', 0.9)  # Volume (0-1)
        # Set voice (uncomment to use different voice)
        # self.engine.setProperty('voice', voices[1].id)
        
    def start_voice_command(self):
        self.is_listening = True
        self.callback("Assistant", "Voice command activated. Say 'help' for available commands.")
        
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            
            while self.is_listening:
                try:
                    audio = self.recognizer.listen(source, timeout=5)
                    text = self.recognizer.recognize_google(audio)
                    self.callback("You", text)
                    
                    response = self.process_command(text)
                    self.speak(response)
                    self.callback("Assistant", response)
                    
                    # Save command to history
                    self.command_history.append({
                        'timestamp': datetime.now(),
                        'command': text,
                        'response': response
                    })
                    
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    self.callback("Assistant", "Sorry, there was an error with the speech service")
                    break
    
    def process_command(self, text):
        text = text.lower()
        
        # Basic commands
        if text == 'help':
            return self.show_help()
        elif text == 'exit' or text == 'stop':
            self.stop_voice_command()
            return "Goodbye!"
            
        # Process complex commands
        for cmd in self.commands:
            if cmd in text:
                return self.commands[cmd](text)
                
        return "I'm not sure how to help with that. Say 'help' for available commands."
    
    def open_application(self, text):
        apps = {
            'chrome': 'google-chrome',
            'firefox': 'firefox',
            'terminal': 'gnome-terminal',
            'calculator': 'gnome-calculator'
        }
        
        for app in apps:
            if app in text:
                os.system(f"nohup {apps[app]} &")
                return f"Opening {app}"
        return "Application not found"
    
    def web_search(self, text):
        search_terms = text.replace('search', '').strip()
        os.system(f'xdg-open "https://www.google.com/search?q={search_terms}"')
        return f"Searching for {search_terms}"
    
    def adjust_volume(self, text):
        try:
            if 'up' in text:
                os.system("amixer -D pulse sset Master 10%+")
                return "Volume increased"
            elif 'down' in text:
                os.system("amixer -D pulse sset Master 10%-")
                return "Volume decreased"
            elif 'mute' in text:
                os.system("amixer -D pulse sset Master mute")
                return "Volume muted"
        except:
            return "Could not adjust volume"
    
    def take_screenshot(self, text):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        try:
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            return f"Screenshot saved as {filename}"
        except:
            return "Could not take screenshot"
    
    def type_text(self, text):
        text_to_type = text.replace('type', '').strip()
        pyautogui.write(text_to_type)
        return f"Typed: {text_to_type}"
    
    def show_history(self, text):
        if not self.command_history:
            return "No command history available"
            
        recent_commands = self.command_history[-5:]  # Show last 5 commands
        history_text = "Recent commands:\n"
        for cmd in recent_commands:
            history_text += f"- {cmd['timestamp'].strftime('%H:%M:%S')}: {cmd['command']}\n"
        return history_text
    
    def show_help(self, text=None):
        help_text = """Available commands:
- open [application]: Opens specified application
- search [terms]: Performs web search
- volume up/down/mute: Controls system volume
- screenshot: Takes a screenshot
- type [text]: Types the specified text
- history: Shows command history
- help: Shows this help message
- exit/stop: Stops voice command"""
        return help_text