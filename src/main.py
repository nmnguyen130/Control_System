import tkinter as tk
from tkinter import scrolledtext
# from core.voice_assistant import VoiceAssistant
from src.core.gesture_detector import GestureDetector
import threading

class AssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Modal Voice and Gesture Assistant")
        self.root.geometry("400x500")

        # Text area for displaying chat
        self.text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', font=("Arial", 12))
        self.text_area.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

        # Entry box for user input
        self.entry = tk.Entry(root, font=("Arial", 14))
        self.entry.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        self.entry.bind("<Return>", self.on_enter)

        # Start Button for voice command
        start_voice_button = tk.Button(root, text="Start Voice Command", command=self.start_voice_thread, font=("Arial", 12))
        start_voice_button.grid(row=2, column=0, pady=10)

        # Start Button for gesture detection
        start_gesture_button = tk.Button(root, text="Start Gesture Detection", command=self.start_gesture_thread, font=("Arial", 12))
        start_gesture_button.grid(row=3, column=0, pady=10)

        # Configure grid weights
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Khởi tạo các module
        self.gesture_detection = GestureDetector(self.add_message)
        # self.voice_assistant = VoiceAssistant(self.add_message)

    def add_message(self, speaker, message):
        self.text_area.config(state='normal')
        self.text_area.insert(tk.END, f"{speaker}: {message}\n")
        self.text_area.yview(tk.END)
        self.text_area.config(state='disabled')

    def on_enter(self, event):
        user_input = self.entry.get()
        if user_input:
            self.add_message("You", user_input)
            response = self.voice_assistant.respond(user_input)
            self.add_message("NL", response)
            self.entry.delete(0, tk.END)

    def start_voice_thread(self):
        threading.Thread(target=self.voice_assistant.start_voice_command, daemon=True).start()

    def start_gesture_thread(self):
        threading.Thread(target=self.gesture_detection.start_detection, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantApp(root)
    root.mainloop()