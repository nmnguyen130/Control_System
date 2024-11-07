import tkinter as tk
from tkinter import scrolledtext
from src.core.gesture_detector import GestureDetector
import threading
import queue

class AssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Modal Voice and Gesture Assistant")
        self.root.geometry("400x500")

        # Message queue for thread-safe communication
        self.message_queue = queue.Queue()

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

        # Initialize modules
        self.gesture_detection = GestureDetector(self.queue_message)
        self.gesture_thread_running = False
        
        # Start message processing
        self.process_messages()

    def queue_message(self, speaker, message):
        """Thread-safe message queueing"""
        self.message_queue.put((speaker, message))

    def process_messages(self):
        """Process messages from queue and update UI"""
        try:
            while True:
                speaker, message = self.message_queue.get_nowait()
                self.text_area.config(state='normal')
                self.text_area.insert(tk.END, f"{speaker}: {message}\n")
                self.text_area.yview(tk.END)
                self.text_area.config(state='disabled')
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.process_messages)

    def on_enter(self, event):
        user_input = self.entry.get()
        if user_input:
            self.queue_message("You", user_input)
            self.entry.delete(0, tk.END)

    def start_voice_thread(self):
        pass  # Voice assistant functionality removed to reduce complexity

    def start_gesture_thread(self):
        if not self.gesture_thread_running:
            self.gesture_thread_running = True
            thread = threading.Thread(target=self.gesture_detection.start_detection, daemon=True)
            thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantApp(root)
    root.mainloop()