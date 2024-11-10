import queue
import threading
import tkinter as tk
from tkinter import scrolledtext
from src.core.gesture_detector import GestureDetector
from src.core.speech_listener import SpeechListener

class AssistantApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Modal Voice and Gesture Assistant")
        self.root.geometry("500x600")

        # Message queue for thread-safe communication
        self.message_queue = queue.Queue()

        # Main frame to hold the canvas and scrollbar
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=1)

        # Create a canvas widget
        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Add a scrollbar to the canvas
        self.scrollbar = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Create a frame inside the canvas to hold the message labels
        self.canvas_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.canvas_frame, anchor="nw")

        # Entry box for user input
        self.entry = tk.Entry(root, font=("Arial", 14))
        self.entry.pack(pady=10, padx=10, fill=tk.X)
        self.entry.bind("<Return>", self.on_enter)

        # Start Button for voice command
        start_voice_button = tk.Button(root, text="Start Voice Command", command=self.start_voice_thread, font=("Arial", 12))
        start_voice_button.pack(pady=5)

        # Start Button for gesture detection
        start_gesture_button = tk.Button(root, text="Start Gesture Detection", command=self.start_gesture_thread, font=("Arial", 12))
        start_gesture_button.pack(pady=5)

        # Initialize modules
        self.gesture_detection = GestureDetector(self.queue_message)
        self.speech_listener = SpeechListener(self.queue_message)
        self.gesture_thread_running = False
        self.voice_thread_running = False
        
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
                self.display_message(speaker, message)
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.process_messages)

    def display_message(self, speaker, message):
        """Display messages in a chat-like layout"""
        msg_frame = tk.Frame(self.canvas_frame, bg="white")
        msg_frame.pack(fill=tk.X, pady=5, padx=10)

        if speaker == "AI":  # AI messages on the left
            label = tk.Label(msg_frame, text=message, bg="lightblue", anchor="w", font=("Arial", 12))
            label.pack(side=tk.LEFT, fill=tk.X, expand=1)
        else:  # User messages on the right
            label = tk.Label(msg_frame, text=message, bg="lightgreen", anchor="e", font=("Arial", 12))
            label.pack(side=tk.RIGHT, fill=tk.X, expand=1)

    def on_enter(self, event):
        user_input = self.entry.get()
        if user_input:
            self.queue_message("You", user_input)
            self.entry.delete(0, tk.END)

    def start_voice_thread(self):
        if not self.voice_thread_running:
            self.voice_thread_running = True
            thread = threading.Thread(target=self.speech_listener.start_listening, daemon=True)
            thread.start()

    def start_gesture_thread(self):
        if not self.gesture_thread_running:
            self.gesture_thread_running = True
            thread = threading.Thread(target=self.gesture_detection.start_detection, daemon=True)
            thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantApp(root)
    root.mainloop()
