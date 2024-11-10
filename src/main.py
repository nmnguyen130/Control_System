import queue
import threading
import tkinter as tk
from tkinter import ttk
from src.core.gesture_detector import GestureDetector
from src.core.speech_listener import SpeechListener

class AssistantApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Multi-Modal Voice and Gesture Assistant")
        self.geometry("500x600")

        # Message queue for thread-safe communication
        self.message_queue = queue.Queue()

        # Main frame to hold the canvas and scrollbar
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=1)

        # Create a canvas widget
        self.canvas = tk.Canvas(self.main_frame, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

        # Add a scrollbar to the canvas
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure the canvas
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        # Create a frame inside the canvas to hold the message labels
        self.chat_frame = tk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")

        # Start Button for voice command
        start_voice_button = tk.Button(self, text="Start Voice Command", command=self.start_voice_thread, font=("Arial", 12))
        start_voice_button.pack(pady=5)

        # Start Button for gesture detection
        start_gesture_button = tk.Button(self, text="Start Gesture Detection", command=self.start_gesture_thread, font=("Arial", 12))
        start_gesture_button.pack(pady=5)

        # Initialize modules
        self.gesture_detection = GestureDetector(self.queue_message)
        self.speech_listener = SpeechListener(self.queue_message)
        self.gesture_thread_running = threading.Event()
        self.voice_thread_running = threading.Event()

        self.bind("<Configure>", self.on_resize)
        self.canvas.bind_all("<MouseWheel>", self.on_mouse_wheel)
        
        # Start message processing
        self.process_messages()

    def on_mouse_wheel(self, event):
        """Scroll canvas vertically based on mouse wheel movement"""
        scroll_units = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(scroll_units, "units")

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
            self.after(100, self.process_messages)

    def display_message(self, speaker, message):
        """Display messages in a chat-like layout"""
        msg_frame = tk.Frame(self.chat_frame, bg="white")
        msg_frame.pack(fill=tk.X, pady=5, padx=10)

        # Determine background color and text alignment
        bg_color = "lightblue" if speaker == "AI" else "lightgreen"
        anchor = "w" if speaker == "AI" else "e"

        label = tk.Label(msg_frame, text=message, bg=bg_color, anchor=anchor, wraplength=self.chat_frame.winfo_width(), justify="left")
        label.pack(side=tk.LEFT if speaker == "AI" else tk.RIGHT, fill=tk.X, expand=1)

        self._update_chat_frame_width()

    def on_resize(self, event):
        self._update_chat_frame_width()

    def _update_chat_frame_width(self):
        canvas_width = self.canvas.winfo_width()
        self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        self.chat_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        for label in self.chat_frame.winfo_children():
            for widget in label.winfo_children():
                widget.configure(wraplength=canvas_width - 100)

    def start_voice_thread(self):
        if not self.voice_thread_running.is_set():
            self.voice_thread_running.set()
            thread = threading.Thread(target=self.speech_listener.start_listening, daemon=True)
            thread.start()

    def start_gesture_thread(self):
        if not self.gesture_thread_running.is_set():
            self.gesture_thread_running.set()
            thread = threading.Thread(target=self.gesture_detection.start_detection, daemon=True)
            thread.start()

if __name__ == "__main__":
    app = AssistantApp()
    app.mainloop()
