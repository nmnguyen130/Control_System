import os
import threading
import queue
import sounddevice as sd
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import wave

class AudioRecorder:
    def __init__(self, sample_rate=16000, channels=1):
        """Initialize AudioRecorder with configurable parameters."""
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = False
        self.audio_queue = queue.Queue()
        self.format = np.int16
        self.stream = None
        self._lock = threading.Lock()  # Add thread safety

    def callback(self, indata, frames, time, status):
        if status:
            print(f'Audio callback error: {status}')
            return
            
        if self.recording:
            audio_int16 = (indata * 32767).astype(self.format)
            self.audio_queue.put_nowait(audio_int16.copy())
            
    def start_recording(self):
        with self._lock:
            if self.stream is not None:
                self.stop_recording()
                
            self.recording = True
            try:
                self.stream = sd.InputStream(
                    callback=self.callback,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    dtype=np.float32,
                    blocksize=1024,  # Optimize buffer size
                    latency='low'    # Reduce latency
                )
                self.stream.start()
            except Exception as e:
                self.recording = False
                raise RuntimeError(f"Failed to start recording: {str(e)}")
        
    def stop_recording(self):
        with self._lock:
            self.recording = False
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                finally:
                    self.stream = None

            audio_chunks = []
            try:
                while not self.audio_queue.empty():
                    audio_chunks.append(self.audio_queue.get_nowait())
                return np.concatenate(audio_chunks) if audio_chunks else np.array([], dtype=self.format)
            except queue.Empty:
                return np.array([], dtype=self.format)

class SpeechDataCollector:
    def __init__(self, output_dir='data/speech'):
        self.output_dir = output_dir
        self.commands_file = os.path.join(output_dir, 'commands.csv')
        self.recorder = AudioRecorder()
        
        # Ensure base directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.setup_gui()
        
    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Speech Data Collector")
        self.root.resizable(False, False)
        
        # Style configuration
        style = ttk.Style()
        style.configure('Record.TButton', padding=10)
        style.configure('Status.TLabel', font=('Helvetica', 10))
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title label
        title_label = ttk.Label(main_frame, text="Speech Data Collector", font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 15))
        
        # Command input frame
        input_frame = ttk.LabelFrame(main_frame, text="Voice Command", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.command_entry = ttk.Entry(input_frame, width=40, font=('Helvetica', 11))
        self.command_entry.grid(row=0, column=0, padx=5, pady=5)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=15)
        
        self.record_button = ttk.Button(
            control_frame, 
            text="Start Recording",
            command=self.toggle_recording,
            style='Record.TButton',
            width=20
        )
        self.record_button.grid(row=0, column=0, padx=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_label = ttk.Label(
            status_frame, 
            text="Ready to record",
            style='Status.TLabel',
            wraplength=300
        )
        self.status_label.grid(row=0, column=0, pady=5)
        
    def toggle_recording(self):
        """Toggle recording state with input validation."""
        if not hasattr(self, 'recording'):
            self.recording = False
            
        if not self.recording:
            command = self.command_entry.get().strip()
            if not command:
                self.show_status("Please enter a command", error=True)
                return
            self.start_recording()
        else:
            self.stop_recording()
            
    def show_status(self, message, error=False):
        """Update status label with optional error styling."""
        self.status_label.config(
            text=message,
            foreground='red' if error else 'black'
        )
        
    def save_audio_file(self, audio_data, command):
        """Save audio data in command-specific folders."""
        # Create command-specific directory
        command_dir = os.path.join(self.output_dir, command)
        os.makedirs(command_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.wav"
        filepath = os.path.join(command_dir, filename)
        
        try:
            with wave.open(filepath, 'wb') as wf:
                wf.setnchannels(self.recorder.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.recorder.sample_rate)
                wf.writeframes(audio_data.tobytes())
            return os.path.join(command, filename)  # Return relative path
        except Exception as e:
            self.show_status(f"Error saving audio: {str(e)}", error=True)
            return None
            
    def update_commands_csv(self, filepath, command):
        """Update CSV with relative filepath."""
        try:
            if os.path.exists(self.commands_file):
                existing_df = pd.read_csv(self.commands_file)
                if command in existing_df['command'].values:
                    return
                
            new_data = {
                'command': [command]
            }
            df = pd.DataFrame(new_data)
            
            if os.path.exists(self.commands_file):
                df = pd.concat([existing_df, df], ignore_index=True)
                
            df.to_csv(self.commands_file, index=False)
        except Exception as e:
            self.show_status(f"Error updating commands: {str(e)}", error=True)
        
    def start_recording(self):
        self.recording = True
        self.record_button.config(text="Stop Recording")
        self.status_label.config(text="Recording...")
        self.recorder.start_recording()
        
    def stop_recording(self):
        self.recording = False
        self.record_button.config(text="Start Recording")
        
        audio_data = self.recorder.stop_recording()
        
        if len(audio_data) > 0:
            command = self.command_entry.get().strip()
            filepath = self.save_audio_file(audio_data, command)
            
            if filepath:
                self.update_commands_csv(filepath, command)
                self.status_label.config(text=f"Saved: {filepath}")
            else:
                self.status_label.config(text="Error saving audio")
        else:
            self.status_label.config(text="No audio recorded")
            
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    collector = SpeechDataCollector()
    collector.run() 