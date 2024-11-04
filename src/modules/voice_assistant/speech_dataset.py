import os
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import pandas as pd

class SpeechDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        
        # Create a mapping from command to label
        self.command_to_label = {command: idx for idx, command in enumerate(self.data['command'].unique())}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        command = self.data.iloc[idx]['command']
        command_dir = os.path.join(self.audio_dir, command)
        
        # List all .wav files in the command directory
        wav_files = [f for f in os.listdir(command_dir) if f.endswith('.wav')]
        
        # Select a file (e.g., the first one)
        if not wav_files:
            raise FileNotFoundError(f"No .wav files found in directory: {command_dir}")
        
        audio_path = os.path.join(command_dir, wav_files[0])
        print(f"Loading audio file: {audio_path}")
        label = self.command_to_label[command]
        
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Apply transformations if any
        if self.transform:
            waveform = self.transform(waveform)
            
        # Convert to mel spectrogram
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )(waveform)
        
        # Convert to log scale
        mel_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        
        return mel_spectrogram, torch.tensor(label)