import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import librosa
import librosa.display

class SpeechDataset(Dataset):
    def __init__(self, csv_file, audio_dir, transform=None, target_length=128):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.target_length = target_length  # Fixed length for all spectrograms
        
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
        waveform, sample_rate = librosa.load(audio_path, sr=22050)
        
        # Convert to torch tensor and reshape for consistency
        waveform = torch.FloatTensor(waveform).unsqueeze(0)  # Add channel dimension
        
        # Apply transformations if any
        if self.transform:
            waveform = self.transform(waveform)
            
        # Replace torchaudio mel spectrogram with librosa
        mel_spectrogram = librosa.feature.melspectrogram(
            y=waveform.squeeze().numpy(),
            sr=sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=128
        )
        
        # Convert to log scale
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Convert to torch tensor
        mel_spectrogram = torch.FloatTensor(mel_spectrogram).unsqueeze(0)
        
        # After creating mel_spectrogram, pad or trim it to target length
        current_length = mel_spectrogram.shape[2]
        if current_length > self.target_length:
            # Trim
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros
            padding = torch.zeros(1, 128, self.target_length - current_length)
            mel_spectrogram = torch.cat([mel_spectrogram, padding], dim=2)

        return mel_spectrogram, torch.tensor(label)