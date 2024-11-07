import torch
import torch.nn as nn
import librosa
import pandas as pd
import numpy as np

class MFCC(nn.Module):
    def __init__(self, sample_rate, n_fft=400, hop_length=200, n_mels=40, n_mfcc=40):
        super(MFCC, self).__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
    
    def forward(self, x):
        # Convert torch tensor to numpy array
        x = x.squeeze(0).numpy()
        mfcc = librosa.feature.mfcc(
            y=x,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            n_mfcc=self.n_mfcc
        )
        return torch.Tensor(mfcc).unsqueeze(0)


def get_featurizer(sample_rate):
    return MFCC(sample_rate=sample_rate)


class RandomCut(nn.Module):
    """Augmentation technique that randomly cuts start or end of audio"""

    def __init__(self, max_cut=10):
        super(RandomCut, self).__init__()
        self.max_cut = max_cut

    def forward(self, x):
        """Randomly cuts from start or end of batch"""
        side = torch.randint(0, 2, (1,))
        cut = torch.randint(1, self.max_cut, (1,))
        if side == 0:
            return x[:-cut,:,:]
        else:
            return x[cut:,:,:]


class SpecAugment(nn.Module):
    """Augmentation technique to add masking on the time or frequency domain"""

    def __init__(self, rate, policy=3, freq_mask=2, time_mask=4):
        super(SpecAugment, self).__init__()
        self.rate = rate
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.policy = policy

    def _freq_mask(self, x):
        mask_value = x.mean()
        for i in range(self.freq_mask):
            freq_idx = torch.randint(0, x.shape[1] - self.freq_mask, (1,))
            x[:, freq_idx:freq_idx + self.freq_mask] = mask_value
        return x

    def _time_mask(self, x):
        mask_value = x.mean()
        for i in range(self.time_mask):
            time_idx = torch.randint(0, x.shape[2] - self.time_mask, (1,))
            x[:, :, time_idx:time_idx + self.time_mask] = mask_value
        return x

    def forward(self, x):
        probability = torch.rand(1, 1).item()
        
        if self.policy == 1 and probability < self.rate:
            x = self._freq_mask(x)
            x = self._time_mask(x)
        elif self.policy == 2 and probability < self.rate:
            x = self._freq_mask(x)
            x = self._time_mask(x)
            x = self._freq_mask(x)
            x = self._time_mask(x)
        elif self.policy == 3:
            if probability > 0.5:
                x = self._freq_mask(x)
                x = self._time_mask(x)
            else:
                x = self._freq_mask(x)
                x = self._time_mask(x)
                x = self._freq_mask(x)
                x = self._time_mask(x)
        return x


class WakeWordData(torch.utils.data.Dataset):
    """Load and process wakeword data"""

    def __init__(self, data_json, sample_rate=8000, valid=False):
        self.sr = sample_rate
        self.data = pd.read_json(data_json, lines=True)
        if valid:
            self.audio_transform = get_featurizer(sample_rate)
        else:
            self.audio_transform = nn.Sequential(
                get_featurizer(sample_rate),
                SpecAugment(rate=0.5)
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        try:    
            file_path = self.data.key.iloc[idx]
            # Use librosa instead of torchaudio
            waveform, sr = librosa.load(file_path, sr=self.sr)
            # Convert to torch tensor
            waveform = torch.FloatTensor(waveform).unsqueeze(0)
            mfcc = self.audio_transform(waveform)
            label = self.data.label.iloc[idx]

        except Exception as e:
            print(str(e), file_path)
            return self.__getitem__(torch.randint(0, len(self), (1,)))

        return mfcc, label


rand_cut = RandomCut(max_cut=10)

def collate_fn(data):
    """Batch and pad wakeword data"""
    mfccs = []
    labels = []
    for d in data:
        mfcc, label = d
        mfccs.append(mfcc.squeeze(0).transpose(0, 1))
        labels.append(label)

    # pad mfccs to ensure all tensors are same size in the time dim
    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True)  # batch, seq_len, feature
    mfccs = mfccs.transpose(0, 1) # seq_len, batch, feature
    mfccs = rand_cut(mfccs)
    #print(mfccs.shape)
    labels = torch.Tensor(labels)
    return mfccs, labels