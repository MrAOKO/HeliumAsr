import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import random

class HeliumASRDataset(Dataset):
    def __init__(self, data_dir, vocab, feature_extractor):
        self.data = []
        self.vocab = vocab
        self.feature_extractor = feature_extractor
        for fname in os.listdir(data_dir):
            if fname.endswith('.WAV') or fname.endswith('.wav'):
                wav_path = os.path.join(data_dir, fname)
                txt_path = os.path.splitext(wav_path)[0] + '.txt'
                if os.path.exists(txt_path):
                    self.data.append((wav_path, txt_path))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        wav_path, txt_path = self.data[idx]
        waveform, sr = torchaudio.load(wav_path)
        waveform, sr = pretreatment(waveform, sr)
        features = self.feature_extractor(waveform, sr)  
        features = spec_augment(features) 
        with open(txt_path, encoding='utf-8') as f:
            text = f.read().strip()
        targets = torch.tensor([self.vocab[c] for c in text if c in self.vocab], dtype=torch.long)
        input_length = features.shape[-1]
        target_length = len(targets)
        return features, targets, input_length, target_length

def build_vocab(data_dir):
    chars = set()
    for fname in os.listdir(data_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(data_dir, fname), encoding='utf-8') as f:
                chars.update(f.read().strip())
    vocab = {c: i+1 for i, c in enumerate(sorted(chars))} 
    vocab['<blank>'] = 0
    return vocab

def extract_fbank(waveform, sr, n_mels=80):
    return torchaudio.compliance.kaldi.fbank(
        waveform, num_mel_bins=n_mels, sample_frequency=sr
    ).transpose(0, 1).unsqueeze(0)  # (1, freq_bins, time_steps)


def collate_fn(batch):
    features, targets, input_lengths, target_lengths = zip(*batch)
    features = [f.squeeze(0).transpose(0, 1) for f in features]
    features = pad_sequence(features, batch_first=True) 
    features = features.transpose(1, 2).unsqueeze(1)
    targets = pad_sequence(targets, batch_first=True, padding_value=0)
    input_lengths = torch.tensor([f.shape[-1] for f in features], dtype=torch.long)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return features, targets, input_lengths, target_lengths

def pretreatment(waveform, sr, target_sr=16000, preemph=0.97):
    # 1. 重采样
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    # 2. 预加重
    waveform = torch.cat([waveform[:, :1], waveform[:, 1:] - preemph * waveform[:, :-1]], dim=1)
    # 3. 音量归一化
    waveform = waveform / waveform.abs().max()
    return waveform, target_sr
def spec_augment(mel_spectrogram, freq_mask_param=8, time_mask_param=10, num_masks=2):
    # mel_spectrogram: (1, freq, time)
    augmented = mel_spectrogram.clone()
    _, freq_len, time_len = augmented.shape
    for _ in range(num_masks):
        # 频率遮挡
        f = random.randint(0, freq_mask_param)
        f0 = random.randint(0, max(0, freq_len - f))
        augmented[0, f0:f0+f, :] = 0
        # 时间遮挡
        t = random.randint(0, time_mask_param)
        t0 = random.randint(0, max(0, time_len - t))
        augmented[0, :, t0:t0+t] = 0
    return augmented