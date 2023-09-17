import os
import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random


class UrbanSound8KDataset(Dataset):
    def __init__(self, data_path, train=True, transform=None,test_size=0.2):
        metadata_path = os.path.join(data_path, "metadata", "UrbanSound8K.csv")
        self.data = pd.read_csv(metadata_path)
        self.file_paths = []
        self.labels = []
        for i in range(len(self.data)):
            file_path = os.path.join(data_path, "audio", "fold" + str(self.data.iloc[i]["fold"]) + "/",
                                     self.data.iloc[i]["slice_file_name"])
            label = self.data.iloc[i]["classID"]
            self.file_paths.append(file_path)
            self.labels.append(label)

        self.transform = transform
        self.train = train
        train_indices, test_indices = train_test_split(
            list(range(len(self.data))),
            test_size=test_size,
            random_state=42
        )

        self.file_paths = [self.file_paths[i] for i in train_indices] if self.train else [self.file_paths[i] for i in
                                                                                          test_indices]
        self.labels = [self.labels[i] for i in train_indices] if self.train else [self.labels[i] for i in test_indices]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=22050)
        if audio.shape[0] > 88200:
            max_audio_start = audio.shape[0] - 88200
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + 88200]
        else:
            audio = librosa.util.pad_center(audio, size=88200)

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=2048,
                                                  hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db + 80) / (0.000003814697265625 + 80)
        #mel_spec_db = mel_spec_db / 80
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
        mel_spec_db = torch.from_numpy(mel_spec_db).float()

        if self.transform:
            mel_spec_db = self.transform(mel_spec_db)

        return mel_spec_db, label


class UrbanSound8KDataset_classwise_onlyone(torch.utils.data.Dataset):
    def __init__(self, data_path, class_num):
        metadata_path = os.path.join(data_path, "metadata", "UrbanSound8K.csv")
        self.data = pd.read_csv(metadata_path)
        self.data = self.data.groupby("classID").apply(lambda x: x[x["classID"] == class_num]).reset_index(
            drop=True)
        self.file_paths = []
        self.labels = []
        for i in range(len(self.data)):
            file_path = os.path.join(data_path, "audio", "fold" + str(self.data.iloc[i]["fold"]) + "/",
                                     self.data.iloc[i]["slice_file_name"])
            label = self.data.iloc[i]["classID"]
            self.file_paths.append(file_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        audio, sr = librosa.load(file_path, sr=22050)
        if audio.shape[0] > 88200:
            max_audio_start = audio.shape[0] - 88200
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start: audio_start + 88200]
        else:
            audio = librosa.util.pad_center(audio, size=88200)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=2048,
                                                  hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db + 80) / (0.000003814697265625 + 80)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
        mel_spec_db = torch.from_numpy(mel_spec_db).float()
        return mel_spec_db, label
