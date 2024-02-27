import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_directory):
        self.file_paths = []
        for subdir, _, files in os.walk(root_directory):
            for file in files:
                file_path = os.path.join(subdir, file)
                if file.endswith(".npz") and "subvideo_0" not in file:
                    self.file_paths.append(file_path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        loaded_data = np.load(file_path)
        melspec = torch.from_numpy(loaded_data["melspectrogram"]).cuda()
        mfcc = torch.from_numpy(loaded_data["mfcc"]).cuda()
        pitch = torch.from_numpy(loaded_data["pitch"]).cuda()

        #melspec_normalized = 1*( (melspec - torch.min(melspec)) / (torch.max(melspec) - torch.min(melspec)) )
        #mfcc_normalized = 1*( (mfcc - torch.min(mfcc)) / (torch.max(mfcc) - torch.min(mfcc)) )
        #pitch_normalized = 1*( (pitch - torch.min(pitch)) / (torch.max(pitch) - torch.min(pitch)) )

        #return melspec_normalized, mfcc_normalized, pitch_normalized
        return melspec, mfcc, pitch

