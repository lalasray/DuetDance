import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    ax.grid(True)

def plot_pitch(waveform, sr, pitch, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Pitch Feature")
    ax.grid(True)

    end_time = waveform.shape[1] / sr
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    ax.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    ax2 = ax.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    ax2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax2.set_ylabel("Pitch (Hz)")
    ax2.legend(loc=0)

def plot_melody_mir(melody, ax=None, sr=44100):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    times = np.arange(len(melody)) * (1/sr)
    ax.plot(times, melody)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Melody Extraction MIR')
    ax.grid(True)

# Load saved variables
DIRECTORY = "C:/Users/lalas/Desktop/wham/segaud/45/"
LOAD_FILE = os.path.join(DIRECTORY, "subvideo_3_variables_3.npz")
loaded_data = np.load(LOAD_FILE)

melspec = torch.from_numpy(loaded_data["melspectrogram"])
mfcc = torch.from_numpy(loaded_data["mfcc"])
pitch = torch.from_numpy(loaded_data["pitch"])
melody_mir = loaded_data["melody"]
#print(torch.min(melspec), torch.max(melspec),torch.min(mfcc),torch.max(mfcc),torch.min(pitch),torch.max(pitch))
melspec_normalized = (melspec - torch.min(melspec)) / (torch.max(melspec) - torch.min(melspec))
mfcc_normalized = (mfcc - torch.min(mfcc)) / (torch.max(mfcc) - torch.min(mfcc))
pitch_normalized = (pitch - torch.min(pitch)) / (torch.max(pitch) - torch.min(pitch))
#print(torch.min(melspec_normalized), torch.max(melspec_normalized),torch.min(mfcc_normalized),torch.max(mfcc_normalized),torch.min(pitch_normalized),torch.max(pitch_normalized))

# Plot Mel spectrogram and MFCC
#fig, axs = plt.subplots(6, 1, figsize=(10, 12))
#plot_spectrogram(melspec[0], title="Mel Spectrogram", ax=axs[0])
#plot_spectrogram(mfcc[0], title="MFCC", ax=axs[2])
#plot_pitch(pitch, 44100, pitch, ax=axs[4]) 
#plot_spectrogram(melspec_normalized[0], title="Normalized Mel Spectrogram", ax=axs[1])
#plot_spectrogram(mfcc_normalized[0], title="Normalized MFCC", ax=axs[3])
#plot_pitch(pitch_normalized, 44100, pitch_normalized, ax=axs[5]) 
#plt.tight_layout()
#plt.show()
