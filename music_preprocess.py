import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

torch.random.manual_seed(0)

SAMPLE_SPEECH = "/home/lala/Downloads/funky_dealer.mp3"

# Get the sample rate of the audio file using torchaudio.info
info = torchaudio.info(SAMPLE_SPEECH)
SAMPLE_RATE = info.sample_rate

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1, figsize=(10, 6))
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    ax.grid(True)

# Melody extraction
def extract_melody(audio_file, sr):
    y, sr = librosa.load(audio_file, sr=sr)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    return f0


# Load audio
SPEECH_WAVEFORM, _ = torchaudio.load(SAMPLE_SPEECH, num_frames=int(SAMPLE_RATE * 2))  # Load only first 2 seconds

n_fft = 2048
win_length = None
hop_length = 512
n_lfcc = 256
n_mels = 256
n_mfcc = 256

mfcc_transform = T.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=n_mfcc,
    melkwargs={
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "mel_scale": "htk",
    },
)

mfcc = mfcc_transform(SPEECH_WAVEFORM)

mel_spectrogram = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    n_mels=n_mels,
    mel_scale="htk",
)

melspec = mel_spectrogram(SPEECH_WAVEFORM)


spectrogram = T.Spectrogram(
    n_fft=n_fft,
      win_length=win_length,
        hop_length=hop_length
        )
#spec = spectrogram(SPEECH_WAVEFORM)

lfcc_transform = T.LFCC(
    sample_rate=SAMPLE_RATE,
    n_lfcc=n_lfcc,
    speckwargs={
        "n_fft": n_fft,
        "win_length": win_length,
        "hop_length": hop_length,
    },
)

#lfcc = lfcc_transform(SPEECH_WAVEFORM)

pitch = F.detect_pitch_frequency(SPEECH_WAVEFORM, SAMPLE_RATE)

melody_mir = extract_melody(SAMPLE_SPEECH, SAMPLE_RATE)



def plot_pitch(waveform, sr, pitch, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
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

fig, axs = plt.subplots(7, 1, figsize=(10, 20))
plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform", ax=axs[0])
#plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
plot_spectrogram(melspec[0], title="MelSpectrogram", ax=axs[2])
plot_spectrogram(mfcc[0], title="MFCC", ax=axs[3])
#plot_spectrogram(lfcc[0], title="LFCC", ax=axs[4])
plot_pitch(SPEECH_WAVEFORM, SAMPLE_RATE, pitch, ax=axs[5]) 
plot_melody_mir(melody_mir, ax=axs[6], sr=SAMPLE_RATE)
fig.tight_layout()
plt.show()

print(SPEECH_WAVEFORM.shape, melspec.shape, mfcc.shape, pitch.shape, melody_mir.shape)