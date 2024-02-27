import os
import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import numpy as np


# Define the parent directory path
PARENT_DIRECTORY = "C:/Users/lalas/Desktop/wham/segaud/"

# Iterate over numbers from 2 to 74
for folder_number in range(2, 75):
    folder_name = str(folder_number)
    DIRECTORY = os.path.join(PARENT_DIRECTORY, folder_name)

    # Iterate over all files in the directory
    for filename in os.listdir(DIRECTORY):
        if filename.endswith(".mp3"):
            # Full path to the MP3 file
            SAMPLE_SPEECH = os.path.join(DIRECTORY, filename)
            torch.random.manual_seed(0)

            # Get information about the audio
            info = torchaudio.info(SAMPLE_SPEECH)
            SAMPLE_RATE = info.sample_rate

            # Melody extraction function
            def extract_melody(audio_file, sr):
                y, sr = librosa.load(audio_file, sr=sr)
                f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
                return f0

            # Load audio segment function
            def load_audio_segment(audio_file, start_time, duration, sr):
                start_frame = int(start_time * sr)
                num_frames = int(duration * sr)
                audio, _ = torchaudio.load(audio_file, frame_offset=start_frame, num_frames=num_frames)
                return audio

            # Define parameters for transforms
            n_fft = 2048
            win_length = None
            hop_length = 512
            n_mels = 256
            n_mfcc = 256

            # Define MFCC and Mel Spectrogram transforms
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

            melody_mir = extract_melody(SAMPLE_SPEECH, SAMPLE_RATE)

            # Check if the directory exists, if not, create it
            if not os.path.exists(DIRECTORY):
                os.makedirs(DIRECTORY)

            # Sliding window parameters
            window_size = 2  # in seconds
            step_size = 1    # in seconds

            # Iterate over the audio file with sliding windows
            start_times = np.arange(0, info.num_frames / SAMPLE_RATE - window_size + 1, step_size)
            for i, start_time in enumerate(start_times):
                #print(f"Processing window {i+1}/{len(start_times)} of file {filename}")
                audio_segment = load_audio_segment(SAMPLE_SPEECH, start_time, window_size, SAMPLE_RATE)
                mfcc = mfcc_transform(audio_segment)
                melspec = mel_spectrogram(audio_segment)
                pitch = F.detect_pitch_frequency(audio_segment, SAMPLE_RATE)
                #print(audio_segment.shape, mfcc.shape, melspec.shape, pitch.shape, melody_mir.shape)
                # Save the variables for each segment
                SAVE_FILE = os.path.join(DIRECTORY, f"{os.path.splitext(filename)[0]}_variables_{i}.npz")
                np.savez(SAVE_FILE,
                        speech_waveform=audio_segment.numpy(),
                        melspectrogram=melspec.numpy(),
                        mfcc=mfcc.numpy(),
                        pitch=pitch.numpy(),
                        melody=np.array(melody_mir))

            print(f"Variables for file {filename} saved successfully.")
