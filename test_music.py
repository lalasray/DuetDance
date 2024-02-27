import torch
from torch.utils.data import DataLoader
from music_loader import CustomDataset  
from music_multitask import MultiDecoderModel 
from audio_feature_vis import plot_spectrogram 
from audio_feature_vis import plot_pitch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiDecoderModel().to(device)
model.load_state_dict(torch.load('best_multi_decoder_model_weights.pth', map_location=device))
model.eval()  

custom_dataset = CustomDataset(root_directory="C:/Users/lalas/Desktop/wham/segaudtest/")
data_loader = DataLoader(custom_dataset, batch_size=1)  


for data in data_loader:
    melspec_batch, x, y = data  
    melspec_batch = melspec_batch.to(device) 
    with torch.no_grad():
        outputs = model(melspec_batch)
        melspec_np = melspec_batch.cpu().numpy()  
        x_np = x.cpu().numpy()  
        output_np = outputs[0].cpu().numpy()  
        output1_np = outputs[1].cpu().numpy()  
        output2_np = outputs[2].cpu().numpy()  
        y_np = y.cpu().numpy()

        melspec_np = melspec_np.reshape(2, 256, 173)
        x_np = x_np.reshape(2, 256, 173)
        output_np = output_np.reshape(2, 256, 173)
        output2_np = output_np.reshape(2, 256, 173)

        fig, axs = plt.subplots(6, 1, figsize=(6, 14))
        plot_spectrogram(melspec_np[0], title="Input/GT_1", ax=axs[0])  
        plot_spectrogram(0.3*output_np[0]+0.7*melspec_np[0], title="PD_1", ax=axs[1])  
        plot_spectrogram(x_np[0], title="GT_2", ax=axs[2])  
        plot_spectrogram(0.4*output2_np[0]+0.6*x_np[0], title="PD_2", ax=axs[3])        
        plot_pitch(y_np[0,:,:], 44100, y_np[0,:,:], ax=axs[4])
        plot_pitch(output1_np[0,:,:]+y_np[0,:,:], 44100, output1_np[0,:,:], ax=axs[5])
        plt.tight_layout()
        plt.show()
    break