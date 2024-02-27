import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from music_loader import CustomDataset
from music_multitask import MultiDecoderModel, MultiTaskLoss
import numpy as np
import matplotlib.pyplot as plt

batch_size = 64
initial_learning_rate = 0.001
num_epochs = 50
lr_decay_epoch = 10
lr_decay_factor = 0.5  # Factor by which to decay learning rate
patience = 12  # Number of epochs with no improvement after which to stop training
reconstruction_weight = 1
pitch_weight = 1
mfcc_weight = 1

# Assuming your dataset is CustomDataset and you have a total of 1000 samples
custom_dataset = CustomDataset(root_directory="C:/Users/lalas/Desktop/wham/segaudtest/")

# Determine the sizes for the training and validation sets
train_size = int(0.8 * len(custom_dataset))  # 80% of the data for training
val_size = len(custom_dataset) - train_size  # Remaining 20% for validation

# Split the dataset into training and validation sets
train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiDecoderModel().to(device)
criterion = MultiTaskLoss().to(device)
#optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate)

prev_val_loss = np.inf
no_improvement_count = 0
best_val_loss = float('inf')
best_model_weights = None

# Create data loaders for training and validation sets
train_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
#train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Lists to store the losses
train_losses = []
val_losses = []
reconstruction_losses = []
pitch_losses = []
mfcc_losses = []

for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (melspec_batch, mfcc_batch, pitch_batch) in enumerate(train_loader):
        melspec_batch = melspec_batch.to(device)
        mfcc_batch = mfcc_batch.to(device)
        pitch_batch = pitch_batch.to(device)

        optimizer.zero_grad()

        decoded_output, pitch_output, mfcc_output = model(melspec_batch)

        total_loss, reconstruction_loss, pitch_loss_val, mfcc_loss = criterion(decoded_output, melspec_batch, pitch_output, pitch_batch, mfcc_output, mfcc_batch)

        loss = reconstruction_weight*reconstruction_loss + pitch_weight*pitch_loss_val + mfcc_weight*mfcc_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if batch_idx % 100 == 99: 
            print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

    with torch.no_grad():
        val_loss = 0.0
        for val_batch_idx, (val_melspec_batch, val_mfcc_batch, val_pitch_batch) in enumerate(val_loader):
            val_melspec_batch = val_melspec_batch.to(device)
            val_mfcc_batch = val_mfcc_batch.to(device)
            val_pitch_batch = val_pitch_batch.to(device)

            val_decoded_output, val_pitch_output, val_mfcc_output = model(val_melspec_batch)

            val_total_loss, val_reconstruction_loss, val_pitch_loss, val_mfcc_loss = criterion(val_decoded_output, val_melspec_batch, val_pitch_output, val_pitch_batch, val_mfcc_output, val_mfcc_batch)

            # Combine the losses with appropriate weights
            val_loss = reconstruction_weight*val_reconstruction_loss + pitch_weight*val_pitch_loss + mfcc_weight*val_mfcc_loss

            val_loss += val_loss.item()  # Add the scalar loss to the running total loss

        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, MelSpec Loss: {val_reconstruction_loss:.4f}, Pitch Loss: {val_pitch_loss:.4f}, MFCC Loss: {val_mfcc_loss:.4f}')

        if (epoch + 1) % lr_decay_epoch == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay_factor

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict()

        if val_loss >= prev_val_loss:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping. No improvement in validation loss.")
                break
        else:
            no_improvement_count = 0
            prev_val_loss = val_loss

    # Store losses
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss)
    reconstruction_losses.append(val_reconstruction_loss)
    pitch_losses.append(val_pitch_loss)
    mfcc_losses.append(val_mfcc_loss)

print('Finished Training')

# Plotting
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, val_losses, label='Validation Loss')
plt.plot(epochs, reconstruction_losses, label='Melspec Loss')
plt.plot(epochs, pitch_losses, label='Pitch Loss')
plt.plot(epochs, mfcc_losses, label='MFCC Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Reconstruction, Pitch, and MFCC Losses')
plt.legend()
plt.show()

# Save the weights of the best model
if best_model_weights is not None:
    torch.save(best_model_weights, 'best_multi_decoder_model_weights.pth')
