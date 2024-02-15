import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(16, 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

class MelodyDecoder(nn.Module):
    def __init__(self):
        super(MelodyDecoder, self).__init__()
        self.fc1 = nn.Linear(32 * 64 * 43, 18271)

    def forward(self, x):
        x = x.reshape(-1, 32 * 64 * 43)
        x = self.fc1(x)
        return x

class PitchDecoder(nn.Module):
    def __init__(self):
        super(PitchDecoder, self).__init__()
        self.fc1 = nn.Linear(32 * 64 * 43, 370)

    def forward(self, x):
        x = x.reshape(-1, 32 * 64 * 43)
        x = self.fc1(x)
        return x

class MFCCDecoder(nn.Module):
    def __init__(self):
        super(MFCCDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv2 = nn.ConvTranspose2d(16, 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

class MultiDecoderModel(nn.Module):
    def __init__(self):
        super(MultiDecoderModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.melody_decoder = MelodyDecoder()
        self.pitch_decoder = PitchDecoder()
        self.mfcc_decoder = MFCCDecoder()

    def forward(self, x):
        encoded_output = self.encoder(x)
        decoded_output = self.decoder(encoded_output)
        melody_output = self.melody_decoder(encoded_output)
        pitch_output = self.pitch_decoder(encoded_output)
        mfcc_output = self.mfcc_decoder(encoded_output)
        return decoded_output, melody_output, pitch_output, mfcc_output

class MultiTaskLoss(nn.Module):
    def __init__(self, weight_reconstruction=1.0, weight_melody=1.0, weight_pitch=1.0, weight_mfcc=1.0):
        super(MultiTaskLoss, self).__init__()
        self.weight_reconstruction = weight_reconstruction
        self.weight_melody = weight_melody
        self.weight_pitch = weight_pitch
        self.weight_mfcc = weight_mfcc

    def forward(self, decoded_output, target, melody_output, target_melody, pitch_output, target_pitch, mfcc_output, target_mfcc):
        reconstruction_loss = F.mse_loss(decoded_output, target)
        melody_loss = F.cross_entropy(melody_output, target_melody)
        pitch_loss = F.cross_entropy(pitch_output, target_pitch)
        mfcc_loss = F.mse_loss(mfcc_output, target_mfcc)

        # Combine losses with weights
        total_loss = (self.weight_reconstruction * reconstruction_loss) + \
                     (self.weight_melody * melody_loss) + \
                     (self.weight_pitch * pitch_loss) + \
                     (self.weight_mfcc * mfcc_loss)

        return total_loss

# Instantiate the model
model = MultiDecoderModel()

# Define input
input_melspectrogram = torch.randn(1, 2, 256, 173)

# Forward pass
decoded_output, melody_output, pitch_output, mfcc_output = model(input_melspectrogram)


print(decoded_output.shape,melody_output.shape,pitch_output.shape,mfcc_output.shape)

# Define targets for each task (these are just placeholders, replace them with actual targets)
target_reconstruction = torch.randn_like(decoded_output)
target_melody = torch.randint(0, 10, (1,), dtype=torch.long)
target_pitch = torch.randint(0, 10, (1,), dtype=torch.long)
target_mfcc = torch.randn_like(mfcc_output)

# Define the loss function
criterion = MultiTaskLoss()

# Calculate the loss
loss = criterion(decoded_output, target_reconstruction, melody_output, target_melody, pitch_output, target_pitch, mfcc_output, target_mfcc)

# Backpropagation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer.zero_grad()
loss.backward()

# Optimization step
optimizer.step()

