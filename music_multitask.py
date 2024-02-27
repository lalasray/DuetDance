import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, activation='relu'):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.5)
        if activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)
        return x

class Decoder(nn.Module):
    def __init__(self, activation='relu'):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.ConvTranspose2d(16, 2, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout(0.2)
        if activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.conv2(x, output_size=(x.size(0), 2, 256, 173)))
        x = self.dropout(x)
        return x
    
class MFCCDecoder(nn.Module):
    def __init__(self, activation='relu'):
        super(MFCCDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.ConvTranspose2d(16, 2, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(2)
        self.dropout = nn.Dropout(0.2)
        if activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.conv2(x, output_size=(x.size(0), 2, 256, 173)))
        x = self.dropout(x)
        return x


class PitchDecoder(nn.Module):
    def __init__(self, activation='relu'):
        super(PitchDecoder, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc = nn.Linear(256 * 64 * 43, 2 * 185)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.2)
        
        if activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), 2, 185)
        x = self.dropout(x)
        return x
    
class MultiDecoderModel(nn.Module):
    def __init__(self, activation='relu'):
        super(MultiDecoderModel, self).__init__()
        self.encoder = Encoder(activation=activation)
        self.decoder = Decoder(activation=activation)
        self.pitch_decoder = PitchDecoder(activation=activation)
        self.mfcc_decoder = MFCCDecoder(activation=activation)

    def forward(self, x):
        batch_size = x.size(0)
        encoded_output = self.encoder(x)
        decoded_output = self.decoder(encoded_output)
        pitch_output = self.pitch_decoder(encoded_output)
        mfcc_output = self.mfcc_decoder(encoded_output)
        return decoded_output, pitch_output, mfcc_output


#class MultiTaskLoss(nn.Module):
    #def __init__(self, weight_reconstruction=0.6, weight_pitch=0.3, weight_mfcc=0.3):
        #super(MultiTaskLoss, self).__init__()
        #self.weight_reconstruction = weight_reconstruction
        #self.weight_pitch = weight_pitch
        #self.weight_mfcc = weight_mfcc

    #def forward(self, decoded_output, target, pitch_output, target_pitch, mfcc_output, target_mfcc):
        #reconstruction_loss = F.mse_loss(decoded_output, target)
        #pitch_loss = F.mse_loss(pitch_output, target_pitch)
        #mfcc_loss = F.mse_loss(mfcc_output, target_mfcc)
        #print("Mel:",reconstruction_loss,"Pitch:",pitch_loss, "MFCC:",mfcc_loss)
        #total_loss = (self.weight_reconstruction * reconstruction_loss) + \
                     #(self.weight_pitch * pitch_loss) + \
                     #(self.weight_mfcc * mfcc_loss)

        #return total_loss



def pitch_loss(predictions, targets):
    
    predictions = predictions.view(-1, 2, 185)
    targets = targets.view(-1, 2, 185)
    pitch_diff = torch.abs(predictions[:, :, 1:] - predictions[:, :, :-1])
    target_pitch_diff = torch.abs(targets[:, :, 1:] - targets[:, :, :-1])
    pitch_loss = torch.mean(torch.abs(pitch_diff - target_pitch_diff))
    
    return pitch_loss

class MultiTaskLoss(nn.Module):
    def __init__(self, weight_reconstruction=1, weight_pitch=250, weight_mfcc=50):
        super(MultiTaskLoss, self).__init__()
        self.weight_reconstruction = weight_reconstruction
        self.weight_pitch = weight_pitch
        self.weight_mfcc = weight_mfcc

    def forward(self, decoded_output, target, pitch_output, target_pitch, mfcc_output, target_mfcc, epoch=None):
        
        reconstruction_loss = F.mse_loss(decoded_output, target) + 1e-16
        elem_reconstruction_loss = reconstruction_loss / reconstruction_loss.numel() + 1e-16
        pitch_loss_val = F.mse_loss(pitch_output, target_pitch) + 100
        elem_pitch_loss_val = pitch_loss_val / pitch_loss_val.numel() + 100
        mfcc_loss = F.mse_loss(mfcc_output, target_mfcc) + 1000
        elem_mfcc_loss = mfcc_loss / mfcc_loss.numel() + 1000
        total_loss = self.weight_reconstruction * elem_reconstruction_loss + self.weight_pitch * elem_pitch_loss_val + self.weight_mfcc * elem_mfcc_loss 

        return total_loss,reconstruction_loss,pitch_loss_val,mfcc_loss
# Example usage
#activation = 'relu'
#activation = 'silu'
#model = MultiDecoderModel(activation=activation)

#batch_size = 16
#input_melspectrogram = torch.randn(batch_size, 2, 256, 173)

#decoded_output, pitch_output, mfcc_output = model(input_melspectrogram)

#print(decoded_output.shape, pitch_output.shape, mfcc_output.shape)


