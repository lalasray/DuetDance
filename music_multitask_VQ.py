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

class PitchDecoder(nn.Module):
    def __init__(self, activation='relu'):
        super(PitchDecoder, self).__init__()
        self.fc1 = nn.Linear(32 * 64 * 43, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2 * 185)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.2)
        if activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.activation(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        x = x.view(x.size(0), 2, 185)
        x = self.dropout(x)
        return x

class QuantizeEMAReset(nn.Module):
    def __init__(self, nb_code, code_dim):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.reset_codebook()

    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else:
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True

    @torch.no_grad()
    def compute_perplexity(self, code_idx):
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)
        code_count = code_onehot.sum(dim=-1)
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity

    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)
        code_sum = torch.matmul(code_onehot, x)
        code_count = code_onehot.sum(dim=-1)

        out = self._tile(x)
        code_rand = out[:self.nb_code]

        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count

        usage = (self.code_count.view(self.nb_code, 1) >= 1.0).float()
        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = usage * code_update + (1 - usage) * code_rand
        prob = code_count / torch.sum(code_count)
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))

        return perplexity

    def preprocess(self, x):
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])
        return x

    def quantize(self, x):
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0, keepdim=True)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    def forward(self, x):
        N, width, T = x.shape

        x = self.preprocess(x)

        if self.training and not self.init:
            self.init_codebook(x)

        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else:
            perplexity = self.compute_perplexity(code_idx)

        commit_loss = F.mse_loss(x, x_d.detach())

        x_d = x + (x_d - x).detach()

        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()

        return x_d, commit_loss, perplexity

class MultiTaskModel(nn.Module):
    def __init__(self, activation='relu', nb_code=512, code_dim=64):
        super(MultiTaskModel, self).__init__()
        self.encoder = Encoder(activation=activation)
        self.quantizer = QuantizeEMAReset(nb_code=nb_code, code_dim=code_dim)
        self.decoder = Decoder(activation=activation)
        self.pitch_decoder = PitchDecoder(activation=activation)
        self.mfcc_decoder = Decoder(activation=activation)

    def forward(self, x):
        batch_size = x.size(0)
        encoded_output = self.encoder(x)
        quantized_output, commit_loss, perplexity = self.quantizer(encoded_output)
        decoded_output = self.decoder(quantized_output)
        pitch_output = self.pitch_decoder(encoded_output)
        mfcc_output = self.mfcc_decoder(encoded_output)
        return decoded_output, pitch_output, mfcc_output, commit_loss, perplexity


class MultiTaskLoss(nn.Module):
    def __init__(self, weight_reconstruction=1.0, weight_pitch=0.3, weight_mfcc=0.3, weight_commit=0.2, weight_perplexity=0.2):
        super(MultiTaskLoss, self).__init__()
        self.weight_reconstruction = weight_reconstruction
        self.weight_pitch = weight_pitch
        self.weight_mfcc = weight_mfcc
        self.weight_commit = weight_commit
        self.weight_perplexity = weight_perplexity

    def forward(self, decoded_output, target, pitch_output, target_pitch, mfcc_output, target_mfcc, commit_loss, perplexity_loss):
        reconstruction_loss = F.mse_loss(decoded_output, target)
        pitch_loss = 1 - F.cosine_similarity(pitch_output, target_pitch).mean()
        mfcc_loss = F.mse_loss(mfcc_output, target_mfcc)

        total_loss = (self.weight_reconstruction * reconstruction_loss) + \
                     (self.weight_pitch * pitch_loss) + \
                     (self.weight_mfcc * mfcc_loss) + \
                     (self.weight_commit * commit_loss) + \
                     (self.weight_perplexity * perplexity_loss)

        return total_loss

# Example usage
activation = 'relu'
#activation = 'silu'
model = MultiTaskModel(activation=activation)

batch_size = 16
input_melspectrogram = torch.randn(batch_size, 2, 256, 173)

decoded_output, pitch_output, mfcc_output = model(input_melspectrogram)

print(decoded_output.shape, pitch_output.shape, mfcc_output.shape)


