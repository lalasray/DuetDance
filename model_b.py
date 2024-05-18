import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Norm(nn.Module):
    """Layer normalization."""
    def __init__(self, fn):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(fn.net[0].in_features, eps=1e-5)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(nn.Module):
    """Residual layer."""
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class MLP(nn.Module):
    """Feedforward layer."""
    def __init__(self, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    """Attention layer."""
    def __init__(self, dim, heads=8):
        super(Attention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = torch.softmax(dots, dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Transformer(nn.Module):
    """Transformer Encoder."""
    def __init__(self, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072):
        super(Transformer, self).__init__()
        layers = []
        for _ in range(num_hidden_layers):
            layers.extend([
                Residual(Norm(Attention(hidden_size, heads=num_attention_heads))),
                Residual(Norm(MLP(hidden_size, intermediate_size)))
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class LinearEmbedding(nn.Module):
    """Linear projection."""
    def __init__(self, dim):
        super(LinearEmbedding, self).__init__()
        self.net = nn.Linear(dim, dim)

    def forward(self, x):
        return self.net(x)

class PositionEmbedding(nn.Module):
    """Position Embedding layer."""
    def __init__(self, seq_length, dim):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(seq_length, dim))

    def forward(self, x):
        return x + self.pos_embedding

class CrossModalLayer(nn.Module):
    """Cross-modal layer."""
    def __init__(self, config):
        super(CrossModalLayer, self).__init__()
        self.transformer_layer = Transformer(
            hidden_size=config['hidden_size'],
            num_hidden_layers=config['num_hidden_layers'],
            num_attention_heads=config['num_attention_heads'],
            intermediate_size=config['intermediate_size']
        )
        self.cross_output_layer = nn.Linear(config['hidden_size'], config['output_dim'])

    def forward(self, modal_a_sequences, modal_b_sequences):
        if modal_a_sequences.size(-1) != modal_b_sequences.size(-1):
            raise ValueError("The hidden sizes of modal_a and modal_b should be the same")
        merged_sequences = torch.cat([modal_a_sequences, modal_b_sequences], dim=1)
        merged_sequences = self.transformer_layer(merged_sequences)
        logits = self.cross_output_layer(merged_sequences)
        return logits

class FACTModel(nn.Module):
    """Audio Motion Multi-Modal model."""
    def __init__(self, config, is_training=True):
        super(FACTModel, self).__init__()
        self.config = config
        self.is_training = is_training
        self.feature_to_model, self.feature_to_params, self.feature_to_preprocessor = self.build_modalities_model(config['modality'])

        self.cross_modal_layer = CrossModalLayer(config['cross_modal_model'])
        motion_transformer_config = self.feature_to_model["motion"]["transformer_layer"]
        audio_transformer_config = self.feature_to_model["audio"]["transformer_layer"]

        self.motion_transformer = Transformer(
            hidden_size=motion_transformer_config['hidden_size'],
            num_hidden_layers=motion_transformer_config['num_hidden_layers'],
            num_attention_heads=motion_transformer_config['num_attention_heads'],
            intermediate_size=motion_transformer_config['intermediate_size']
        )
        self.motion_pos_embedding = PositionEmbedding(
            self.feature_to_params["motion"]["sequence_length"],
            motion_transformer_config['hidden_size']
        )
        self.motion_linear_embedding = LinearEmbedding(motion_transformer_config['hidden_size'])
        
        self.audio_transformer = Transformer(
            hidden_size=audio_transformer_config['hidden_size'],
            num_hidden_layers=audio_transformer_config['num_hidden_layers'],
            num_attention_heads=audio_transformer_config['num_attention_heads'],
            intermediate_size=audio_transformer_config['intermediate_size']
        )
        self.audio_pos_embedding = PositionEmbedding(
            self.feature_to_params["audio"]["sequence_length"],
            audio_transformer_config['hidden_size']
        )
        self.audio_linear_embedding = LinearEmbedding(audio_transformer_config['hidden_size'])

    def build_modalities_model(self, modality_config):
        # This function is a placeholder. You need to define how to build modalities model.
        # Returning mock values for demonstration purposes.
        feature_to_model = {
            "motion": {"transformer_layer": modality_config['motion']},
            "audio": {"transformer_layer": modality_config['audio']}
        }
        feature_to_params = {
            "motion": {"sequence_length": modality_config['motion']['sequence_length']},
            "audio": {"sequence_length": modality_config['audio']['sequence_length']}
        }
        feature_to_preprocessor = None
        return feature_to_model, feature_to_params, feature_to_preprocessor

    def forward(self, inputs):
        """Predict sequences from inputs."""
        # Computes motion features.
        motion_features = self.motion_linear_embedding(inputs["motion_input"])
        motion_features = self.motion_pos_embedding(motion_features)
        motion_features = self.motion_transformer(motion_features)

        # Computes audio features.
        audio_features = self.audio_linear_embedding(inputs["audio_input"])
        audio_features = self.audio_pos_embedding(audio_features)
        audio_features = self.audio_transformer(audio_features)

        # Computes cross modal output.
        output = self.cross_modal_layer(motion_features, audio_features)

        return output

    def infer_auto_regressive(self, inputs, steps=1200):
        """Predict sequences from inputs in an auto-regressive manner."""
        audio_seq_length = self.feature_to_params["audio"]["sequence_length"]
        outputs = []
        motion_input = inputs["motion_input"]
        for i in range(steps):
            audio_input = inputs["audio_input"][:, i: i + audio_seq_length]
            if audio_input.size(1) < audio_seq_length:
                break
            output = self.forward({"motion_input": motion_input, "audio_input": audio_input})
            output = output[:, 0:1, :]  # only keep the first frame
            outputs.append(output)
            # update motion input
            motion_input = torch.cat([motion_input[:, 1:, :], output], dim=1)
        return torch.cat(outputs, dim=1)

    def compute_motion_generation_loss(self, pred, target):
        """Compute motion generation loss from layer output."""
        target_seq_len = target.size(1)
        diff = target - pred[:, :target_seq_len]
        l2_loss = torch.mean(torch.square(diff))
        return l2_loss

    def loss(self, target, pred):
        """Compute loss."""
        motion_generation_loss = self.compute_motion_generation_loss(pred, target)
        return motion_generation_loss

    def get_metrics(self, eval_config):
        """Computes metrics."""
        # Placeholder for metrics computation.
        return []

# Sample configuration
class SampleConfig:
    hidden_size = 512
    num_hidden_layers = 6
    num_attention_heads = 8
    intermediate_size = 2048
    sequence_length = 100
    output_dim = 512
    cross_modal_model = {
        'hidden_size': 512,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 2048,
        'output_dim': 512
    }
    modality = {
        'motion': {
            'hidden_size': 512,
            'num_hidden_layers': 6,
            'num_attention_heads': 8,
            'intermediate_size': 2048,
            'sequence_length': 100
        },
        'audio': {
            'hidden_size': 512,
            'num_hidden_layers': 6,
            'num_attention_heads': 8,
            'intermediate_size': 2048,
            'sequence_length': 100
        }
    }

# Example usage
config = SampleConfig()
model = FACTModel(config.__dict__)

# Dummy inputs
inputs = {
    "motion_input": torch.randn(32, 100, 512),  # [batch_size, seq_length, feature_dim]
    "audio_input": torch.randn(32, 100, 512)
}

# Forward pass
outputs = model(inputs)
print("Model output shape:", outputs.shape)
