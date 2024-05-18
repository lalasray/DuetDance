import torch
import torch.nn as nn

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


# Define the modified FACTModel
class FACTModel(nn.Module):
    """Audio Motion Multi-Modal model."""
    def __init__(self, config, is_training=True):
        super(FACTModel, self).__init__()
        self.config = config
        self.is_training = is_training

        # Motion Transformer 1 (existing)
        # Motion Transformer 2 (new)
        motion_transformer_config2 = self.config['motion_transformer2']
        self.motion_transformer2 = Transformer(
            hidden_size=motion_transformer_config2['hidden_size'],
            num_hidden_layers=motion_transformer_config2['num_hidden_layers'],
            num_attention_heads=motion_transformer_config2['num_attention_heads'],
            intermediate_size=motion_transformer_config2['intermediate_size']
        )
        self.motion_pos_embedding2 = PositionEmbedding(
            self.config['motion_seq_length'],
            motion_transformer_config2['hidden_size']
        )
        self.motion_linear_embedding2 = LinearEmbedding(motion_transformer_config2['hidden_size'])
        
        # Audio Transformer (existing)
        # Cross Modal Layer (existing)

    def forward(self, inputs):
        """Forward pass."""
        # Motion features pass through first motion transformer (existing)
        motion_features1 = self.motion_linear_embedding1(inputs["motion_input"])
        motion_features1 = self.motion_pos_embedding1(motion_features1)
        motion_features1 = self.motion_transformer1(motion_features1)

        # Motion features pass through second motion transformer (new)
        motion_features2 = self.motion_linear_embedding2(inputs["motion_input2"])
        motion_features2 = self.motion_pos_embedding2(motion_features2)
        motion_features2 = self.motion_transformer2(motion_features2)

        # Concatenate motion features
        motion_concatenated = torch.cat([motion_features1, motion_features2], dim=-1)

        # Audio features (existing)
        audio_features = self.audio_linear_embedding(inputs["audio_input"])
        audio_features = self.audio_pos_embedding(audio_features)
        audio_features = self.audio_transformer(audio_features)

        # Concatenate motion and audio features
        concatenated_features = torch.cat([motion_concatenated, audio_features], dim=1)

        # Cross Modal Layer (existing)
        output = self.cross_modal_layer(concatenated_features)

        return output

# Example usage
config = {
    'motion_transformer1': {
        'hidden_size': 512,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 2048
    },
    'motion_transformer2': {
        'hidden_size': 512,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 2048
    },
    'audio_transformer': {
        'hidden_size': 512,
        'num_hidden_layers': 6,
        'num_attention_heads': 8,
        'intermediate_size': 2048
    },
    'cross_modal_model': {
        'hidden_size': 768,
        'output_dim': 2  # Example output dimension
    }
}
model = FACTModel(config)

# Dummy inputs
inputs = {
    "motion_input": torch.randn(32, 100, 512),  # First motion input
    "motion_input2": torch.randn(32, 100, 512),  # Second motion input
    "audio_input": torch.randn(32, 100, 512)  # Audio input
}

# Forward pass
outputs = model(inputs)
print("Model output shape:", outputs.shape)
