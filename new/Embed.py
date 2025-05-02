import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, seq_len, d_model, num_features, dropout=0.1):
        """
        iTransformer Embedding with Convolutional Feature Extraction.

        Args:
            seq_len (int): Input sequence length (L).
            d_model (int): Dimension of model hidden states.
            num_features (int): Number of variates/features (N).
            dropout (float): Dropout rate.
        """
        super(DataEmbedding_inverted, self).__init__()
        self.num_features = num_features
        self.d_model = d_model

        # 1. 1D Convolutional Layers to process the time dimension (L)
        # We treat each variate's sequence independently.
        # Input shape for convs: [B*N, 1, L]
        # Output shape target after convs and pooling: [B*N, d_model]

        # Example CNN structure (tune these parameters):
        # Layer 1: Input [1, L] -> Output [d_model/2, L_out1]
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=d_model // 2,
            kernel_size=3,
            padding=1, # Preserves length with kernel_size=3
            bias=False
        )
        self.relu1 = nn.ReLU()
        # Layer 2: Input [d_model/2, L_out1] -> Output [d_model, L_out2]
        self.conv2 = nn.Conv1d(
            in_channels=d_model // 2,
            out_channels=d_model,
            kernel_size=3,
            padding=1, # Preserves length with kernel_size=3
            bias=False
        )
        self.relu2 = nn.ReLU()

        # 2. Global Average Pooling to aggregate features over the time dimension
        # Input shape: [B*N, d_model, L_out2] -> Output shape: [B*N, d_model, 1]
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 3. Positional Embedding (learnable) for each variate
        # Shape: [1, N, d_model]
        self.position_embedding = nn.Parameter(torch.randn(1, num_features, d_model))

        # 4. Dropout Layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark): # x shape: [B, L, N], x_mark is ignored here
        """
        Forward pass for convolutional inverted embedding.

        Args:
            x (torch.Tensor): Input tensor with shape [Batch, SeqLen, NumFeatures].
            x_mark (torch.Tensor): Temporal features (ignored in this embedding).

        Returns:
            torch.Tensor: Output embedding with shape [Batch, NumFeatures, d_model].
        """
        B, L, N = x.shape
        assert N == self.num_features, f"Input feature mismatch: expected {self.num_features}, got {N}"

        # 1. Permute: [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1)

        # 2. Reshape for independent convolutional processing: [B, N, L] -> [B*N, 1, L]
        x_reshaped = x.reshape(B * N, 1, L)

        # 3. Apply CNN layers
        x_conv = self.relu1(self.conv1(x_reshaped)) # [B*N, d_model/2, L]
        x_conv = self.relu2(self.conv2(x_conv))     # [B*N, d_model, L]

        # 4. Apply Global Average Pooling over the time dimension (L)
        x_pooled = self.global_pool(x_conv) # [B*N, d_model, 1]

        # 5. Squeeze the last dimension: [B*N, d_model, 1] -> [B*N, d_model]
        x_squeezed = x_pooled.squeeze(-1)

        # 6. Reshape back to [B, N, d_model]
        value_embedding = x_squeezed.reshape(B, N, self.d_model)

        # 7. Add positional embedding (broadcasts along Batch dimension)
        output = value_embedding + self.position_embedding

        # 8. Apply dropout
        return self.dropout(output) # Final shape: [B, N, d_model]

class DataEmbedding_inverted_Patch(nn.Module):
    def __init__(self, seq_len, patch_len, stride, d_model, num_features, dropout=0.1):
        """
        iTransformer Embedding using Patches inspired by PatchTST.

        Args:
            seq_len (int): Input sequence length (L).
            patch_len (int): Length of each patch (P).
            stride (int): Stride between consecutive patches (S).
            d_model (int): Dimension of model hidden states.
            num_features (int): Number of variates/features (N).
            dropout (float): Dropout rate.
        """
        super(DataEmbedding_inverted_Patch, self).__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_features = num_features

        # Calculate the number of patches
        # Formula: floor((L - P) / S) + 1
        self.num_patches = math.floor((seq_len - patch_len) / stride) + 1
        print(f"--- Patch Embedding Info ---")
        print(f"Seq Len: {seq_len}, Patch Len: {patch_len}, Stride: {stride}")
        print(f"Number of patches: {self.num_patches}")
        print(f"---------------------------")


        # 1. Patching happens implicitly in the forward pass using unfold.

        # 2. Patch Embedding Layer
        # Embeds each patch of length `patch_len` into `d_model` dimension.
        self.patch_embedding = nn.Linear(patch_len, d_model, bias=True)

        # 3. Positional Embedding (learnable) for each variate (remains the same)
        # Shape: [1, N, d_model]
        self.position_embedding = nn.Parameter(torch.randn(1, num_features, d_model))

        # 4. Aggregation Strategy (Here we use simple averaging over patches)
        # Other strategies like using a small Transformer could be explored.

        # 5. Dropout Layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark): # x shape: [B, L, N], x_mark is ignored here
        """
        Forward pass for patch-based inverted embedding.

        Args:
            x (torch.Tensor): Input tensor with shape [Batch, SeqLen, NumFeatures].
            x_mark (torch.Tensor): Temporal features (ignored in this embedding).

        Returns:
            torch.Tensor: Output embedding with shape [Batch, NumFeatures, d_model].
        """
        B, L, N = x.shape
        assert L == self.seq_len, f"Input sequence length mismatch: expected {self.seq_len}, got {L}"
        assert N == self.num_features, f"Input feature mismatch: expected {self.num_features}, got {N}"

        # 1. Permute: [B, L, N] -> [B, N, L]
        # We want to patch along the L dimension for each N
        x = x.permute(0, 2, 1)

        # 2. Reshape for patching: [B, N, L] -> [B*N, L]
        # Process each variate's sequence independently
        x_reshaped = x.reshape(B * N, L)

        # 3. Create Patches using unfold: [B*N, L] -> [B*N, num_patches, patch_len]
        # unfold(dimension, size, step)
        x_unfolded = x_reshaped.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # 4. Apply Patch Embedding Layer: [B*N, num_patches, patch_len] -> [B*N, num_patches, d_model]
        x_embedded_patches = self.patch_embedding(x_unfolded)

        # 5. Aggregate Patch Embeddings: [B*N, num_patches, d_model] -> [B*N, d_model]
        # Using simple average pooling over the num_patches dimension.
        value_embedding_aggregated = torch.mean(x_embedded_patches, dim=1)

        # 6. Reshape back to [B, N, d_model]
        value_embedding = value_embedding_aggregated.reshape(B, N, self.d_model)

        # 7. Add positional embedding (broadcasts along Batch dimension)
        output = value_embedding + self.position_embedding

        # 8. Apply dropout
        return self.dropout(output) # Final shape: [B, N, d_model]


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars