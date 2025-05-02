import torch
import torch.nn as nn
import torch.nn.functional as F
# Make sure these imports point to the correct file locations in your project
from Transformer_EncDec import Encoder, EncoderLayer
from SelfAttention_Family import FullAttention, AttentionLayer
from Embed import DataEmbedding_inverted
import numpy as np


class Model(nn.Module):
    """
    iTransformer model implementation.
    Paper link: https://arxiv.org/abs/2310.06625
    Modified forward method to select a single target feature for forecasting.
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in # Number of input features/variates (e.g., 513)

        # --- Add target_feature_idx to configs or set a default ---
        # Example: Get from configs, default to 0 if not present
        self.target_feature_idx = getattr(configs, 'target_feature_idx', 0)
        print(f"--- Model Info: Targeting feature index: {self.target_feature_idx} ---")
        # ----------------------------------------------------------

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.enc_in, configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder Projection Layer
        # This layer projects the embedding dimension back to the prediction length for *each* variate's representation
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Task-specific layers (keep others as they were)
        if self.task_name == 'imputation':
            # For imputation, projection might need to map back to seq_len
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True) # Adjust if needed
        if self.task_name == 'anomaly_detection':
             # For anomaly detection, projection might need to map back to seq_len
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True) # Adjust if needed
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            # For classification, project from flattened features to num_classes
            self.projection = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)


    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Core forecasting logic. Outputs predictions for ALL features.
        Output shape: [B, pred_len, N] where N is the number of features (enc_in).
        """
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach() # Mean across seq_len dimension; Shape: [B, 1, N]
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # Std dev across seq_len; Shape: [B, 1, N]
        x_enc /= stdev

        _, _, N = x_enc.shape # N is the number of input features (e.g., 513)

        # Embedding: Input shape [B, L, N], Output shape [B, N, d_model]
        # The DataEmbedding_inverted embeds features (N) instead of time steps (L)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder: Input shape [B, N, d_model], Output shape [B, N, d_model]
        # Attention is applied across the N features
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Decoder Projection: Input shape [B, N, d_model], Output shape [B, N, pred_len]
        # Projects d_model dimension to pred_len for each feature's representation
        dec_out = self.projection(enc_out)

        # Permute to bring pred_len to the middle dimension: Shape [B, pred_len, N]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        # Reshape stdev and means to match dec_out for broadcasting
        stdev_rep = stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1) # Shape [B, pred_len, N]
        means_rep = means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1) # Shape [B, pred_len, N]

        dec_out = dec_out * stdev_rep
        dec_out = dec_out + means_rep

        # Return predictions for all features
        return dec_out # Shape: [B, pred_len, N]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape # L is seq_len, N is num_features

        # Embedding & Encoder (operates on features)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # [B, N, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # [B, N, d_model]

        # Projection (for imputation, projects d_model back to seq_len)
        dec_out = self.projection(enc_out) # [B, N, seq_len]
        dec_out = dec_out.permute(0, 2, 1) # [B, seq_len, N]

        # De-Normalization
        stdev_rep = stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1) # Shape [B, L, N]
        means_rep = means[:, 0, :].unsqueeze(1).repeat(1, L, 1) # Shape [B, L, N]
        dec_out = dec_out * stdev_rep
        dec_out = dec_out + means_rep
        return dec_out # Shape: [B, L, N]

    def anomaly_detection(self, x_enc):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        _, L, N = x_enc.shape

        # Embedding & Encoder
        enc_out = self.enc_embedding(x_enc, None) # [B, N, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # [B, N, d_model]

        # Projection (for anomaly detection, projects d_model back to seq_len)
        dec_out = self.projection(enc_out) # [B, N, seq_len]
        dec_out = dec_out.permute(0, 2, 1) # [B, seq_len, N]

        # De-Normalization
        stdev_rep = stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1) # Shape [B, L, N]
        means_rep = means[:, 0, :].unsqueeze(1).repeat(1, L, 1) # Shape [B, L, N]
        dec_out = dec_out * stdev_rep
        dec_out = dec_out + means_rep
        return dec_out # Shape: [B, L, N] (outputs reconstruction)

    def classification(self, x_enc, x_mark_enc):
        # Embedding & Encoder
        enc_out = self.enc_embedding(x_enc, None) # [B, N, d_model] where N=enc_in
        enc_out, attns = self.encoder(enc_out, attn_mask=None) # [B, N, d_model]

        # Output Layer
        # Pool/Flatten features: Take the representations across all features
        output = enc_out.reshape(enc_out.shape[0], -1) # Shape: [B, N * d_model]
        output = self.act(output) # Apply activation
        output = self.dropout(output)
        output = self.projection(output) # Project to number of classes; Shape: [B, num_classes]
        return output # Shape: [B, num_classes]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 1. Get predictions for all features from the forecast method
            dec_out_all_features = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # dec_out_all_features shape: [B, pred_len, N] (e.g., [16, 1, 513])

            # 2. --- Modification Start: Select the target feature ---
            # We assume self.pred_len corresponds to the desired prediction steps.
            # We select the column corresponding to the target feature index.
            dec_out_target_feature = dec_out_all_features[:, :, self.target_feature_idx]
            # dec_out_target_feature shape: [B, pred_len] (e.g., [16, 1])
            # -------------------------------------------------------

            # 3. --- Reshape to match target/loss calculation ---
            # Add a final dimension to make it [B, pred_len, 1]
            # This matches the expected [B, 1, 1] if pred_len is 1.
            final_output = dec_out_target_feature.unsqueeze(-1)
            # --- Modification End ---

            return final_output # Shape: [B, pred_len, 1] (e.g., [16, 1, 1])

        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, num_classes]

        return None


