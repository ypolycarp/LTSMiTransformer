import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MemoryAugmentedModule(nn.Module):
    def __init__(self, d_model, memory_size, n_heads=None):
        super(MemoryAugmentedModule, self).__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads if n_heads else d_model

        # Memory is now per-head
        self.memory = nn.Parameter(torch.randn(memory_size, self.d_head))
        self.fc = nn.Linear(self.d_head, self.d_head)

    def forward(self, x):
        """
        x: (B, L, H*d_head) or (B, L, d_model)
        """
        B, L, _ = x.shape

        # Reshape based on whether we have heads or not
        if self.n_heads is not None:
            x = x.view(B, L, self.n_heads, self.d_head)
            # Process each head separately
            attn_scores = torch.einsum("blhd,md->blhm", x, self.memory)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            memory_out = torch.einsum("blhm,md->blhd", attn_weights, self.memory)
            memory_out = self.fc(memory_out)
            return memory_out.reshape(B, L, -1)
        else:
            # Original processing for non-head case
            attn_scores = torch.einsum("bld,md->blm", x, self.memory)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            memory_out = torch.einsum("blm,md->bld", attn_weights, self.memory)
            return self.fc(memory_out)

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, memory_size, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model

        self.attention = AttentionLayer(
            LearnableTemporalSparseMemoryAttention(
                d_model,
                memory_size,
                mask_flag=True,
                attention_dropout=dropout,
                n_heads=n_heads  # Pass n_heads here
            ),
            d_model,
            n_heads
        )

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            tau=tau,
            delta=delta
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class LearnableTemporalSparseMemoryEfficientAttention(nn.Module):
    def __init__(self, d_model, memory_size, mask_flag=True, scale=None,
                 attention_dropout=0.1, output_attention=False, n_heads=8, chunk_size=64):
        super(LearnableTemporalSparseMemoryEfficientAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads
        self.chunk_size = chunk_size  # Added chunk size parameter

        # Learnable sparsity generator
        self.sparsity_generator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Memory Module with heads information
        self.memory_module = MemoryAugmentedModule(d_model, memory_size, n_heads=n_heads)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        # Initialize output tensor
        V_out = torch.zeros(B, L, H, D, device=queries.device)

        # Process in chunks to save memory
        for i in range(0, L, self.chunk_size):
            # Get current chunk
            q_chunk = queries[:, i:i + self.chunk_size]
            chunk_len = q_chunk.size(1)

            # Compute attention scores for this chunk
            scores = torch.einsum("blhe,bshe->bhls", q_chunk, keys) * scale

            if self.mask_flag:
                if attn_mask is None:
                    # Generate dynamic sparse mask for this chunk
                    time_indices = torch.arange(i, i + chunk_len, device=queries.device).float().unsqueeze(1)
                    sparsity_scores = self.sparsity_generator(time_indices).squeeze(-1)
                    threshold = sparsity_scores.mean()
                    sparse_mask = sparsity_scores > threshold
                    sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0).expand(B, H, chunk_len, S)
                    chunk_mask = torch.ones_like(scores, dtype=torch.bool, device=queries.device) & sparse_mask
                else:
                    chunk_mask = attn_mask[:, :, i:i + self.chunk_size]

                scores.masked_fill_(~chunk_mask, -torch.inf)

            # Compute attention and output for this chunk
            A = self.dropout(torch.softmax(scores, dim=-1))
            V_chunk = torch.einsum("bhls,bshd->blhd", A, values)

            # Store chunk results
            V_out[:, i:i + self.chunk_size] = V_chunk

        # Integrate Memory Module
        V_out = self.memory_module(V_out.contiguous().view(B, L, -1))

        if self.output_attention:
            # Note: We don't store all attention matrices to save memory
            return (V_out.contiguous(), None)
        else:
            return (V_out.contiguous(), None)

class LearnableTemporalSparseMemoryAttention(nn.Module):
    def __init__(self, d_model, memory_size, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, n_heads=8):
        super(LearnableTemporalSparseMemoryAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads

        # Learnable sparsity generator
        self.sparsity_generator = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # Memory Module with heads information
        self.memory_module = MemoryAugmentedModule(d_model, memory_size, n_heads=n_heads)

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1. / math.sqrt(E)

        # Compute raw attention scores
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.ones_like(scores, dtype=torch.bool)
                time_indices = torch.arange(L, device=queries.device).float().unsqueeze(1)
                sparsity_scores = self.sparsity_generator(time_indices).squeeze(-1)
                threshold = sparsity_scores.mean()
                sparse_mask = sparsity_scores > threshold
                sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, S)
                attn_mask = attn_mask & sparse_mask

            scores.masked_fill_(~attn_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # Integrate Memory Module - no need to reshape as memory module handles it
        V_out = self.memory_module(V.contiguous().view(B, L, -1))

        if self.output_attention:
            return (V_out.contiguous(), A)
        else:
            return (V_out.contiguous(), None)

class LTSMiTransformer(nn.Module):
    """
       Modified iTransformer with TemporalSparseAttention that can learn, adapt and decide the appropriate top-k queries to search for
    """

    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len,
                 factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, class_strategy='projection', use_norm=True, memory_slots=20,
                 device=torch.device('cuda:0')):
        super(LTSMiTransformer, self).__init__()
        self.seq_len = seq_len
        self.pred_len = out_len
        self.output_attention = output_attention
        self.use_norm = use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)

        # Encoder with Memory-Augmented Temporal Sparse Attention
        self.encoder = Encoder(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    memory_size=memory_slots,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.projector = nn.Linear(d_model, out_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]
