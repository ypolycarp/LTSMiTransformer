import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
def compute_K_tau_L(tau: torch.Tensor, L: int) -> int:
    """Compute K(τ,L)=ceil((1+τ) log2 L)."""
    return int(math.ceil((1.0 + float(torch.clamp(tau, 0.0, 1.0).item())) * math.log2(max(int(L), 2))))

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

class LearnableTemporalSparseMemoryAttention(nn.Module):
    """
    Learnable Temporal Sparse Attention with an external Memory-Augmented Module (MAM).

    This implementation matches the paper's τ–TopK formulation:
      - score shifting: s̃_t = s_t - τ
      - τ-controlled logarithmic budget: K(τ,L)=ceil((1+τ) log2 L)
      - hard TopK sparsification in the forward pass, with an STE-style surrogate in backprop.
    """

    def __init__(
        self,
        d_model,
        memory_size,
        mask_flag=True,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        n_heads=8,
        tau_init=0.5,
        ste_temperature=10.0,
        neg_inf=-1e9,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads

        # Learnable sparsity controller τ (scalar in (0,1))
        self.tau = nn.Parameter(torch.tensor(float(tau_init), dtype=torch.float32))

        # Lightweight scorer: maps per-timestep head-dim features -> scalar score in (0,1)
        # We score using key representations aggregated across batch and heads.
        d_head = d_model // n_heads
        self.score_net = nn.Sequential(
            nn.Linear(d_head, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        # Memory module (per-head memory, as implemented)
        self.memory_module = MemoryAugmentedModule(d_model, memory_size, n_heads=n_heads)

        # STE controls
        self.ste_temperature = float(ste_temperature)
        self.neg_inf = float(neg_inf)

    @staticmethod
    def _K_tau_L(tau_scalar: float, L: int) -> int:
        # K(τ,L)=ceil((1+τ) log2 L)
        return int(math.ceil((1.0 + tau_scalar) * math.log2(max(int(L), 2))))

    def effective_K(self, L: int, tau_override: torch.Tensor | None = None) -> int:
        tau_eff = tau_override if tau_override is not None else self.tau
        tau_val = float(torch.clamp(tau_eff, 0.0, 1.0).item())
        return self._K_tau_L(tau_val, L)

    def _build_mask_ste(self, s_tilde: torch.Tensor, K: int) -> torch.Tensor:
        """
        Builds an STE mask over key positions.

        Forward: hard TopK mask.
        Backward: uses a smooth surrogate (sigmoid over s̃) to provide gradients.
        """
        L = s_tilde.shape[0]
        K = max(1, min(int(K), int(L)))

        # Hard TopK (forward)
        _, idx = torch.topk(s_tilde, k=K, dim=0)
        m_hard = torch.zeros(L, device=s_tilde.device, dtype=torch.float32)
        m_hard[idx] = 1.0

        # Smooth surrogate (backward)
        m_soft = torch.sigmoid(self.ste_temperature * s_tilde)

        # STE composition
        m = m_hard + (m_soft - m_soft.detach())
        return m  # [L] in (approx) [0,1]

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        """
        Args:
            queries, keys, values: [B, L, H, E] where E is head dim
            attn_mask: optional boolean mask [B, 1 or H, L, S] (self-attn so S=L)
            tau: optional override τ for ablations (e.g., fixed τ)
        Returns:
            V_out: [B, L, d_model]
            A (optional): [B, H, L, S]
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        assert L == S, "This LTSA implementation assumes self-attention (L == S)."

        scale = self.scale or 1.0 / math.sqrt(E)
        device = queries.device

        # τ (override for ablations if provided)
        tau_eff = tau if tau is not None else self.tau
        tau_eff = torch.clamp(tau_eff, 0.0, 1.0)

        # 1) Temporal scoring (global per-position) from keys: average over batch and heads
        # keys: [B, L, H, E] -> [L, E]
        k_feat = keys.mean(dim=(0, 2))                      # [L, E]
        s = self.score_net(k_feat).squeeze(-1)              # [L] in (0,1)

        # 2) τ-modulated score shifting: s̃ = s - τ
        s_tilde = s - tau_eff                               # [L]

        # 3) τ-controlled logarithmic budget
        K = self.effective_K(L, tau_override=tau_eff)

        # 4) Hard TopK mask with STE gradients
        m = self._build_mask_ste(s_tilde, K)                # [L]

        # 5) Full attention scores then sparsify keys via mask
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * scale  # [B,H,L,S]
        # Apply mask over S dimension: add a large negative bias where m≈0 (forward hard)
        scores = scores + (m.view(1, 1, 1, S) - 1.0) * self.neg_inf

        # Optional causal/other mask (if provided)
        if self.mask_flag and attn_mask is not None:
            if attn_mask.dim() != 4:
                raise ValueError("attn_mask should have shape [B, 1/H, L, S].")
            if attn_mask.shape[1] == 1:
                attn_mask = attn_mask.expand(B, H, L, S)
            scores = scores.masked_fill(~attn_mask, self.neg_inf)

        A = self.dropout(torch.softmax(scores, dim=-1))     # [B,H,L,S]
        V = torch.einsum("bhls,bshd->blhd", A, values)      # [B,L,H,D]

        # 6) Memory integration
        V_flat = V.contiguous().view(B, L, -1)              # [B,L,d_model]
        V_out = self.memory_module(V_flat)                  # [B,L,d_model]

        if self.output_attention:
            return V_out.contiguous(), A
        return V_out.contiguous(), None

class TSMiTransformer(nn.Module):
    """
    Fixed-τ ablation (TSMiTransformer): same backbone as LTSMiTransformer, but τ is held constant
    (selected via a lightweight validation sweep) and passed as an override to LTSA.
    """

    def __init__(
        self,
        enc_in,
        dec_in,
        c_out,
        seq_len,
        label_len,
        out_len,
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.0,
        attn='prob',
        embed='fixed',
        freq='h',
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True,
        class_strategy='projection',
        use_norm=True,
        memory_slots=20,
        fixed_tau=0.5,
        device=torch.device('cuda:0'),
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = out_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.fixed_tau = float(fixed_tau)

        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, embed, freq, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    memory_size=memory_slots,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.projector = nn.Linear(d_model, out_len, bias=True)

    def get_tau(self) -> float:
        return float(self.fixed_tau)

    def effective_K(self, L: int | None = None) -> int:
        L = int(L) if L is not None else int(self.seq_len)
        return int(math.ceil((1.0 + self.fixed_tau) * math.log2(max(int(L), 2))))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        tau_override = torch.tensor(self.fixed_tau, device=enc_out.device, dtype=torch.float32)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau_override)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]