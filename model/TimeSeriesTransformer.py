from typing import Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

@dataclass
class TimeSeriesTransformerConfig:
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    max_len: int = 512
    dropout_rate: float = 0.1
    n_signals: int = 5            # number of input variables
    n_quantiles: int = 3          # >1 if forecasting quantiles
    activation: str = "relu"

def sinusoidal_init(max_len: int, d_model: int) -> jnp.ndarray:
    position = jnp.arange(max_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(jnp.log(10000.0) / d_model))
    pe = jnp.zeros((max_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

def make_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return mask[None, None, :, :]

class MultiHeadSelfAttention(nn.Module):
    d_model: int
    num_heads: int
    dropout_rate: float

    @nn.compact
    def __call__(self, q, k, v, mask: Optional[jnp.ndarray], deterministic=True):
        head_dim = self.d_model // self.num_heads
        dense = nn.DenseGeneral
        q_proj = dense((self.num_heads, head_dim))(q)
        k_proj = dense((self.num_heads, head_dim))(k)
        v_proj = dense((self.num_heads, head_dim))(v)
        
        logits = jnp.einsum("...qhd,...khd->...hqk", q_proj, k_proj) / jnp.sqrt(head_dim)

        if mask is not None:
            logits = jnp.where(mask, logits, -1e9)

        weights = nn.softmax(logits, axis=-1)
        weights = nn.Dropout(self.dropout_rate)(weights, deterministic=deterministic)
        attn = jnp.einsum("...hqk,...khd->...qhd", weights, v_proj)

        out = attn.reshape(*attn.shape[:-2], self.d_model)
        out = nn.Dense(self.d_model)(out)
        out = nn.Dropout(self.dropout_rate)(out, deterministic=deterministic)
        return out

class FeedForward(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, deterministic=True):
        act_fn = {"relu": nn.relu, "gelu": nn.gelu}[self.activation]
        x = nn.Dense(self.d_ff)(x)
        x = act_fn(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)
        return x

class EncoderLayer(nn.Module):
    cfg: TimeSeriesTransformerConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        y = nn.LayerNorm()(x)
        y = MultiHeadSelfAttention(self.cfg.d_model, self.cfg.n_heads, self.cfg.dropout_rate)(y, y, y, None, deterministic)
        x = x + y
        y = nn.LayerNorm()(x)
        y = FeedForward(self.cfg.d_model, self.cfg.d_ff, self.cfg.dropout_rate, self.cfg.activation)(y, deterministic)
        return x + y

class DecoderLayer(nn.Module):
    cfg: TimeSeriesTransformerConfig

    @nn.compact
    def __call__(self, x, enc_out, tgt_mask, deterministic=True):
        y = nn.LayerNorm()(x)
        y = MultiHeadSelfAttention(self.cfg.d_model, self.cfg.n_heads, self.cfg.dropout_rate)(y, y, y, tgt_mask, deterministic)
        x = x + y
        y = nn.LayerNorm()(x)
        y = MultiHeadSelfAttention(self.cfg.d_model, self.cfg.n_heads, self.cfg.dropout_rate)(y, enc_out, enc_out, None, deterministic)
        x = x + y
        y = nn.LayerNorm()(x)
        y = FeedForward(self.cfg.d_model, self.cfg.d_ff, self.cfg.dropout_rate, self.cfg.activation)(y, deterministic)
        return x + y

class TimeSeriesTransformer(nn.Module):
    cfg: TimeSeriesTransformerConfig
    __name__: str = "TimeSeriesTransformer"

    @nn.compact
    def __call__(self, enc_in, dec_in, deterministic=True):
        enc = nn.Dense(self.cfg.d_model)(enc_in)
        dec = nn.Dense(self.cfg.d_model)(dec_in)

        pe = sinusoidal_init(self.cfg.max_len, self.cfg.d_model)
        enc = enc + pe[:enc.shape[1]]
        dec = dec + pe[:dec.shape[1]]

        for _ in range(self.cfg.num_encoder_layers):
            enc = EncoderLayer(self.cfg)(enc, deterministic)

        tgt_mask = make_causal_mask(dec.shape[1])
        for _ in range(self.cfg.num_decoder_layers):
            dec = DecoderLayer(self.cfg)(dec, enc, tgt_mask, deterministic)

        out = nn.Dense(self.cfg.n_signals * self.cfg.n_quantiles)(dec)
        out = out.reshape(dec.shape[0], dec.shape[1], self.cfg.n_signals, self.cfg.n_quantiles)
        return out


@jax.jit
def quantile_loss_complex(
    y_pred: jnp.ndarray,        # (B, dec_len, n_signals, n_quantiles)
    y_true: jnp.ndarray,        # (B, dec_len, n_signals, 1)
    quantiles: jnp.ndarray,     # (n_quantiles,)
    crossing_penalty_coef: float = 0.2,
    cov_weight: float = 1.0,
    k: float = 100,
    mae_coef: float = 1.0
):
    # Broadcast y_true to match quantile dimension
    # y_true_q = jnp.broadcast_to(y_true, y_pred.shape)  # (B, dec_len, n_signals, n_quantiles)

    error = y_true[..., None] - y_pred
    abs_e = jnp.abs(error)

    # Huberized error
    delta = 0.2
    huber_e = jnp.where(abs_e <= delta, 0.5 * error**2 / delta, abs_e - 0.5 * delta)

    # Pinball / quantile loss
    quantile_loss = jnp.maximum(quantiles * huber_e, (quantiles - 1.0) * huber_e)
    quantile_loss = jnp.mean(quantile_loss)

    # Crossing penalty
    def compute_penalty(_):
        return jnp.mean(jnp.maximum(0, y_pred[..., :-1] - y_pred[..., 1:]))

    crossing_penalty = jax.lax.cond(
        crossing_penalty_coef > 0.0, compute_penalty, lambda _: 0.0, operand=None
    )

    # Coverage using extreme quantiles only
    indicator_low = jax.nn.sigmoid(k * (y_true - y_pred[..., 0]))
    indicator_high = jax.nn.sigmoid(k * (y_pred[..., -1] - y_true))
    cov = jnp.mean(indicator_high * indicator_low)
    target_cov = quantiles.max() - quantiles.min()
    cov_loss = (cov - target_cov)**2

    # Median absolute error
    median_idx = jnp.argmin(jnp.abs(quantiles - 0.5))
    mae = jnp.mean(jnp.abs(y_true - y_pred[..., median_idx]))

    total_loss = (
        quantile_loss
        + crossing_penalty_coef * crossing_penalty
        + cov_weight * cov_loss
        + mae_coef * mae
    )

    return total_loss
