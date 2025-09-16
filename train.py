
import jax 
import jax.numpy as jnp
import optax
import numpy as np

from utils.trainmodel import TrainModel
from utils.datautils import (
    feature_engineering, build_transformer_dataset, create_train_val_test
)

from model.TimeSeriesTransformer import (
    TimeSeriesTransformer, TimeSeriesTransformerConfig, quantile_loss_complex
)

sites = ["08165300", "08165500", "08166000", "08166140", "08166200"]
nan_sites = ["08166140", "08166200", "08165500"]

def main():
    # --------------------------------------
    # Get curated discharge data in the span of start_date="2005-01-01", end_date="2025-08-31"
    # --------------------------------------
    Q, cols, time = feature_engineering("./data/Q_raw.csv", sites, nan_sites)

    # --------------------------------------
    # Define Transformer encoder input stations
    # --------------------------------------
    in_stations = np.array([i for i in range(Q.shape[1])])
    
    # --------------------------------------
    # Define Transformer decoder target stations
    # --------------------------------------
    out_stations = np.array([cols.index("08166200")])

    # --------------------------------------
    # Define time windows for input and output
    # Encoder input -> 128 steps -> 32 h at 15 min resolution  
    # --------------------------------------
    enc_len = 128

    # --------------------------------------
    # Decoder input/output -> 32 steps -> 8 h at 15 min resolution  
    # --------------------------------------
    dec_len = 32

    # --------------------------------------
    # Build seq2seq training dataset 
    # --------------------------------------
    enc_seq, dec_in_seq, dec_target, dec_time = build_transformer_dataset(
        Q, time, in_stations, out_stations, enc_len, dec_len
    )

    # --------------------------------------
    # split dataset in train, val, test in these ratios (0.7, 0.15, 0.15) respectively 
    # --------------------------------------
    train, val, test = create_train_val_test(enc_seq, dec_in_seq, dec_target, dec_time)

    # --------------------------------------
    # initialize model training class (initializes distributed environment) 
    # --------------------------------------
    train_model = TrainModel()

    # --------------------------------------
    # we want to predict these quantiles
    # --------------------------------------
    quantiles = jnp.array([0.05, 0.5, 0.95])
    
    # --------------------------------------
    # set up model parameters
    # --------------------------------------
    d_model = 256
    n_heads = 4
    d_ff = 128
    num_encoder_layers = 2
    num_decoder_layers = 2
    dropout_rate = 0.1
    n_signals = train["dec_target"].shape[-1]
    n_quantiles = len(quantiles)
    max_len = max(enc_len, dec_len)

    cfg = TimeSeriesTransformerConfig(
        d_model = d_model,
        n_heads = n_heads,
        d_ff = d_ff,
        num_encoder_layers = num_encoder_layers,
        num_decoder_layers = num_decoder_layers,
        dropout_rate = dropout_rate,
        n_signals = n_signals,
        n_quantiles = n_quantiles,
        max_len = max_len
    )
    
    # --------------------------------------
    # set up optimizer parameters
    # --------------------------------------

    tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    
    loss_fn = lambda x,y: quantile_loss_complex(
        x, y, quantiles, 
        crossing_penalty_coef=0.2, cov_weight=1.0, k=100, mae_coef=1.0
    )

    # --------------------------------------
    # set training parameters
    # --------------------------------------

    params = {
        "model": TimeSeriesTransformer(cfg), 
        "train_set": train,
        "val_set" : val,
        "n_epochs": 10, 
        "optimizer": tx, 
        "loss_fn": loss_fn, 
        "device_batches": 64,
        "seed": 123
    }

    train_model.set_training_params(params)
    train_model.train(in_stations, out_stations)
    train_model.plot_losses()


if __name__ == "__main__":
    main()

