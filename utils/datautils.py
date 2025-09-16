import dataretrieval.nwis as nwis
import re

import numpy as np
import pandas as pd
import polars as pl

import jax
import jax.numpy as jnp

import threading
import queue

# Constants
CFS2M3S = 0.028316846592        # cfs to m^3/s

# As some time series might be incomplete we use brownian bridges to interpolate in between the gaps

def brownian_bridge(start, end, n_steps, step_std=1.0):
    """Generate Brownian bridge from start to end with n_steps points."""
    increments = np.random.normal(0, step_std, n_steps)
    walk = np.cumsum(increments)
    t = np.linspace(0, 1, n_steps)
    bridge = start + (walk - t * walk[-1]) + t * (end - start)
    return bridge

def fill_nans_with_random_walk(s, step_std=1.0, threshold=0.1):
    """Fill NaNs in df[col] with Brownian bridge random walks."""
    n = len(s)
    series = s.copy()
    series[series < threshold] = np.nan
    series[series < 0] = np.nan
    i = 0
    while i < n:
        if pd.isna(series.iloc[i]):
            # Start of NaN block
            start_idx = i - 1
            while i < n and pd.isna(series.iloc[i]):
                i += 1
            end_idx = i
            if start_idx >= 0 and end_idx < n:
                start_val = series.iloc[start_idx]
                end_val = series.iloc[end_idx]
                n_steps = end_idx - start_idx + 1  # include start and end
                bridge = brownian_bridge(start_val, end_val, n_steps, step_std)
                series.iloc[start_idx:end_idx + 1] = bridge
        else:
            i += 1
    return series.clip(lower=threshold)

def get_discharges(sites, nan_sites, service="iv", start_date="2005-01-01", end_date="2025-08-15"):

    df = nwis.get_record(sites=sites, service=service, start=start_date, end=end_date)
    Q = pd.DataFrame()
    if service == "iv":
        for site in sites:
            station_df = df.loc[site]
            if site in nan_sites:
                data = station_df["00060"]
            else:
                data = station_df["00060_15-minute update"]

            Q[site] = data
    else:
        for site in sites:
            station_df = df.loc[site]
            data = station_df["00060_Mean"]
            Q[site] = data
    return Q, df
    # bear in mind that the points are within 15 minute intervals

def get_data(path, sites, nan_sites, start_date="2005-01-01", end_date="2025-08-31"):
    try: 
        Q = pd.read_csv(path)
    except:
        Q, df = get_discharges(
            sites, nan_sites, 
            service="iv", 
            start_date=start_date, end_date=end_date
        )        
        Q.to_csv("./data/Q_raw.csv")
        df.to_csv("./data/df_raw.csv")
        #print(Q.isna().sum())
        #print(df.isna().sum())

    Q_filled = pd.DataFrame()
    for col in Q.columns:
        if col == "datetime":
            Q_filled[col] = Q[col]
        else:
            data = Q[col].interpolate(method="linear")
            data = fill_nans_with_random_walk(data, 0.5, 0.1)
            Q_filled[col] = data
    
    Q_filled = Q_filled.dropna()
    return Q, Q_filled

def feature_engineering(path, sites, nan_sites):
    Q_raw, Q = get_data(path, sites, nan_sites)
    Q = pl.from_pandas(Q)

    time = Q.select("datetime")
    time = time["datetime"].str.to_datetime("%Y-%m-%d %H:%M:%S%z")

    Q = Q.select(pl.col(pl.Float64))
    Q = Q.select([
        pl.col(c).mul(CFS2M3S).log10().alias(c)

        for c in Q.columns
    ])

    return Q.to_numpy(), Q.columns, time

def build_transformer_dataset(Q, time, in_stations, out_stations, enc_len, dec_len):
    """
    Build encoder/decoder sequences for Transformer training.

    Parameters
    ----------
    Q : np.array of shape (time_steps, stations)
        Full time series data.
    time : array-like of length time_steps
        Timestamps for each step.
    in_stations : list[int]
        Stations to use as encoder input.
    out_stations : list[int]
        Stations to predict (decoder target).
    enc_len : int
        Number of time steps for encoder.
    dec_len : int
        Number of time steps for decoder (horizon).

    Returns
    -------
    enc_seq : np.array of shape (num_samples, enc_len, len(in_stations))
    dec_in_seq : np.array of shape (num_samples, dec_len, len(out_stations))
        Decoder input (teacher forcing: starts with last encoder step repeated).
    dec_target : np.array of shape (num_samples, dec_len, len(out_stations))
        True decoder targets.
    dec_time : np.array of shape (num_samples, dec_len)
        Timestamps for each decoder step.
    """
    Q = np.asarray(Q)
    time = np.asarray(time)
    time_steps = Q.shape[0]

    max_start = time_steps - enc_len - dec_len
    start_idx = np.arange(max_start + 1)

    # Encoder indices
    enc_idx = start_idx[:, None] + np.arange(enc_len)[None, :]
    enc_seq = Q[enc_idx][:, :, in_stations]  # (num_samples, enc_len, n_in_stations)

    # Decoder target indices
    dec_idx = enc_idx[:, -1][:, None] + np.arange(1, dec_len + 1)[None, :]
    dec_target = Q[dec_idx][:, :, out_stations]

    # Decoder input: teacher forcing starts with last encoder step
    # Option 1: repeat last encoder step
    dec_in_seq = np.zeros_like(dec_target)
    dec_in_seq[:, 0, :] = enc_seq[:, -1, :len(out_stations)]
    # Optionally, for teacher forcing, fill the rest with true targets shifted by one:
    dec_in_seq[:, 1:, :] = dec_target[:, :-1, :]

    # Decoder timestamps
    dec_time = time[dec_idx]

    return enc_seq, dec_in_seq, dec_target, dec_time

def create_train_val_test(enc_seq, dec_in_seq, dec_target, dec_time,
                          train_frac=0.7, val_frac=0.15):
    """
    Split encoder/decoder dataset into train/val/test sets.
    Shapes:
        enc_seq: (N, enc_len, n_in)
        dec_in_seq: (N, dec_len, n_out)
        dec_target: (N, dec_len, n_out)
        dec_time: (N, dec_len)
    """
    Nt = enc_seq.shape[0]
    train_end = int(Nt * train_frac)
    val_end = int(Nt * (train_frac + val_frac))

    def slice_data(start, end):
        return {
            "enc": enc_seq[start:end],
            "dec_in": dec_in_seq[start:end],
            "dec_target": dec_target[start:end],
            "time": dec_time[start:end]
        }

    train = slice_data(0, train_end)
    val   = slice_data(train_end, val_end)
    test  = slice_data(val_end, Nt)

    return train, val, test

def batch_iterator(data, batch_size, shuffle=True):
    Nt = data["enc"].shape[0]
    indices = np.arange(Nt)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, Nt, batch_size):
        idx = indices[start_idx:start_idx + batch_size]
        batch = {
            "enc": jnp.array(data["enc"][idx]),
            "dec_in": jnp.array(data["dec_in"][idx]),
            "dec_target": jnp.array(data["dec_target"][idx]),
            "time": data["time"][idx]   # keep times as np/datetime
        }
        yield batch

def prefetch_batches(generator, prefetch_size=2):
    q = queue.Queue(maxsize=prefetch_size)

    def producer():
        for batch in generator:
            batch_device = {
                k: (jax.device_put(v) if k != "time" else v)
                for k, v in batch.items()
            }
            q.put(batch_device)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch

def trim_to_batches(arr, per_device_batch_size):
    """
    Trim array so that it is divisible by the global batch size.
    """
    n = arr.shape[0]
    global_batch_size = jax.device_count() * per_device_batch_size
    n_batches = n // global_batch_size
    valid_rows = n_batches * global_batch_size
    arr = arr[:valid_rows]

    return arr