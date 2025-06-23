"""
The code aims to implement a DeepAR model for time series forecasting.
"""

import pandas as pd
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_forecasting import TimeSeriesDataSet, DeepAR
from pytorch_forecasting.metrics import NormalDistributionLoss
from pytorch_forecasting.data import GroupNormalizer
from pathlib import Path
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import warnings
import random
import shutil

warnings.filterwarnings('ignore')

DATA_PATH = Path('/Users/zhangjiawei/Desktop/python_files/warwick_course/predictions/US_1_Retail_FINAL.csv')
TIME_COL = 'Date'
TARGET_COL = 'Demand Forecast'
GROUP_COL = 'Store'
STATIC_COLS = []
TIME_VARYING_KNOWN_COLS = ['dayofweek', 'weekofyear', 'month', 'is_weekend']
MAX_ENCODER_LENGTH = 35
MAX_PREDICTION_LENGTH = 5
BATCH_SIZE = 32
SEED_COUNT = 1
MAX_EPOCHS = 10
EARLY_STOP_PATIENCE = 5
SAVE_DIR = Path('/Users/zhangjiawei/Desktop/python_files/warwick_course/predictions/DeepARsave')


def prepare_data(data_path: Path):
    df = pd.read_csv(data_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL)
    df['dayofweek'] = df[TIME_COL].dt.dayofweek
    df['weekofyear'] = df[TIME_COL].dt.isocalendar().week.astype(int)
    df['month'] = df[TIME_COL].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['time_idx'] = (df[TIME_COL] - df[TIME_COL].min()).dt.days
    if GROUP_COL not in df.columns:
        df[GROUP_COL] = 1
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('time_idx')
    if TARGET_COL in numeric_cols:
        numeric_cols.remove(TARGET_COL)
    scaler = StandardScaler()
    df[numeric_cols + [TARGET_COL]] = scaler.fit_transform(df[numeric_cols + [TARGET_COL]])
    return df, scaler


def create_datasets(df: pd.DataFrame):
    total_times = df['time_idx'].nunique()
    test_size = int(total_times * 0.2)
    val_size = int(total_times * 0.1)
    train_cutoff = df['time_idx'].max() - test_size - val_size
    val_cutoff = df['time_idx'].max() - test_size

    train_df = df[df.time_idx <= train_cutoff]
    dataset_args = dict(
        time_idx='time_idx',
        target=TARGET_COL,
        group_ids=[GROUP_COL],
        min_encoder_length=MAX_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=STATIC_COLS,
        time_varying_known_reals=TIME_VARYING_KNOWN_COLS,
        time_varying_unknown_reals=[TARGET_COL],
        target_normalizer=GroupNormalizer(groups=[GROUP_COL]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    training = TimeSeriesDataSet(train_df, **dataset_args)

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        train_cutoff + 1,
        val_cutoff
    )

    testing = TimeSeriesDataSet.from_dataset(
        training,
        df,
        val_cutoff + 1
    )

    return training, validation, testing


def train_with_seed(seed: int, train_ds, val_ds):
    pl.seed_everything(seed)
    torch.manual_seed(seed)

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename=f'deepar-seed{seed}' + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    earlystop_cb = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        mode='min',
        min_delta=1e-4
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='cpu',
        callbacks=[earlystop_cb, checkpoint_cb],
        gradient_clip_val=0.1,
        enable_progress_bar=True,
        check_val_every_n_epoch=1 
    )

    model = DeepAR.from_dataset(
        train_ds,
        learning_rate=1e-3,
        hidden_size=128,
        rnn_layers=3,
        dropout=0.1,
        loss=NormalDistributionLoss()
    )

    train_dl = train_ds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0, shuffle=True)
    val_dl = val_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    
    try:
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    except Exception as e:
        return None, float('inf')

    return checkpoint_cb.best_model_path, checkpoint_cb.best_model_score.item()


def evaluate_model(model, test_ds, df, scaler):
    test_dl = test_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    preds = model.predict(test_dl, mode="prediction", return_x=False, n_samples=100)
    mean_preds = preds.mean(1).cpu().numpy()
    lower = np.quantile(preds.cpu().numpy(), 0.025, axis=1)
    upper = np.quantile(preds.cpu().numpy(), 0.975, axis=1)
    actuals = np.concatenate([y[0].cpu().numpy() for x, y in test_dl])

    mean_orig = scaler.inverse_transform(mean_preds.reshape(-1,1)).reshape(-1)
    lower_orig = scaler.inverse_transform(lower.reshape(-1,1)).reshape(-1)
    upper_orig = scaler.inverse_transform(upper.reshape(-1,1)).reshape(-1)
    actuals_orig = scaler.inverse_transform(actuals.reshape(-1,1)).reshape(-1)

    n_test = len(actuals_orig)
    dates_test = df.sort_values(TIME_COL)[TIME_COL].iloc[-n_test:].reset_index(drop=True)
    times_vis = dates_test + pd.Timedelta(days=MAX_ENCODER_LENGTH)

    plt.figure(figsize=(12,6))
    plt.fill_between(times_vis, lower_orig, upper_orig, alpha=0.3, label='95% CI')
    plt.plot(times_vis, actuals_orig, marker='o', markersize=3, linestyle='-', label='Ground Truth')
    plt.plot(times_vis, mean_orig, marker='s', markersize=3, linestyle='--', label='Prediction')
    plt.title('DeepAR Forecast vs Reality (Last 20% Test Period)')
    plt.xlabel('Date')
    plt.ylabel(TARGET_COL)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('deepar_predictions_with_ci.png', dpi=300, bbox_inches='tight')
    plt.close()

    mae = mean_absolute_error(actuals_orig, mean_orig)
    rmse = mean_squared_error(actuals_orig, mean_orig, squared=False)
    r2 = r2_score(actuals_orig, mean_orig)
    mape = mean_absolute_percentage_error(actuals_orig, mean_orig)
    print("\nModel Performance Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

if __name__ == "__main__":
    df, scaler = prepare_data(DATA_PATH)
    train_ds, val_ds, test_ds = create_datasets(df)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    seeds = random.sample(range(0, 10000), SEED_COUNT)
    best_loss = float('inf')
    best_ckpt = None
    for seed in seeds:
        ckpt_path, val_loss = train_with_seed(seed, train_ds, val_ds)
        if val_loss < best_loss:
            best_loss = val_loss
            best_ckpt = ckpt_path
    shutil.copy(best_ckpt, SAVE_DIR / 'best_model.ckpt')
    best_model = DeepAR.load_from_checkpoint(SAVE_DIR / 'best_model.ckpt')
    evaluate_model(best_model, test_ds, df, scaler)
