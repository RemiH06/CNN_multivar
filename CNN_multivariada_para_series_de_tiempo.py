# wind_cnn_48h_exogenous.py
# Predicción 48h de velocidad del viento usando SOLO temp, humidity, clouds como entradas.

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import requests
import pandas as pd
import plotly.express as px
import tensorflow as tf
from tensorflow.keras import layers, models
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import os
import argparse
from typing import List

# -------------------------------------
# Parámetros por defecto
# -------------------------------------
CITY_TZ = "America/Mexico_City"
LAT, LON = 20.31, -103.18  # Chapala
PAST_DAYS = 15

TARGET_COL = "wind"
X_FEATURES = ["temp", "humidity", "clouds"]  # <- SOLO estas tres variables como entrada

OUT_CSV = "wind_forecast_48h_exogenous.csv"
OUT_HTML = "wind_forecast_48h_exogenous.html"

# -------------------------------------
# Utilidades
# -------------------------------------
def set_seeds(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def safe_localize_to_tz(idx: pd.DatetimeIndex, tz_str: str) -> pd.DatetimeIndex:
    tz = pytz.timezone(tz_str)
    if idx.tz is None:
        return idx.tz_localize(tz)
    return idx.tz_convert(tz)

def fetch_data(lat: float, lon: float, city_tz: str, past_days: int) -> pd.DataFrame:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,cloud_cover,wind_speed_10m"
        f"&past_days={past_days}"
        f"&timezone={city_tz}"
    )
    res = requests.get(url).json()
    df = pd.DataFrame({
        "time": pd.to_datetime(res["hourly"]["time"]),
        "temp": res["hourly"]["temperature_2m"],
        "humidity": res["hourly"]["relative_humidity_2m"],
        "clouds": res["hourly"]["cloud_cover"],
        "wind": res["hourly"]["wind_speed_10m"],
    }).set_index("time")
    df.index = safe_localize_to_tz(df.index, city_tz)

    # Filtrar hasta una hora atrás
    now_local = datetime.now(pytz.timezone(city_tz))
    cutoff = now_local - timedelta(hours=1)
    df = df[df.index <= cutoff]
    return df

# === Crear ventanas temporales ===
def create_sequences(X_scaled: pd.DataFrame, y_scaled: pd.Series, window: int):
    X, y = [], []
    X_vals = X_scaled.values
    y_vals = y_scaled.values
    for i in range(len(X_scaled) - window):
        X.append(X_vals[i:i+window, :])     # (window, n_features)
        y.append(y_vals[i+window])          # target en t+0
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def prepare_data(df: pd.DataFrame, x_features, target_col: str, window: int):
    # === Escalado con 2 scalers: uno para X y otro para y ===
    X_raw = df[x_features].copy()
    y_raw = df[target_col].copy()

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_scaled = pd.DataFrame(
        x_scaler.fit_transform(X_raw),
        columns=x_features,
        index=df.index
    )
    y_scaled = pd.Series(
        y_scaler.fit_transform(y_raw.values.reshape(-1, 1)).ravel(),
        index=df.index,
        name=target_col
    )

    # === Ventanas y split ===
    X, y = create_sequences(X_scaled, y_scaled, window)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_raw, y_raw, X_scaled, y_scaled, x_scaler, y_scaler, X_train, X_test, y_train, y_test

# === Construir la CNN en Keras ===
def build_cnn(window: int, n_features: int, filters: List[int], kernel_size: int, activation: str) -> tf.keras.Model:
    model = models.Sequential([layers.Input(shape=(window, n_features))])
    for f in filters:
        model.add(layers.Conv1D(f, kernel_size=kernel_size, activation=activation, padding='causal'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation=activation))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_evaluate(
    X_train, y_train, X_test, y_test,
    x_features, x_scaler, y_scaler,
    window: int, filters: List[int], kernel_size: int, activation: str,
    epochs: int, batch_size: int, seed: int
):
    print("\n[Construir CNN] filtros={}, kernel={}, activación={}".format(filters, kernel_size, activation))
    model = build_cnn(window, n_features=len(x_features), filters=filters, kernel_size=kernel_size, activation=activation)

    print("\n[Entrenamiento] epochs={}, batch_size={}, seed={}".format(epochs, batch_size, seed))
    set_seeds(seed)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # === Evaluación en escala original ===
    y_pred_test_scaled = model.predict(X_test, verbose=0).flatten()
    y_test_inv = y_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred_test_inv = y_scaler.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

    mae = np.mean(np.abs(y_test_inv - y_pred_test_inv))
    rmse = np.sqrt(np.mean((y_test_inv - y_pred_test_inv) ** 2))
    print(f"\n[Evaluación en test] MAE: {mae:.3f} m/s | RMSE: {rmse:.3f} m/s")

    # === Visualización Plotly: pred vs real en test ===
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test_inv, mode='lines', name='Real (test)'))
    fig.add_trace(go.Scatter(y=y_pred_test_inv, mode='lines', name='Predicción (test)'))
    fig.update_layout(title="Evaluación en conjunto de prueba (wind)",
                      template="plotly_white",
                      xaxis_title="Paso temporal (test)", yaxis_title="m/s")
    try:
        fig.show()
    except Exception:
        pass

    return model

# === Pronóstico 48 h autoregresivo con exógenas hold-last ===
def forecast_48h_exogenous_holdlast(
    X_scaled: pd.DataFrame,  # SOLO features escaladas (temp, humidity, clouds)
    model: tf.keras.Model,
    x_features, y_scaler,
    window: int, city_tz: str
):
    last_window = X_scaled.iloc[-window:].copy()       # (window, 3)
    last_feat_row = last_window.iloc[-1].values.copy() # valores más recientes de exógenas escaladas
    preds_scaled = []

    for _ in range(48):
        x = last_window.values[np.newaxis, ...]  # (1, window, n_features)
        yhat_scaled = model.predict(x, verbose=0)[0, 0]
        preds_scaled.append(yhat_scaled)

        # Mantener exógenas constantes (hold-last)
        new_future_feats = last_feat_row.copy()

        # Desplazar ventana y anexar la "nueva" fila de features
        last_window = pd.DataFrame(
            np.vstack([last_window.values[1:], new_future_feats]),
            columns=x_features,
            index=last_window.index[1:].append(pd.Index([last_window.index[-1] + pd.Timedelta(hours=1)]))
        )

    # Desescalar la serie pronosticada de viento
    wind_future = y_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).ravel()

    # Índice temporal futuro (48 horas siguientes)
    last_time = X_scaled.index[-1]
    future_idx = pd.date_range(
        last_time + pd.Timedelta(hours=1),
        periods=48,
        freq="H",
        tz=pytz.timezone(city_tz)
    )
    pred_df = pd.DataFrame({"pred_wind": wind_future}, index=future_idx)
    return pred_df

# === Visualización final y guardado ===
def plot_and_save(df: pd.DataFrame, pred_df: pd.DataFrame, out_html: str):
    fig = go.Figure()
    hist = df["wind"].iloc[-48:] if len(df) >= 48 else df["wind"]
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, mode='lines', name='Últimas observadas (wind)'))
    fig.add_trace(go.Scatter(x=pred_df.index, y=pred_df["pred_wind"].values, mode='lines+markers', name='Pronóstico 48 h (wind)'))
    fig.update_layout(title="Pronóstico de velocidad del viento (m/s) — 48 h (exógenas: temp/humidity/clouds)",
                      xaxis_title="Hora local", yaxis_title="m/s",
                      template="plotly_white")
    try:
        fig.show()
    except Exception:
        pass
    fig.write_html(out_html, include_plotlyjs="cdn")
    print(f"[OK] Gráfica guardada en: {os.path.abspath(out_html)}")

def save_csv(pred_df: pd.DataFrame, out_csv: str):
    pred_df.to_csv(out_csv, index_label="time")
    print(f"[OK] Predicciones guardadas en: {os.path.abspath(out_csv)}")

# -------------------------------------
# CLI
# -------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="CNN 1D multivariada (X: temp, humidity, clouds) → y: wind, horizonte 48h")
    p.add_argument("--window", type=int, default=24, help="Tamaño de ventana (12 o 24 sugeridos)")
    p.add_argument("--filters", type=int, nargs="+", default=[64, 32], help="Filtros por capa Conv1D (ej. 64 32 o 32 16)")
    p.add_argument("--kernel", type=int, default=3, help="Tamaño de kernel (2, 3, 5)")
    p.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"], help="Función de activación")
    p.add_argument("--epochs", type=int, default=30, help="Número de épocas")
    p.add_argument("--batch", type=int, default=32, help="Batch size")
    p.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad")
    return p.parse_args()

# -------------------------------------
# main
# -------------------------------------
def main():
    args = parse_args()
    print("\n==============================================")
    print(" CNN 1D — X:[temp, humidity, clouds] → y: wind")
    print("==============================================")
    print(f"Ventana: {args.window} | Filtros: {args.filters} | Kernel: {args.kernel} | Activación: {args.activation}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Seed: {args.seed}\n")

    # 1) Descargar y visualizar histórico
    df = fetch_data(LAT, LON, CITY_TZ, PAST_DAYS)
    try:
        fig_init = px.line(df, y=["temp","humidity","clouds","wind"], title="Variables meteorológicas (histórico)", template="plotly_white")
        fig_init.show()
    except Exception:
        pass

    # 2) Escalar datos + Crear ventanas + Split train/test
    _, _, X_scaled, y_scaled, x_scaler, y_scaler, X_train, X_test, y_train, y_test = prepare_data(
        df, X_FEATURES, TARGET_COL, args.window
    )

    # 3) Construir, entrenar (con semilla), evaluar y visualizar en test
    model = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        X_FEATURES, x_scaler, y_scaler,
        window=args.window,
        filters=args.filters,
        kernel_size=args.kernel,
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch,
        seed=args.seed
    )

    # 4) Pronóstico 48 h autoregresivo (exógenas hold-last)
    pred_48h = forecast_48h_exogenous_holdlast(
        X_scaled, model, X_FEATURES, y_scaler, args.window, CITY_TZ
    )

    # 5) Guardar y graficar
    save_csv(pred_48h, OUT_CSV)
    plot_and_save(df, pred_48h, OUT_HTML)

if __name__ == "__main__":
    main()
