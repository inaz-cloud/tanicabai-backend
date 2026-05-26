import os
import traceback
import joblib
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime, timedelta, date

# =========================================================
# PATH CONFIG
# =========================================================

CURRENT_FILE = os.path.abspath(__file__)
SERVICE_DIR  = os.path.dirname(CURRENT_FILE)
APP_DIR      = os.path.dirname(SERVICE_DIR)
ROOT_DIR     = os.path.dirname(APP_DIR)

MODEL_DIR = os.path.join(ROOT_DIR, "models")
DATA_PATH = os.path.join(ROOT_DIR, "data", "data_bersih.csv")

WINDOW_SIZE = 45

# HARUS IDENTIK dengan FEATURE_COLS di train.py
FEATURE_COLS = [
    "harga",
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "avg_7hari",
    "rolling_std_7",
    "momentum_1",
    "momentum_7",
    "harga_rata_kota",
    "bulan_sin",
    "bulan_cos",
    "minggu_sin",
]

NUM_FEATURES      = len(FEATURE_COLS)   # 13
MAX_FORECAST_DAYS = 30
HISTORY_DAYS      = 30

PROVINSI  = os.environ.get("PROVINSI", "Jawa_Barat")
MAX_CACHE = 10

# =========================================================
# SUPABASE CLIENT
# =========================================================

try:
    from app.db.supabase_client import supabase as _supabase_client
    SUPABASE_AVAILABLE = True
except Exception:
    _supabase_client   = None
    SUPABASE_AVAILABLE = False

# =========================================================
# GLOBAL CACHE
# =========================================================

model_cache  = {}
scaler_cache = {}
cache_order  = []

# =========================================================
# INFO
# =========================================================

print("\n================================================")
print("🌶️  SISTEM PREDIKSI HARGA CABAI")
print("================================================")
print(f"📦 TensorFlow   : {tf.__version__}")
print(f"📁 MODEL DIR    : {MODEL_DIR}")
print(f"☁️  SUPABASE    : {'✅ tersedia' if SUPABASE_AVAILABLE else '⚠️  fallback CSV'}")
print(f"⚙️  NUM FEATURES : {NUM_FEATURES}")
print(f"⚙️  MAX FORECAST: H+{MAX_FORECAST_DAYS}")
print(f"⚙️  HISTORY DAYS: {HISTORY_DAYS} hari")
print("================================================\n")

# =========================================================
# HELPERS
# =========================================================

def clean_text(text: str) -> str:
    return "_".join([
        word.capitalize()
        for word in str(text).strip().replace("_", " ").split()
    ])

def _evict_cache_if_full():
    global model_cache, scaler_cache, cache_order
    while len(model_cache) >= MAX_CACHE and cache_order:
        oldest = cache_order.pop(0)
        model_cache.pop(oldest, None)
        scaler_cache.pop(oldest, None)

def _update_cache_order(key: str):
    if key in cache_order:
        cache_order.remove(key)
    cache_order.append(key)

# =========================================================
# LOAD MODEL + SCALER
# =========================================================

def load_model_and_scaler(wilayah: str, jenis_cabai: str):
    cache_key = f"{wilayah.lower()}_{jenis_cabai.lower()}"

    if cache_key in model_cache:
        _update_cache_order(cache_key)
        return model_cache[cache_key], scaler_cache[cache_key]

    kota  = clean_text(wilayah)
    jenis = clean_text(jenis_cabai)

    model_path  = os.path.join(MODEL_DIR, f"model_{PROVINSI}_{kota}_{jenis}.keras")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{PROVINSI}_{kota}_{jenis}.save")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler tidak ditemukan: {scaler_path}")
    if os.path.getsize(scaler_path) == 0:
        raise ValueError(f"File scaler corrupt: {scaler_path}")

    try:
        model  = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        print(f"✅ Model loaded: {os.path.basename(model_path)}")
    except Exception:
        traceback.print_exc()
        raise

    _evict_cache_if_full()
    model_cache[cache_key]  = model
    scaler_cache[cache_key] = scaler
    _update_cache_order(cache_key)

    return model, scaler

# =========================================================
# FETCH DATA — SUPABASE
# =========================================================

def _fetch_from_supabase(wilayah: str, jenis_cabai: str) -> pd.DataFrame | None:
    if not SUPABASE_AVAILABLE or _supabase_client is None:
        return None
    try:
        result = (
            _supabase_client
            .table("harga_cabai")
            .select("tanggal,harga,kota,provinsi,jenis_cabai")
            .ilike("kota", wilayah)
            .ilike("jenis_cabai", jenis_cabai)
            .order("tanggal", desc=False)
            .execute()
        )
        rows = result.data or []
        if len(rows) < WINDOW_SIZE + 14:   # butuh cukup data untuk semua lag
            print(f"  ⚠️  Supabase: data kurang ({len(rows)}), fallback CSV")
            return None

        df = pd.DataFrame(rows)
        df["tanggal"] = pd.to_datetime(df["tanggal"])
        df["harga"]   = pd.to_numeric(df["harga"], errors="coerce")
        df = df.dropna(subset=["harga"]).sort_values("tanggal").reset_index(drop=True)
        print(f"  ✅ Supabase: {len(df)} baris, terakhir {df.iloc[-1]['tanggal'].strftime('%Y-%m-%d')}")
        return df

    except Exception as e:
        print(f"  ⚠️  Supabase error: {e}, fallback CSV")
        return None

# =========================================================
# FETCH DATA — CSV (fallback)
# =========================================================

def _fetch_from_csv(wilayah: str, jenis_cabai: str) -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File CSV tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["tanggal"] = pd.to_datetime(df["tanggal"], dayfirst=True, errors="coerce")
    df["harga"]   = pd.to_numeric(df["harga"], errors="coerce")
    df = df.dropna(subset=["tanggal", "harga"])

    if "komoditas" in df.columns:
        df.rename(columns={"komoditas": "jenis_cabai"}, inplace=True)

    mask = (
        (df["kota"].str.lower() == wilayah.lower()) &
        (
            df["jenis_cabai"].str.replace("_", " ").str.lower()
            == jenis_cabai.replace("_", " ").lower()
        )
    )

    group = df[mask].copy().sort_values("tanggal").reset_index(drop=True)

    if len(group) == 0:
        raise ValueError(
            f"Data tidak ditemukan untuk wilayah='{wilayah}', jenis='{jenis_cabai}'"
        )

    print(f"  ✅ CSV: {len(group)} baris untuk {wilayah} - {jenis_cabai}")
    return group

# =========================================================
# FEATURE ENGINEERING  ← identik dengan train.py
# =========================================================

def _build_features(df: pd.DataFrame, wilayah: str) -> pd.DataFrame:
    """
    Membangun fitur yang SAMA PERSIS dengan train.py.
    df harus memiliki kolom: tanggal, harga.
    Karena service hanya menangani satu (kota, jenis_cabai),
    kolom provinsi / kota tidak digunakan untuk groupby —
    semua operasi langsung pada kolom harga.
    """
    df = df.copy().sort_values("tanggal").reset_index(drop=True)

    df["lag_1"]  = df["harga"].shift(1)
    df["lag_3"]  = df["harga"].shift(3)
    df["lag_7"]  = df["harga"].shift(7)
    df["lag_14"] = df["harga"].shift(14)

    df["avg_7hari"]     = df["harga"].rolling(7).mean()
    df["rolling_std_7"] = df["harga"].rolling(7).std()

    df["momentum_1"] = df["harga"] - df["lag_1"]
    df["momentum_7"] = df["harga"] - df["lag_7"]

    # harga_rata_kota: rata-rata keseluruhan harga pada kota ini
    df["harga_rata_kota"] = df["harga"].mean()

    bulan  = df["tanggal"].dt.month
    minggu = df["tanggal"].dt.isocalendar().week.astype(int)

    df["bulan_sin"]  = np.sin(2 * np.pi * bulan / 12)
    df["bulan_cos"]  = np.cos(2 * np.pi * bulan / 12)
    df["minggu_sin"] = np.sin(2 * np.pi * minggu / 52)

    df = df.dropna().reset_index(drop=True)
    return df

# =========================================================
# PREPARE WINDOW DATA
# =========================================================

def prepare_window_data(wilayah: str, jenis_cabai: str):
    df_raw = _fetch_from_supabase(wilayah, jenis_cabai)
    if df_raw is None:
        df_raw = _fetch_from_csv(wilayah, jenis_cabai)

    # Simpan df_raw hanya dengan kolom tanggal & harga untuk mode history
    df_raw_slim = df_raw[["tanggal", "harga"]].copy()

    df = _build_features(df_raw_slim, wilayah)

    if len(df) < WINDOW_SIZE:
        raise ValueError(
            f"Data historis kurang. Dibutuhkan {WINDOW_SIZE}, tersedia {len(df)}."
        )

    # Ambil WINDOW_SIZE baris terakhir dengan 13 fitur
    window_data = df[FEATURE_COLS].tail(WINDOW_SIZE).values.astype(np.float32)

    last_row         = df.iloc[-1]
    harga_terakhir   = float(last_row["harga"])
    tanggal_terakhir = last_row["tanggal"]

    return window_data, harga_terakhir, tanggal_terakhir, df_raw_slim

# =========================================================
# RECURSIVE MULTI-STEP FORECASTING
# =========================================================

def recursive_forecast(
    model,
    scaler,
    window_data: np.ndarray,
    n_days: int,
    start_date: datetime,
) -> list:
    """
    Prediksi rekursif H+1 … H+n_days.
    window_data : (WINDOW_SIZE, 13) — belum di-scale.
    """
    results      = []
    current_win  = window_data.copy()          # (45, 13), raw

    # Simpan harga mentah untuk menghitung fitur lag / rolling berikutnya
    harga_history = list(current_win[:, 0])    # kolom 0 = harga

    for day_idx in range(n_days):
        target_date = start_date + timedelta(days=day_idx + 1)

        # --- scale & predict ---
        scaled_win = scaler.transform(current_win)         # (45, 13)
        X          = scaled_win.reshape(1, WINDOW_SIZE, NUM_FEATURES)
        y_scaled   = model.predict(X, verbose=0)[0][0]

        # inverse transform: masukkan prediksi ke slot kolom-0, sisanya 0
        dummy       = np.zeros((1, NUM_FEATURES))
        dummy[0, 0] = y_scaled
        harga_pred  = float(scaler.inverse_transform(dummy)[0][0])
        harga_pred  = max(0.0, harga_pred)

        results.append({
            "tanggal": target_date.strftime("%Y-%m-%d"),
            "harga"  : round(harga_pred),
            "tipe"   : "prediksi",
        })

        harga_history.append(harga_pred)

        # --- bangun baris fitur baru (13 kolom, urutan sama dengan FEATURE_COLS) ---
        h   = harga_history
        n   = len(h)
        lag1  = h[-2] if n >= 2  else harga_pred
        lag3  = h[-4] if n >= 4  else harga_pred
        lag7  = h[-8] if n >= 8  else harga_pred
        lag14 = h[-15] if n >= 15 else harga_pred

        avg7  = float(np.mean(h[-7:]))
        std7  = float(np.std(h[-7:])) if len(h) >= 7 else 0.0

        mom1  = harga_pred - lag1
        mom7  = harga_pred - lag7

        # harga_rata_kota tetap sama (rata-rata historis)
        harga_rata = float(current_win[:, 9].mean())

        bulan  = target_date.month
        minggu = int(target_date.isocalendar()[1])

        bulan_sin  = np.sin(2 * np.pi * bulan / 12)
        bulan_cos  = np.cos(2 * np.pi * bulan / 12)
        minggu_sin = np.sin(2 * np.pi * minggu / 52)

        new_row = np.array([[
            harga_pred, lag1, lag3, lag7, lag14,
            avg7, std7, mom1, mom7, harga_rata,
            bulan_sin, bulan_cos, minggu_sin,
        ]], dtype=np.float32)                              # (1, 13)

        current_win = np.vstack([current_win[1:], new_row])  # geser window

    return results

# =========================================================
# MODE HISTORY
# =========================================================

def get_history(
    tanggal_target: datetime,
    df_raw: pd.DataFrame,
) -> list:
    start = tanggal_target - timedelta(days=HISTORY_DAYS)
    mask  = (
        (df_raw["tanggal"] >= pd.Timestamp(start)) &
        (df_raw["tanggal"] <= pd.Timestamp(tanggal_target))
    )
    subset = df_raw[mask].sort_values("tanggal")

    return [
        {
            "tanggal": row["tanggal"].strftime("%Y-%m-%d"),
            "harga"  : int(row["harga"]),
            "tipe"   : "aktual",
        }
        for _, row in subset.iterrows()
    ]

# =========================================================
# PREDICT PRICE — entry point utama
# =========================================================

def predict_price(data) -> dict:
    try:
        print("\n================================================")
        print("📈 PREDICT PRICE REQUEST")
        print(f"   Wilayah       : {data.wilayah}")
        print(f"   Jenis Cabai   : {data.jenis_cabai}")
        print(f"   Tanggal Target: {data.tanggal_target}")
        print("================================================")

        try:
            tanggal_target = datetime.strptime(data.tanggal_target, "%Y-%m-%d")
        except ValueError:
            raise ValueError("Format tanggal harus YYYY-MM-DD")

        today = datetime.combine(date.today(), datetime.min.time())

        # --- ambil & siapkan data historis ---
        window_data, harga_terakhir, tanggal_terakhir, df_raw = \
            prepare_window_data(data.wilayah, data.jenis_cabai)

        is_forecast = tanggal_target > today

        # =====================================================
        # MODE FORECAST
        # =====================================================
        if is_forecast:
            n_days = (tanggal_target - tanggal_terakhir).days

            if n_days <= 0:
                raise ValueError(
                    f"Tanggal target ({data.tanggal_target}) harus setelah "
                    f"data terakhir ({tanggal_terakhir.strftime('%Y-%m-%d')})."
                )
            if n_days > MAX_FORECAST_DAYS:
                raise ValueError(
                    f"Maksimal prediksi H+{MAX_FORECAST_DAYS}. "
                    f"Tanggal yang dipilih terlalu jauh ({n_days} hari)."
                )

            model, scaler = load_model_and_scaler(data.wilayah, data.jenis_cabai)

            print(f"📅 [FORECAST] {n_days} hari ke depan...")

            forecast_list = recursive_forecast(
                model=model,
                scaler=scaler,
                window_data=window_data,
                n_days=n_days,
                start_date=tanggal_terakhir,
            )

            harga_target = forecast_list[-1]["harga"]
            selisih      = harga_target - harga_terakhir
            tren         = "naik" if selisih > 0 else ("turun" if selisih < 0 else "stabil")
            pct_change   = (selisih / harga_terakhir * 100) if harga_terakhir > 0 else 0.0

            chart_data = [{
                "tanggal": tanggal_terakhir.strftime("%Y-%m-%d"),
                "harga"  : int(round(harga_terakhir)),
                "tipe"   : "aktual",
            }] + forecast_list

            print(f"✅ Forecast: Rp {harga_target:,.0f} ({tren} {abs(pct_change):.1f}%)")

            return {
                "status": "success",
                "mode"  : "forecast",
                "data"  : {
                    "wilayah"          : data.wilayah,
                    "jenis_cabai"      : data.jenis_cabai,
                    "tanggal_target"   : data.tanggal_target,
                    "tanggal_terakhir" : tanggal_terakhir.strftime("%Y-%m-%d"),
                    "harga_prediksi"   : harga_target,
                    "harga_terakhir"   : int(round(harga_terakhir)),
                    "tren"             : tren,
                    "perubahan_persen" : round(pct_change, 2),
                    "n_hari"           : n_days,
                    "sumber_data"      : "supabase" if SUPABASE_AVAILABLE else "csv",
                    "chart_data"       : chart_data,
                },
            }

        # =====================================================
        # MODE HISTORY
        # =====================================================
        else:
            print(f"📅 [HISTORY] Ambil data aktual untuk {data.tanggal_target}...")

            history_list = get_history(
                tanggal_target=tanggal_target,
                df_raw=df_raw,
            )

            if not history_list:
                raise ValueError(
                    f"Data aktual tidak ditemukan untuk tanggal {data.tanggal_target}."
                )

            harga_target = history_list[-1]["harga"]

            print(f"✅ History: {len(history_list)} titik, harga {data.tanggal_target}: Rp {harga_target:,.0f}")

            return {
                "status": "success",
                "mode"  : "history",
                "data"  : {
                    "wilayah"        : data.wilayah,
                    "jenis_cabai"    : data.jenis_cabai,
                    "tanggal_target" : data.tanggal_target,
                    "harga_aktual"   : harga_target,
                    "sumber_data"    : "supabase" if SUPABASE_AVAILABLE else "csv",
                    "chart_data"     : history_list,
                },
            }

    except FileNotFoundError as e:
        return {"status": "error", "code": "MODEL_NOT_FOUND", "message": str(e)}
    except ValueError as e:
        return {"status": "error", "code": "DATA_ERROR",      "message": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "code": "INTERNAL_ERROR",  "message": str(e)}