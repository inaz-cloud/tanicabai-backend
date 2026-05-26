import os

# =========================================================
# Set env SEBELUM import TF/Keras
# =========================================================
os.environ["KERAS_BACKEND"] = "tensorflow"

import traceback
import tensorflow as tf
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.routes import prediction, price, history  # ← history sudah diimport di sini

_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")

ALLOWED_ORIGINS = (
    [o.strip() for o in _origins_env.split(",")]
    if _origins_env != "*"
    else ["*"]
)

# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "=" * 60)
    print("🌶️  SISTEM CABAI AI STARTED")
    print("=" * 60)
    print(f"✅ TensorFlow : {tf.__version__}")
    print(f"✅ CORS origins: {ALLOWED_ORIGINS}")
    print("=" * 60 + "\n")

    yield

    print("\n🌶️  SISTEM CABAI AI SHUTDOWN")

# =========================================================
# FASTAPI INIT
# =========================================================

app = FastAPI(
    title       = "Sistem Deteksi & Prediksi Cabai",
    description = (
        "API Deteksi Penyakit Daun Cabai "
        "dan Prediksi Harga Cabai "
        "menggunakan Deep Learning"
    ),
    version     = "2.0.0",
    lifespan    = lifespan,
)

# =========================================================
# CORS MIDDLEWARE
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOWED_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# =========================================================
# GLOBAL ERROR HANDLER
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"\n❌ UNHANDLED ERROR: {request.method} {request.url}")
    print(f"DETAIL: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "status" : "error",
            "message": "Internal server error",
            "detail" : str(exc),
        },
    )

# =========================================================
# ROUTES
# =========================================================

app.include_router(
    prediction.router,
    prefix="/predict",
    tags=["Disease Detection"],
)

app.include_router(
    price.router,
    prefix="/price",
    tags=["Price Prediction"],
)

app.include_router(
    history.router,      # ← pakai history.router, bukan history_router
    prefix="/history",
    tags=["History"],
)

# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "status"     : "online",
        "message"    : "Sistem Cabai AI berjalan dengan normal",
        "tensorflow" : tf.__version__,
        "version"    : "2.0.0",
    }

# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status"     : "healthy",
        "tensorflow" : tf.__version__,
    }