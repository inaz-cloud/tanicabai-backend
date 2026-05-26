from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from app.services.disease_services import predict_disease
from app.db.supabase_client import supabase
from app.utils.auth import get_user_id
from datetime import datetime
import traceback
import uuid
import io

# =========================================================
# CONFIG
# =========================================================

MAX_FILE_SIZE_MB    = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_TYPES       = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
    "application/octet-stream"
}

STORAGE_BUCKET      = "prediction-images"
THUMBNAIL_SIZE      = (224, 224)   # ukuran thumbnail yang disimpan
THUMBNAIL_QUALITY   = 75           # JPEG quality 75 → ~15-30 KB per gambar

# =========================================================
# ROUTER
# =========================================================

router = APIRouter()

# =========================================================
# PREDICT ROUTE
# =========================================================

@router.post("/")
async def predict(request: Request, file: UploadFile = File(...)):

    try:
        print("\n================================================")
        print("🌶️  REQUEST DETEKSI PENYAKIT")
        print("================================================")
        print(f"📄 Filename : {file.filename}")
        print(f"📦 Content  : {file.content_type}")
        print("================================================")

        # Ambil user_id dari token (opsional)
        user_id = await get_user_id(request)
        print(f"👤 User ID  : {user_id or 'anonymous'}")

        # Validasi filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="File tidak ditemukan")

        # Validasi format
        if file.content_type not in ALLOWED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Format tidak didukung: '{file.content_type}'. "
                    "Gunakan JPG, PNG, atau WEBP."
                )
            )

        # Baca file + validasi ukuran
        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File gambar kosong")

        if len(contents) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Ukuran file terlalu besar "
                    f"({len(contents)/1024/1024:.1f} MB). "
                    f"Maksimal {MAX_FILE_SIZE_MB} MB."
                )
            )

        # Prediksi
        result = await _predict_from_bytes(contents)

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])

        # Low confidence — tidak disimpan ke DB
        if result["status"] == "low_confidence":
            return {
                "status" : "low_confidence",
                "message": result["message"],
                "data"   : result.get("data", {})
            }

        # ── Success ──────────────────────────────────────────
        prediction_data = result["data"]

        # Upload thumbnail ke Supabase Storage
        image_url = await _upload_thumbnail(
            contents=contents,
            user_id=user_id,
        )

        # Simpan ke Supabase DB
        try:
            db_payload = {
                "penyakit"      : prediction_data["penyakit"],
                "penyebab"      : prediction_data["penyebab"],
                "confidence"    : float(prediction_data["confidence"]),
                "gejala"        : prediction_data["gejala"],
                "penanggulangan": prediction_data["penanggulangan"],
                "raw_label"     : prediction_data["raw_label"],
                "filename"      : file.filename,
                "image_url"     : image_url,      # ← kolom baru
                "created_at"    : datetime.utcnow().isoformat(),
            }

            if user_id:
                db_payload["user_id"] = user_id

            supabase.table("predictions").insert(db_payload).execute()
            print(f"✅ Disimpan ke Supabase (user_id={user_id or 'anonymous'})")
            print(f"🖼️  image_url: {image_url or 'tidak ada'}")

        except Exception as db_error:
            print(f"⚠️  Gagal simpan ke Supabase: {db_error}")
            traceback.print_exc()

        return {
            "status" : "success",
            "message": "Prediksi berhasil",
            "data"   : {
                **prediction_data,
                "image_url": image_url,   # ← ikut dikembalikan ke Flutter
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"\n❌ ERROR ROUTE PREDICT\nDETAIL: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# HELPER: upload thumbnail ke Supabase Storage
# Mengembalikan public URL string, atau None jika gagal
# =========================================================

async def _upload_thumbnail(
    contents : bytes,
    user_id  : str | None,
) -> str | None:
    """
    Kompres gambar ke THUMBNAIL_SIZE lalu upload ke bucket.
    Path: {user_id}/{uuid}.jpg  atau  anonymous/{uuid}.jpg
    """
    try:
        from PIL import Image

        # Kompres & resize
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
        buf.seek(0)
        thumb_bytes = buf.read()

        original_kb  = len(contents) / 1024
        thumb_kb     = len(thumb_bytes) / 1024
        print(f"🗜️  Kompres: {original_kb:.0f} KB → {thumb_kb:.0f} KB")

        # Path di Storage
        folder    = str(user_id) if user_id else "anonymous"
        file_name = f"{uuid.uuid4().hex}.jpg"
        file_path = f"{folder}/{file_name}"

        # Upload
        supabase.storage.from_(STORAGE_BUCKET).upload(
            path         = file_path,
            file         = thumb_bytes,
            file_options = {"content-type": "image/jpeg"},
        )

        # Ambil public URL
        public_url = (
            supabase.storage
            .from_(STORAGE_BUCKET)
            .get_public_url(file_path)
        )

        print(f"✅ Thumbnail terupload: {file_path}")
        return public_url

    except Exception as e:
        # Upload gagal → prediksi tetap jalan, image_url = None
        print(f"⚠️  Gagal upload thumbnail: {e}")
        traceback.print_exc()
        return None


# =========================================================
# HELPER: prediksi dari bytes
# =========================================================

async def _predict_from_bytes(contents: bytes) -> dict:
    from io import BytesIO
    import numpy as np
    from PIL import Image
    from app.services.disease_services import (
        model, CLASS_NAMES, DATABASE_PENYAKIT, CONFIDENCE_THRESHOLD
    )

    if model is None:
        return {"status": "error", "message": "Model belum dimuat. Periksa log server."}

    if not CLASS_NAMES:
        return {"status": "error", "message": "class_names.json belum dimuat."}

    try:
        image     = Image.open(BytesIO(contents)).convert("RGB")
        image     = image.resize((224, 224))
        img_array = np.array(image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        probs      = prediction[0]
        idx        = int(np.argmax(probs))
        confidence = float(probs[idx])
        label_key  = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"

        all_probs = {
            CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "status" : "low_confidence",
                "message": (
                    f"Model tidak cukup yakin "
                    f"(confidence: {round(confidence * 100, 2)}%). "
                    "Coba upload gambar yang lebih jelas."
                ),
                "data": {
                    "penyakit"  : label_key,
                    "confidence": round(confidence * 100, 2),
                    "raw_label" : label_key,
                    "all_probs" : all_probs,
                }
            }

        info = DATABASE_PENYAKIT.get(label_key, {
            "nama"           : label_key,
            "penyebab"       : "Tidak diketahui",
            "gejala"         : [],
            "penanggulangan" : []
        })

        return {
            "status": "success",
            "data"  : {
                "penyakit"      : info["nama"],
                "penyebab"      : info["penyebab"],
                "confidence"    : round(confidence * 100, 2),
                "gejala"        : info["gejala"],
                "penanggulangan": info["penanggulangan"],
                "raw_label"     : label_key,
                "all_probs"     : all_probs,
            }
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}