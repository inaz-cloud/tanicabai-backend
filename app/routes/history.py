from fastapi import APIRouter, HTTPException, Request, Query
from app.db.supabase_client import supabase
from app.utils.auth import get_user_id
import traceback

# =========================================================
# ROUTER  →  prefix: /history
# =========================================================

router = APIRouter()

# =========================================================
# GET /history/deteksi/
# Ambil riwayat prediksi penyakit milik user (terbaru di atas)
# =========================================================

@router.get("/deteksi/")
async def get_deteksi_history(
    request : Request,
    limit   : int = Query(default=50, ge=1, le=200),
):
    try:
        user_id = await get_user_id(request)

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Login diperlukan untuk melihat riwayat."
            )

        result = (
            supabase
            .table("predictions")
            .select(
                "id, penyakit, penyebab, confidence, "
                "raw_label, filename, image_url, created_at"
            )
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        rows = result.data or []
        print(f"✅ Riwayat deteksi: {len(rows)} item (user={user_id})")

        return {
            "status": "success",
            "data"  : rows,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# GET /history/harga/
# Ambil riwayat prediksi harga milik user (terbaru di atas)
# =========================================================

@router.get("/harga/")
async def get_harga_history(
    request : Request,
    limit   : int = Query(default=50, ge=1, le=200),
):
    try:
        user_id = await get_user_id(request)

        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Login diperlukan untuk melihat riwayat."
            )

        result = (
            supabase
            .table("price_predictions")
            .select(
                "id, wilayah, jenis_cabai, tanggal_target, "
                "harga_prediksi, harga_terakhir, tren, "
                "perubahan_persen, n_hari, created_at"
            )
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )

        rows = result.data or []
        print(f"✅ Riwayat harga: {len(rows)} item (user={user_id})")

        return {
            "status": "success",
            "data"  : rows,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))