import traceback

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, field_validator
from datetime import datetime

from app.services.price_services import predict_price
from app.db.supabase_client import supabase
from app.utils.auth import get_user_id

# =========================================================
# ROUTER
# =========================================================

router = APIRouter()

# =========================================================
# REQUEST MODEL
# =========================================================

class PriceRequest(BaseModel):
    wilayah        : str
    jenis_cabai    : str
    tanggal_target : str

    @field_validator("wilayah", "jenis_cabai", mode="before")
    @classmethod
    def tidak_boleh_kosong(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field tidak boleh kosong")
        return v.strip()

    @field_validator("tanggal_target", mode="before")
    @classmethod
    def validasi_tanggal(cls, v: str) -> str:
        try:
            datetime.strptime(v.strip(), "%Y-%m-%d")
        except ValueError:
            raise ValueError("Format tanggal harus YYYY-MM-DD")
        return v.strip()

# =========================================================
# PRICE ENDPOINT
# =========================================================

@router.post("/")
async def get_price_prediction(request: Request, data: PriceRequest):
    try:
        print("\n================================================")
        print("🌶️  REQUEST PREDIKSI / HISTORY HARGA CABAI")
        print(f"   Wilayah       : {data.wilayah}")
        print(f"   Jenis Cabai   : {data.jenis_cabai}")
        print(f"   Tanggal Target: {data.tanggal_target}")
        print("================================================")

        user_id = await get_user_id(request)
        print(f"👤 User ID  : {user_id or 'anonymous'}")

        result = predict_price(data)

        if result.get("status") == "error":
            code    = result.get("code", "INTERNAL_ERROR")
            message = result.get("message", "Terjadi kesalahan")

            status_map = {
                "MODEL_NOT_FOUND": 404,
                "DATA_ERROR"     : 422,
                "INTERNAL_ERROR" : 500,
            }

            raise HTTPException(
                status_code=status_map.get(code, 400),
                detail={"code": code, "message": message},
            )

        prediction_data = result["data"]
        mode            = result.get("mode", "forecast")

        # -------------------------------------------------
        # SIMPAN KE SUPABASE — MODE FORECAST
        # -------------------------------------------------
        if mode == "forecast":
            try:
                db_payload = {
                    "wilayah"         : prediction_data["wilayah"],
                    "jenis_cabai"     : prediction_data["jenis_cabai"],
                    "tanggal_target"  : prediction_data["tanggal_target"],
                    "tanggal_terakhir": prediction_data["tanggal_terakhir"],
                    "harga_prediksi"  : float(prediction_data["harga_prediksi"]),
                    "harga_terakhir"  : float(prediction_data["harga_terakhir"]),
                    "tren"            : prediction_data["tren"],
                    "perubahan_persen": float(prediction_data["perubahan_persen"]),
                    "n_hari"          : prediction_data["n_hari"],
                    "created_at"      : datetime.utcnow().isoformat(),
                }
                if user_id:
                    db_payload["user_id"] = user_id

                supabase.table("price_predictions").insert(db_payload).execute()
                print(f"✅ Forecast disimpan (user_id={user_id or 'anonymous'})")

            except Exception as db_error:
                print(f"⚠️  Gagal simpan forecast: {db_error}")
                traceback.print_exc()

        # -------------------------------------------------
        # SIMPAN KE SUPABASE — MODE HISTORY (hanya jika login)
        # Gunakan harga_aktual, dan isi kolom nullable dengan None
        # -------------------------------------------------
        elif mode == "history" and user_id:
            try:
                db_payload = {
                    "wilayah"         : prediction_data["wilayah"],
                    "jenis_cabai"     : prediction_data["jenis_cabai"],
                    "tanggal_target"  : prediction_data["tanggal_target"],
                    "tanggal_terakhir": None,
                    "harga_prediksi"  : float(prediction_data["harga_aktual"]),
                    "harga_terakhir"  : None,
                    "tren"            : "aktual",
                    "perubahan_persen": None,
                    "n_hari"          : 0,
                    "user_id"         : user_id,
                    "created_at"      : datetime.utcnow().isoformat(),
                }
                supabase.table("price_predictions").insert(db_payload).execute()
                print(f"✅ History disimpan (user_id={user_id})")

            except Exception as db_error:
                print(f"⚠️  Gagal simpan history: {db_error}")
                traceback.print_exc()

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------
        msg = (
            f"Prediksi {prediction_data.get('n_hari', '')} hari berhasil"
            if mode == "forecast"
            else "Data historis berhasil diambil"
        )

        return {
            "status" : "success",
            "mode"   : mode,
            "message": msg,
            "data"   : prediction_data,
        }

    except HTTPException:
        raise

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR\nDETAIL: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))