# =========================================================
# [FIX 1] os.environ HARUS sebelum semua import TF/Keras
# =========================================================
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import io
import json
import zipfile
import traceback

import numpy as np
import tensorflow as tf

# [FIX 2] Pilih salah satu: tf.keras ATAU keras standalone
#         Jangan import keduanya bersamaan → rawan konflik
#         Karena training pakai tf.keras, service juga pakai tf.keras
from tensorflow import keras

from PIL import Image
from fastapi import UploadFile

# =========================================================
# ENABLE UNSAFE DESERIALIZATION
# (diperlukan untuk load model dengan augmentation layer)
# =========================================================

try:
    keras.config.enable_unsafe_deserialization()
    print("✅ Unsafe deserialization enabled")
except AttributeError:
    # tf.keras versi lama tidak punya method ini → abaikan
    pass

# =========================================================
# [FIX 3] HAPUS custom class patch (FixedRandomFlip, dst.)
#
# Alasan: model.keras sudah menyimpan konfigurasi layer
# augmentasi secara lengkap. Patch custom class justru
# bisa menyebabkan mismatch saat deserialisasi karena
# class name-nya berbeda dari yang disimpan di model.
#
# Kalau muncul error "data_format" saat load model,
# gunakan custom_objects kosong + safe_mode=False
# =========================================================

# =========================================================
# CONFIDENCE THRESHOLD
# Prediksi di bawah nilai ini dianggap "tidak yakin"
# =========================================================

CONFIDENCE_THRESHOLD = 0.60   # 60%

# =========================================================
# DIRECTORY SETUP
# =========================================================

CURRENT_FILE = os.path.abspath(__file__)
SERVICE_DIR  = os.path.dirname(CURRENT_FILE)   # .../app/services
APP_DIR      = os.path.dirname(SERVICE_DIR)    # .../app
ROOT_DIR     = os.path.dirname(APP_DIR)        # .../  (root project)

MODEL_DIR = os.path.join(ROOT_DIR, "models")

# =========================================================
# MODEL FILE SEARCH
# Cari file model secara berurutan, pakai yang pertama ada
# =========================================================

POSSIBLE_MODELS = [
    "modeldeteksi_fixed.keras",
    "best_model.keras",
    "modeldeteksi.keras",
    "model.keras",
]

MODEL_PATH = None

for model_name in POSSIBLE_MODELS:
    candidate = os.path.join(MODEL_DIR, model_name)
    if os.path.isfile(candidate):
        MODEL_PATH = candidate
        break

JSON_PATH = os.path.join(MODEL_DIR, "class_names.json")

# =========================================================
# DEBUG INFO
# =========================================================

print("\n================================================")
print("🌶️  SISTEM DETEKSI PENYAKIT CABAI")
print("================================================")
print(f"📦 TensorFlow : {tf.__version__}")
print(f"📦 Keras      : {keras.__version__}")
print(f"📁 ROOT DIR   : {ROOT_DIR}")
print(f"📁 MODEL DIR  : {MODEL_DIR}")
print(f"📁 MODEL PATH : {MODEL_PATH or 'TIDAK DITEMUKAN'}")
print(f"📁 JSON PATH  : {JSON_PATH}")
print("================================================")

try:
    print("\n📂 Isi folder models:")
    if os.path.exists(MODEL_DIR):
        for file in os.listdir(MODEL_DIR):
            size_mb = os.path.getsize(os.path.join(MODEL_DIR, file)) / (1024 * 1024)
            print(f"   - {file}  ({size_mb:.1f} MB)")
    else:
        print("❌ Folder models tidak ditemukan")
except Exception as e:
    print(f"❌ Gagal membaca folder models: {e}")

print("================================================\n")

# =========================================================
# GLOBAL VARIABLES
# =========================================================

model       = None
CLASS_NAMES = []

# =========================================================
# VALIDATE .KERAS FILE
# =========================================================

def validate_keras_file(filepath):
    try:
        if not os.path.exists(filepath):
            return False, "File tidak ditemukan"
        if os.path.getsize(filepath) <= 0:
            return False, "Ukuran file 0 KB"
        if not zipfile.is_zipfile(filepath):
            return False, "Bukan format .keras valid (bukan zip)"
        return True, "VALID"
    except Exception as e:
        return False, str(e)

# =========================================================
# LOAD MODEL
# =========================================================

def load_model_init():

    global model, CLASS_NAMES

    # -----------------------------------------------------
    # CEK MODEL PATH
    # -----------------------------------------------------

    if MODEL_PATH is None:
        print("❌ ERROR: Tidak ada file model ditemukan di folder models/")
        print(f"   Cari: {POSSIBLE_MODELS}")
        return

    # -----------------------------------------------------
    # VALIDASI FILE
    # -----------------------------------------------------

    valid, message = validate_keras_file(MODEL_PATH)

    if not valid:
        print(f"\n❌ ERROR: FILE MODEL INVALID → {message}")
        print("💡 SOLUSI: Jalankan ulang training dan pastikan model.save() berhasil")
        return

    # -----------------------------------------------------
    # LOAD MODEL
    # [FIX 4] Tidak pakai custom_objects untuk augmentation
    #         layer — biarkan tf.keras handle sendiri
    # -----------------------------------------------------

    try:
        print(f"⏳ Memuat model dari: {MODEL_PATH}")

        model = keras.models.load_model(
            MODEL_PATH,
            compile=False,    # tidak perlu compile untuk inferensi
            safe_mode=False   # diperlukan untuk load augmentation layer
        )

        print("✅ Model berhasil dimuat!")
        print(f"   Input shape  : {model.input_shape}")
        print(f"   Output shape : {model.output_shape}")

    except Exception as e:
        print(f"\n❌ ERROR: Gagal load model\nDETAIL: {e}")
        traceback.print_exc()
        model = None
        return

    # -----------------------------------------------------
    # LOAD CLASS NAMES
    # -----------------------------------------------------

    try:
        if not os.path.exists(JSON_PATH):
            print(f"❌ ERROR: class_names.json tidak ditemukan di {JSON_PATH}")
            return

        with open(JSON_PATH, "r") as f:
            CLASS_NAMES = json.load(f)

        print(f"✅ Class Names dimuat: {len(CLASS_NAMES)} kelas")
        print(f"   {CLASS_NAMES}")

    except Exception as e:
        print(f"❌ ERROR: Gagal baca class_names.json\nDETAIL: {e}")
        traceback.print_exc()

# =========================================================
# AUTO LOAD SAAT MODULE DIIMPORT
# =========================================================

load_model_init()

# =========================================================
# DATABASE PENYAKIT
# =========================================================

DATABASE_PENYAKIT = {

    "Chilli __Whitefly": {
        "nama": "Whitefly (Kutu Kebul)",
        "penyebab": "Hama Bemisia tabaci",
        "gejala": [
            "Daun menguning",
            "Pertumbuhan terhambat",
            "Daun keriting ke atas"
        ],
        "penanggulangan": [
            "Gunakan insektisida berbahan aktif imidakloprid",
            "Pasang mulsa perak untuk mengusir hama",
            "Cabut dan musnahkan tanaman yang terinfeksi berat"
        ]
    },

    "Chilli __Yellowish": {
        "nama": "Yellowish (Kekuningan)",
        "penyebab": "Virus atau kekurangan nutrisi (N, Fe, Mg)",
        "gejala": [
            "Daun menguning merata (klorosis)",
            "Daun pucat dan layu",
            "Pertumbuhan lambat"
        ],
        "penanggulangan": [
            "Lakukan uji tanah dan perbaiki nutrisi",
            "Pemupukan NPK secara rutin",
            "Kontrol hama vektor (kutu, thrips)"
        ]
    },

    "Chilli__Anthracnos": {
        "nama": "Antraknosa (Patek)",
        "penyebab": "Jamur Colletotrichum capsici",
        "gejala": [
            "Bercak cokelat kehitaman pada buah",
            "Buah membusuk dan mengering",
            "Lesi cekung dengan tepung spora oranye"
        ],
        "penanggulangan": [
            "Semprot fungisida berbahan mankozeb atau klorotalonil",
            "Buang dan musnahkan buah terinfeksi",
            "Hindari kelembapan berlebih dengan jarak tanam cukup"
        ]
    },

    "Chilli__Damping_Off": {
        "nama": "Damping Off (Rebah Semai)",
        "penyebab": "Jamur tanah Pythium / Rhizoctonia / Fusarium",
        "gejala": [
            "Bibit rebah tiba-tiba",
            "Pangkal batang membusuk berwarna cokelat",
            "Bibit mati massal di persemaian"
        ],
        "penanggulangan": [
            "Gunakan media tanam steril",
            "Kurangi frekuensi penyiraman",
            "Aplikasikan fungisida Previcur N pada media semai"
        ]
    },

    "Chilli__Leaf_Curl_Virus": {
        "nama": "Leaf Curl Virus (Virus Keriting Daun)",
        "penyebab": "Chilli Leaf Curl Virus (ChiLCV) — ditularkan kutu kebul",
        "gejala": [
            "Daun menggulung / keriting ke atas",
            "Pertumbuhan tanaman kerdil",
            "Daun menebal dan berwarna pucat"
        ],
        "penanggulangan": [
            "Kendalikan populasi kutu kebul (vektor utama)",
            "Gunakan benih varietas tahan virus",
            "Cabut dan musnahkan tanaman bergejala parah"
        ]
    },

    "Chilli__Leaf_Spot": {
        "nama": "Leaf Spot (Bercak Daun)",
        "penyebab": "Jamur Cercospora capsici",
        "gejala": [
            "Bercak bulat abu-abu dengan tepi cokelat",
            "Daun menguning di sekitar bercak",
            "Daun gugur prematur"
        ],
        "penanggulangan": [
            "Semprot fungisida berbahan tembaga atau mankozeb",
            "Kurangi kelembapan dengan drainase baik",
            "Rotasi tanaman setiap musim"
        ]
    },

    "Chilli__Veinal_Mottle_Virus": {
        "nama": "Veinal Mottle Virus",
        "penyebab": "Pepper Veinal Mottle Virus (PVMV) — ditularkan kutu daun",
        "gejala": [
            "Mosaik kuning-hijau pada helai daun",
            "Tulang daun berwarna lebih gelap",
            "Daun bergelombang dan buah cacat"
        ],
        "penanggulangan": [
            "Cabut dan bakar tanaman yang terinfeksi",
            "Kendalikan kutu daun dengan insektisida sistemik",
            "Sanitasi lahan secara rutin"
        ]
    },

    "Chilli___healthy": {
        "nama": "Tanaman Sehat",
        "penyebab": "-",
        "gejala": [
            "Daun hijau segar dan mengkilap",
            "Tidak ada bercak atau perubahan warna abnormal",
            "Pertumbuhan tunas dan buah normal"
        ],
        "penanggulangan": [
            "Lanjutkan perawatan rutin (penyiraman, pemupukan)",
            "Lakukan pemantasan berkala untuk sirkulasi udara"
        ]
    }
}

# =========================================================
# PREPROCESS IMAGE
#
# [FIX 5] Normalisasi HARUS sama dengan saat training.
#         Training pakai Rescaling(1./127.5, offset=-1)
#         → output range: -1 sampai 1
#
#         Di sini kita kirim nilai 0–255 mentah ke model
#         karena layer Rescaling sudah ADA di dalam model.
#         Jangan normalisasi manual di sini.
# =========================================================

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))

        img_array = np.array(image, dtype=np.float32)
        # Range 0–255, TIDAK dinormalisasi manual
        # karena layer Rescaling di dalam model yang akan handle

        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
        return img_array

    except Exception as e:
        raise ValueError(f"Gagal preprocess gambar: {e}")

# =========================================================
# PREDICT DISEASE
# =========================================================

async def predict_disease(file: UploadFile) -> dict:

    # -----------------------------------------------------
    # CEK MODEL
    # -----------------------------------------------------

    if model is None:
        return {
            "status": "error",
            "message": "Model belum berhasil dimuat. Periksa log server."
        }

    if not CLASS_NAMES:
        return {
            "status": "error",
            "message": "class_names.json belum dimuat atau kosong."
        }

    try:
        # -------------------------------------------------
        # READ FILE
        # -------------------------------------------------

        contents = await file.read()

        if not contents:
            return {"status": "error", "message": "File gambar kosong"}

        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        img = preprocess_image(contents)

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        prediction  = model.predict(img, verbose=0)
        probs       = prediction[0]                    # shape: (n_classes,)
        idx         = int(np.argmax(probs))
        confidence  = float(probs[idx])

        # -------------------------------------------------
        # AMBIL LABEL
        # -------------------------------------------------

        label_key = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "Unknown"

        # -------------------------------------------------
        # [FIX 6] CONFIDENCE THRESHOLD
        # Jika confidence terlalu rendah → beri peringatan
        # -------------------------------------------------

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "status": "low_confidence",
                "message": (
                    f"Model tidak cukup yakin dengan prediksi ini "
                    f"(confidence: {round(confidence * 100, 2)}%). "
                    f"Coba upload gambar yang lebih jelas."
                ),
                "data": {
                    "penyakit"     : label_key,
                    "confidence"   : round(confidence * 100, 2),
                    "raw_label"    : label_key,
                    "all_probs"    : {
                        CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                        for i in range(len(CLASS_NAMES))
                    }
                }
            }

        # -------------------------------------------------
        # GET DISEASE INFO
        # -------------------------------------------------

        info = DATABASE_PENYAKIT.get(label_key, {
            "nama"           : label_key,
            "penyebab"       : "Tidak diketahui",
            "gejala"         : [],
            "penanggulangan" : []
        })

        # -------------------------------------------------
        # SUCCESS RESPONSE
        # -------------------------------------------------

        return {
            "status": "success",
            "data": {
                "penyakit"       : info["nama"],
                "penyebab"       : info["penyebab"],
                "confidence"     : round(confidence * 100, 2),
                "gejala"         : info["gejala"],
                "penanggulangan" : info["penanggulangan"],
                "raw_label"      : label_key,
                "all_probs"      : {
                    CLASS_NAMES[i]: round(float(probs[i]) * 100, 2)
                    for i in range(len(CLASS_NAMES))
                }
            }
        }

    except Exception as e:
        print(f"\n❌ ERROR SAAT PREDIKSI\nDETAIL: {e}")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e)
        }