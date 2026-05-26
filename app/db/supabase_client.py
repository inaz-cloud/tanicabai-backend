import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Muat file .env secara eksplisit
load_dotenv()

# Ambil data dan gunakan .strip() untuk membuang spasi/newline yang tidak sengaja terbaca
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()

# Bersihkan tanda kutip jika user tidak sengaja menulis SUPABASE_KEY="xxx" di .env
SUPABASE_URL = SUPABASE_URL.replace('"', '').replace("'", "")
SUPABASE_KEY = SUPABASE_KEY.replace('"', '').replace("'", "")

# ================= VALIDASI KREDENSIAL =================
if not SUPABASE_URL or not SUPABASE_KEY:
    print("\n" + "!"*50)
    print("❌ ERROR: SUPABASE_URL atau SUPABASE_KEY KOSONG!")
    print("Pastikan file .env sudah ada di root folder.")
    print("!"*50 + "\n")
    # Jangan raise ValueError di sini agar aplikasi tidak langsung mati sebelum log terbaca
    supabase = None
else:
    try:
        # Inisialisasi client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test koneksi sederhana (opsional, tapi bagus untuk debug)
        print(f"✅ INFO: Menghubungkan ke Supabase di: {SUPABASE_URL[:20]}...")
    except Exception as e:
        print("\n" + "!"*50)
        print(f"❌ ERROR: Gagal inisialisasi Supabase: {str(e)}")
        print("Pastikan SUPABASE_KEY yang digunakan adalah 'anon public' atau 'service_role'.")
        print("!"*50 + "\n")
        supabase = None