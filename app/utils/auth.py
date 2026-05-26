"""
app/utils/auth.py
=================
Helper untuk verifikasi Supabase JWT token di FastAPI.
Dipakai sebagai dependency injection di route yang butuh user_id.
"""

import os
from fastapi import Request
from app.db.supabase_client import supabase

# =========================================================
# GET USER ID DARI TOKEN
#
# Cara kerja:
# 1. Baca header Authorization: Bearer <token>
# 2. Verifikasi token ke Supabase
# 3. Return user_id kalau valid, None kalau tidak ada/invalid
#
# Tidak raise exception — kalau tidak ada token,
# riwayat disimpan tanpa user_id (anonymous)
# =========================================================

async def get_user_id(request: Request) -> str | None:
    """
    Ambil user_id dari JWT token Supabase.
    Return None kalau tidak ada token atau token invalid.
    """

    auth_header = request.headers.get("Authorization", "")

    if not auth_header.startswith("Bearer "):
        return None

    token = auth_header.replace("Bearer ", "").strip()

    if not token:
        return None

    try:
        # Verifikasi token ke Supabase → dapat user data
        response = supabase.auth.get_user(token)

        if response and response.user:
            return response.user.id

        return None

    except Exception as e:
        print(f"⚠️  Token verification failed: {e}")
        return None