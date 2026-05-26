FROM python:3.10-slim

WORKDIR /app

# =========================================================
# SYSTEM DEPENDENCIES
# =========================================================
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    gfortran \
    liblapack-dev \
    libblas-dev \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# UPGRADE TOOLS
# =========================================================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =========================================================
# INSTALL DARI REQUIREMENTS SAJA
# Jangan install TF/numpy terpisah sebelum requirements
# agar tidak conflict versi
# =========================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================================================
# COPY SOURCE CODE
# (models/ dan data/ tidak di-copy, di-mount via volume)
# =========================================================
COPY . .

# =========================================================
# PORT & CMD
# =========================================================
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]