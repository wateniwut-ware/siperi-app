#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "== SIPERI Fixed No CTRL / No scikit-fuzzy =="
echo "Folder: $(pwd)"

if [ ! -d "venv" ]; then
  echo "Membuat virtual environment..."
  python3 -m venv venv
fi

source venv/bin/activate

python -m ensurepip --upgrade >/dev/null 2>&1 || true
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --prefer-binary --no-cache-dir

echo "Menjalankan aplikasi..."
echo "Buka: http://127.0.0.1:5000"
python -u app.py
