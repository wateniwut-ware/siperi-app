# SIPERI - Fixed No CTRL / No scikit-fuzzy

Paket ini sudah lengkap dalam satu folder:

- `app.py`
- `templates/`
- `static/`
- `data/`
- `requirements.txt`
- `run_app_mac.sh`

Versi ini tidak memakai `scikit-fuzzy` dan tidak memakai `ctrl`, sehingga menghindari error:

```text
name 'ctrl' is not defined
```

## Cara menjalankan di Mac

1. Ekstrak ZIP.
2. Buka Terminal.
3. Masuk ke folder hasil ekstrak, misalnya:

```bash
cd "/Users/wellemteniwut/Desktop/27923/PFR Inces dkk 2026/fuzzy_fishery/SIPERI_FIXED_NO_CTRL_READY"
```

4. Jalankan:

```bash
chmod +x run_app_mac.sh
./run_app_mac.sh
```

5. Buka browser:

```text
http://127.0.0.1:5000
```

## Cara manual

```bash
python3 -m venv venv
source venv/bin/activate
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt --prefer-binary --no-cache-dir
python -u app.py
```

## Catatan penting

Jangan menjalankan `pip install scikit-fuzzy`. Aplikasi ini sengaja dibuat tanpa `scikit-fuzzy` karena pada beberapa Mac paket tersebut menyebabkan hang atau crash.
