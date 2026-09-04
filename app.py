"""
AI-Based Fuzzy Expert System Perikanan - Activity-Separated Flask Version

Modules:
1. Budidaya Rumput Laut
2. Penangkapan Ikan dengan Jaring
3. Penangkapan Ikan dengan Bagan Apung
4. Penangkapan Ikan dengan Alat Pancing
5. Pengolahan Hasil Perikanan

Open:
    http://127.0.0.1:5000

Install:
    python3 -m pip install -r requirements.txt

Run:
    python3 app.py
"""

# ============================================================
# PRELIMINARY RECALIBRATION NOTE
# ============================================================
# This version includes FGD-informed and field-data-informed calibration.
# Main changes:
# 1) More stable semantic membership thresholds for 0-100 variables.
# 2) Rule-priority weighting for seaweed, net fishing, and floating lift-net.
# 3) Random Forest is trained to predict positive recommendation/reward.
# 4) Fuzzy score and hybrid fuzzy-RF score are calculated for each evaluation.

from pathlib import Path
import random
import hashlib
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import skfuzzy as fuzz

class SimpleFuzz:
    @staticmethod
    def trimf(x, abc):
        a, b, c = abc
        x = np.asarray(x, dtype=float)
        y = np.zeros_like(x, dtype=float)

        if b != a:
            left = (x - a) / (b - a)
        else:
            left = np.zeros_like(x, dtype=float)

        if c != b:
            right = (c - x) / (c - b)
        else:
            right = np.zeros_like(x, dtype=float)

        y = np.maximum(np.minimum(left, right), 0)
        y[x == b] = 1
        return y

    @staticmethod
    def trapmf(x, abcd):
        a, b, c, d = abcd
        x = np.asarray(x, dtype=float)

        if b != a:
            left = (x - a) / (b - a)
        else:
            left = np.ones_like(x, dtype=float)

        if d != c:
            right = (d - x) / (d - c)
        else:
            right = np.ones_like(x, dtype=float)

        y = np.maximum(np.minimum(left, right), 0)
        y[(x >= b) & (x <= c)] = 1
        return y

    @staticmethod
    def interp_membership(x, xmf, xx):
        return float(np.interp(xx, x, xmf))

fuzz = SimpleFuzz()
#from skfuzzy import control as ctrl

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STATIC_DIR = BASE_DIR / "static"
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. ACTIVITY-SPECIFIC CONFIGURATION
# ============================================================

ACTIVITIES = {
    "seaweed": {
        "name": "Budidaya Rumput Laut",
        "description": "Evaluasi kelayakan budidaya, risiko penyakit, risiko cuaca, kualitas pengeringan, dan peluang pasar.",
        "features": {
            "seed_quality": {"label": "Kualitas Bibit", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "water_condition": {"label": "Kondisi Perairan", "help": "0 = sangat tidak sesuai, 100 = sangat sesuai", "kind": "quality"},
            "disease_risk": {"label": "Risiko Penyakit / Ice-Ice", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "risk"},
            "weather_risk": {"label": "Risiko Musim / Cuaca", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "risk"},
            "drying_condition": {"label": "Kondisi Pengeringan", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "market_price": {"label": "Harga Pasar", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
            "labor_availability": {"label": "Ketersediaan Tenaga Kerja", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
        },
        "risk_label": "Indeks Risiko Budidaya",
        "performance_label": "Indeks Kelayakan Budidaya",
    },
    "net_fishing": {
        "name": "Penangkapan Ikan dengan Jaring",
        "description": "Evaluasi kelayakan trip penangkapan jaring berdasarkan kondisi laut, potensi fishing ground, kondisi alat, biaya, dan dukungan operasi.",
        "features": {
            "sea_condition": {"label": "Kondisi Laut / Cuaca", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "fishing_ground_potential": {"label": "Potensi Daerah Penangkapan", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
            "net_condition": {"label": "Kondisi Jaring", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "fuel_cost": {"label": "Biaya BBM / Logistik", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "cost"},
            "crew_availability": {"label": "Ketersediaan ABK", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
            "market_demand": {"label": "Permintaan Pasar", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
            "ice_availability": {"label": "Ketersediaan Es", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
        },
        "risk_label": "Indeks Risiko Operasi Jaring",
        "performance_label": "Indeks Kelayakan Trip Jaring",
    },
    "floating_liftnet": {
        "name": "Penangkapan Ikan dengan Bagan Apung",
        "description": "Evaluasi kelayakan operasi bagan apung berdasarkan arus, gelombang, angin, kecerahan dan warna air, fase bulan, suhu, kedalaman, salinitas, dan cuaca.",
        "features": {
            "current_suitability": {"label": "Kesesuaian Arus", "help": "0 = sangat tidak sesuai, 100 = sangat sesuai (arus lemah-sedang stabil)", "kind": "quality"},
            "wave_condition": {"label": "Kondisi Gelombang", "help": "0 = sangat buruk, 100 = sangat baik (tenang-sedang)", "kind": "quality"},
            "wind_condition": {"label": "Kondisi Angin", "help": "0 = sangat buruk, 100 = sangat baik (lemah-sedang, stabil)", "kind": "quality"},
            "water_clarity": {"label": "Kecerahan Air", "help": "0 = sangat keruh, 100 = sesuai/baik untuk fishing with light", "kind": "quality"},
            "water_color": {"label": "Warna Air", "help": "0 = sangat tidak sesuai, 100 = sangat sesuai (hijau kebiruan lebih baik)", "kind": "quality"},
            "moon_phase_suitability": {"label": "Kesesuaian Fase Bulan", "help": "0 = sangat tidak sesuai, 100 = sangat sesuai (bulan gelap/bulan baru)", "kind": "quality"},
            "sst_suitability": {"label": "Kesesuaian Suhu Permukaan Laut", "help": "0 = sangat tidak sesuai, 100 = sangat sesuai/stabil", "kind": "quality"},
            "depth_condition": {"label": "Kesesuaian Kedalaman", "help": "0 = sangat tidak sesuai, 100 = sangat sesuai untuk jaring bagan", "kind": "quality"},
            "salinity_stability": {"label": "Stabilitas Salinitas", "help": "0 = sangat tidak stabil, 100 = sangat stabil", "kind": "quality"},
            "weather_condition": {"label": "Kondisi Cuaca", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
        },
        "risk_label": "Indeks Risiko Operasi Bagan Apung",
        "performance_label": "Indeks Kelayakan Operasi Bagan Apung",
    },
    "hook_fishing": {
        "name": "Penangkapan Ikan dengan Alat Pancing",
        "description": "Evaluasi kelayakan operasi penangkapan ikan dengan alat pancing berdasarkan kondisi laut, arus, tanda keberadaan ikan, umpan, alat pancing, waktu, biaya, dan pasar.",
        "features": {
            "sea_condition": {"label": "Kondisi Laut / Cuaca", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "current_suitability": {"label": "Kesesuaian Arus", "help": "0 = sangat tidak sesuai, 100 = sangat sesuai untuk operasi pancing", "kind": "quality"},
            "fish_presence_sign": {"label": "Tanda Keberadaan Ikan", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
            "bait_availability": {"label": "Ketersediaan / Kualitas Umpan", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "hook_line_condition": {"label": "Kondisi Alat Pancing", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "time_suitability": {"label": "Kesesuaian Waktu Penangkapan", "help": "0 = sangat tidak sesuai, 100 = sangat sesuai", "kind": "quality"},
            "fuel_cost": {"label": "Biaya BBM / Operasional", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "cost"},
            "market_demand": {"label": "Permintaan Pasar", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
        },
        "risk_label": "Indeks Risiko Operasi Pancing",
        "performance_label": "Indeks Kelayakan Trip Pancing",
    },
    "processing": {
        "name": "Pengolahan Hasil Perikanan",
        "description": "Evaluasi kualitas bahan baku, rantai dingin, higienitas, sortasi, kemasan, dan risiko kehilangan mutu.",
        "features": {
            "raw_freshness": {"label": "Kesegaran Bahan Baku", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "cold_chain_access": {"label": "Akses Es / Rantai Dingin", "help": "0 = sangat rendah, 100 = sangat tinggi", "kind": "level"},
            "hygiene_condition": {"label": "Kondisi Higienitas", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "sorting_quality": {"label": "Kualitas Sortasi / Grading", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "handling_delay": {"label": "Lama Penundaan Penanganan", "help": "0 = sangat singkat, 100 = sangat lama", "kind": "risk"},
            "packaging_condition": {"label": "Kondisi Kemasan", "help": "0 = sangat buruk, 100 = sangat baik", "kind": "quality"},
            "transport_distance": {"label": "Jarak Transportasi", "help": "0 = sangat dekat, 100 = sangat jauh", "kind": "risk"},
        },
        "risk_label": "Indeks Risiko Kehilangan Mutu",
        "performance_label": "Indeks Kualitas Pengolahan",
    },
}


# ============================================================
# 2. DATA GENERATION FOR ACTIVITY-SPECIFIC AI LAYERS
# ============================================================

def _valid_sample_file(path: Path, features):
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
        required = set(features) | {"outcome_high_risk", "outcome_good_performance", "reward"}
        return required.issubset(df.columns)
    except Exception:
        return False


def ensure_activity_sample_data(activity_key):
    """
    Creates separate sample data for each activity.
    Later, replace these files with real activity-specific field data.
    If an older sample file exists but does not match the current features,
    it will be regenerated automatically.
    """
    path = DATA_DIR / f"{activity_key}_sample_data.csv"
    cfg = ACTIVITIES[activity_key]
    features = list(cfg["features"].keys())

    if _valid_sample_file(path, features):
        return clean_activity_dataframe(pd.read_csv(path), activity_key) if "clean_activity_dataframe" in globals() else pd.read_csv(path)

    seed_map = {
        "seaweed": 101,
        "net_fishing": 202,
        "floating_liftnet": 303,
        "hook_fishing": 404,
        "processing": 505,
    }
    np.random.seed(seed_map[activity_key])
    n = 220
    df = pd.DataFrame({feature: np.random.randint(10, 100, size=n) for feature in features})

    if activity_key == "seaweed":
        risk_score = (
            0.35 * df["disease_risk"]
            + 0.30 * df["weather_risk"]
            + 0.20 * (100 - df["water_condition"])
            + 0.15 * (100 - df["drying_condition"])
        )
        performance_score = (
            0.25 * df["seed_quality"]
            + 0.25 * df["water_condition"]
            + 0.20 * df["drying_condition"]
            + 0.15 * df["market_price"]
            + 0.15 * df["labor_availability"]
        )

    elif activity_key == "net_fishing":
        risk_score = (
            0.35 * (100 - df["sea_condition"])
            + 0.25 * df["fuel_cost"]
            + 0.15 * (100 - df["net_condition"])
            + 0.15 * (100 - df["ice_availability"])
            + 0.10 * (100 - df["crew_availability"])
        )
        performance_score = (
            0.30 * df["fishing_ground_potential"]
            + 0.20 * df["sea_condition"]
            + 0.20 * df["net_condition"]
            + 0.15 * df["market_demand"]
            + 0.15 * (100 - df["fuel_cost"])
        )

    elif activity_key == "floating_liftnet":
        risk_score = (
            0.15 * (100 - df["current_suitability"])
            + 0.10 * (100 - df["wave_condition"])
            + 0.10 * (100 - df["wind_condition"])
            + 0.10 * (100 - df["water_clarity"])
            + 0.10 * (100 - df["moon_phase_suitability"])
            + 0.10 * (100 - df["weather_condition"])
            + 0.10 * (100 - df["salinity_stability"])
            + 0.10 * (100 - df["depth_condition"])
            + 0.15 * (100 - df["sst_suitability"])
        )
        performance_score = (
            0.15 * df["current_suitability"]
            + 0.10 * df["wave_condition"]
            + 0.10 * df["wind_condition"]
            + 0.10 * df["water_clarity"]
            + 0.10 * df["water_color"]
            + 0.15 * df["moon_phase_suitability"]
            + 0.10 * df["sst_suitability"]
            + 0.08 * df["depth_condition"]
            + 0.05 * df["salinity_stability"]
            + 0.07 * df["weather_condition"]
        )

    elif activity_key == "hook_fishing":
        risk_score = (
            0.25 * (100 - df["sea_condition"])
            + 0.15 * (100 - df["current_suitability"])
            + 0.15 * (100 - df["hook_line_condition"])
            + 0.15 * df["fuel_cost"]
            + 0.15 * (100 - df["bait_availability"])
            + 0.15 * (100 - df["time_suitability"])
        )
        performance_score = (
            0.20 * df["fish_presence_sign"]
            + 0.15 * df["bait_availability"]
            + 0.15 * df["hook_line_condition"]
            + 0.15 * df["time_suitability"]
            + 0.15 * df["sea_condition"]
            + 0.10 * df["current_suitability"]
            + 0.10 * df["market_demand"]
        )

    else:  # processing
        risk_score = (
            0.30 * (100 - df["raw_freshness"])
            + 0.25 * (100 - df["cold_chain_access"])
            + 0.20 * df["handling_delay"]
            + 0.15 * df["transport_distance"]
            + 0.10 * (100 - df["hygiene_condition"])
        )
        performance_score = (
            0.25 * df["raw_freshness"]
            + 0.20 * df["cold_chain_access"]
            + 0.20 * df["hygiene_condition"]
            + 0.15 * df["sorting_quality"]
            + 0.10 * df["packaging_condition"]
            + 0.10 * (100 - df["handling_delay"])
        )

    df["outcome_high_risk"] = (risk_score >= 65).astype(int)
    df["outcome_good_performance"] = (performance_score >= 65).astype(int)
    df["reward"] = np.where(df["outcome_high_risk"] == 1, -1, 1)
    df.to_csv(path, index=False)
    return df


# ============================================================
# 2A. PRELIMINARY CALIBRATION SETTINGS
# ============================================================

# Semantic membership thresholds avoid unstable quantile-based membership when
# several field variables have low variation or many repeated values.
# These are suitable for 0-100 questionnaire scores.
SEMANTIC_MEMBERSHIP = {
    "low": [0, 0, 25, 45],
    "medium": [30, 50, 70],
    "high": [55, 75, 100, 100],
}

# Hybrid weights were selected from the preliminary field-data calibration.
# Use module-specific weights when possible; use the pooled value as fallback.
HYBRID_WEIGHTS = {
    "seaweed": {"fuzzy": 0.582, "rf": 0.418},
    "net_fishing": {"fuzzy": 0.375, "rf": 0.625},
    "floating_liftnet": {"fuzzy": 0.635, "rf": 0.365},
    "pooled": {"fuzzy": 0.459, "rf": 0.541},
}


def clean_activity_dataframe(df, activity_key):
    """Ensure field data are numeric and safe for fuzzy/RF processing."""
    features = list(ACTIVITIES[activity_key]["features"].keys())
    df = df.copy()

    for feature in features:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        fill_value = df[feature].median()
        if pd.isna(fill_value):
            fill_value = 50
        df[feature] = df[feature].fillna(fill_value).clip(0, 100)

    if "outcome_high_risk" in df.columns:
        df["outcome_high_risk"] = pd.to_numeric(df["outcome_high_risk"], errors="coerce").fillna(0).astype(int)
    if "outcome_good_performance" in df.columns:
        df["outcome_good_performance"] = pd.to_numeric(df["outcome_good_performance"], errors="coerce").fillna(0).astype(int)
    if "reward" in df.columns:
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce").fillna(-1)

    return df


def build_adaptive_params(df, activity_key):
    """Build FGD-informed membership functions.

    Earlier versions used per-dataset quartiles. That can overfit small or
    low-variance survey data. This calibrated version uses stable semantic
    thresholds and retains quartiles as metadata for reporting.
    """
    df = clean_activity_dataframe(df, activity_key)
    params = {}
    for feature in ACTIVITIES[activity_key]["features"]:
        q25 = float(df[feature].quantile(0.25))
        q50 = float(df[feature].quantile(0.50))
        q75 = float(df[feature].quantile(0.75))
        params[feature] = {
            "low": SEMANTIC_MEMBERSHIP["low"],
            "medium": SEMANTIC_MEMBERSHIP["medium"],
            "high": SEMANTIC_MEMBERSHIP["high"],
            "q25": round(q25, 2),
            "q50": round(q50, 2),
            "q75": round(q75, 2),
            "calibration": "FGD-informed semantic thresholds",
        }
    return params


def train_activity_ml_model(df, activity_key):
    """Train RF to estimate positive recommendation probability.

    Target = 1 means positive/recommended decision. This aligns RF with the
    final hybrid score instead of predicting only high-risk status.
    """
    features = list(ACTIVITIES[activity_key]["features"].keys())
    df = clean_activity_dataframe(df, activity_key)
    X = df[features]

    if "reward" in df.columns:
        y = (df["reward"] > 0).astype(int)
    elif "outcome_good_performance" in df.columns:
        y = df["outcome_good_performance"].astype(int)
    elif "outcome_high_risk" in df.columns:
        y = (1 - df["outcome_high_risk"].astype(int)).astype(int)
    else:
        y = (df[features].mean(axis=1) >= 60).astype(int)

    if len(set(y)) < 2:
        y = (df[features].mean(axis=1) >= 60).astype(int)

    stratify_y = y if len(set(y)) > 1 and min(y.value_counts()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify_y
    )

    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = float(model.score(X_test, y_test))

    importance = pd.DataFrame({
        "feature": features,
        "label": [ACTIVITIES[activity_key]["features"][f]["label"] for f in features],
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return model, accuracy, importance


def discretize_state(activity_key, values):
    if activity_key == "seaweed":
        disease = "high_disease" if values["disease_risk"] >= 70 else "normal_disease"
        weather = "high_weather" if values["weather_risk"] >= 70 else "normal_weather"
        drying = "bad_drying" if values["drying_condition"] <= 40 else "ok_drying"
        return f"{disease}_{weather}_{drying}"

    if activity_key == "net_fishing":
        sea = "bad_sea" if values["sea_condition"] <= 40 else "ok_sea"
        fuel = "high_fuel" if values["fuel_cost"] >= 70 else "normal_fuel"
        ground = "low_ground" if values["fishing_ground_potential"] <= 40 else "ok_ground"
        return f"{sea}_{fuel}_{ground}"

    if activity_key == "floating_liftnet":
        current = "bad_current" if values["current_suitability"] <= 40 else "ok_current"
        moon = "bad_moon" if values["moon_phase_suitability"] <= 40 else "ok_moon"
        weather = "bad_weather" if values["weather_condition"] <= 40 else "ok_weather"
        return f"{current}_{moon}_{weather}"

    if activity_key == "hook_fishing":
        sea = "bad_sea" if values["sea_condition"] <= 40 else "ok_sea"
        fish = "low_fish" if values["fish_presence_sign"] <= 40 else "ok_fish"
        gear = "bad_gear" if values["hook_line_condition"] <= 40 else "ok_gear"
        return f"{sea}_{fish}_{gear}"

    fresh = "low_freshness" if values["raw_freshness"] <= 40 else "ok_freshness"
    cold = "low_coldchain" if values["cold_chain_access"] <= 40 else "ok_coldchain"
    delay = "long_delay" if values["handling_delay"] >= 70 else "normal_delay"
    return f"{fresh}_{cold}_{delay}"


def build_q_policy(df, activity_key):
    actions_by_activity = {
        "seaweed": ["lanjutkan_budidaya", "tunda_panen", "perbaiki_pengeringan", "kendalikan_penyakit", "jual_kolektif"],
        "net_fishing": ["berangkat_melaut", "tunda_trip", "perbaiki_jaring", "kurangi_jarak_trip", "jual_melalui_koperasi"],
        "floating_liftnet": ["operasikan_bagan", "tunda_operasi", "sesuaikan_waktu_bulan_gelap", "cek_stabilitas_bagan", "siapkan_es"],
        "hook_fishing": ["berangkat_memancing", "tunda_trip", "siapkan_umpan", "cek_alat_pancing", "pilih_jam_tangkap_terbaik"],
        "processing": ["lanjutkan_pengolahan", "gunakan_es", "sortasi_segera", "perbaiki_higienitas", "olah_lokal"],
    }

    actions = actions_by_activity[activity_key]
    q_table = {}
    alpha, gamma, epsilon = 0.2, 0.9, 0.2

    for _, row in df.iterrows():
        values = row.to_dict()
        state = discretize_state(activity_key, values)

        if state not in q_table:
            q_table[state] = {a: 0.0 for a in actions}

        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = max(q_table[state], key=q_table[state].get)

        reward = float(row["reward"])
        old = q_table[state][action]
        best_future = max(q_table[state].values())
        q_table[state][action] = old + alpha * (reward + gamma * best_future - old)

    return q_table


def suggest_action(q_policy, activity_key, values):
    state = discretize_state(activity_key, values)
    if state in q_policy:
        action_scores = q_policy[state]
        best = max(action_scores, key=action_scores.get)
        return state, best, round(action_scores[best], 3)

    fallback = {
        "seaweed": "pantau_penyakit_dan_pengeringan",
        "net_fishing": "cek_cuaca_dan_biaya_trip",
        "floating_liftnet": "cek_arus_bulan_dan_cuaca",
        "hook_fishing": "cek_laut_umpan_dan_alat_pancing",
        "processing": "prioritaskan_es_sortasi_higienitas",
    }
    return state, fallback[activity_key], 0.0


# ============================================================
# 3. FUZZY SYSTEM
# ============================================================

def add_lmh(variable, params):
    variable["low"] = fuzz.trapmf(variable.universe, params["low"])
    variable["medium"] = fuzz.trimf(variable.universe, params["medium"])
    variable["high"] = fuzz.trapmf(variable.universe, params["high"])


def category(score):
    if score < 40:
        return "rendah"
    if score < 70:
        return "sedang"
    return "tinggi"


def decision_category(score):
    """Operational decision category for 0-100 final suitability scores."""
    score = float(score)
    if score < 35:
        return "tidak layak / tunda"
    if score < 60:
        return "waspada / kurang ideal"
    if score < 80:
        return "layak"
    return "sangat layak"


def calculate_fuzzy_decision_score(fuzzy_result):
    """Combine fuzzy performance and inverse risk into one decision score."""
    performance = float(fuzzy_result["performance_score"])
    low_risk_component = 100 - float(fuzzy_result["risk_score"])
    return round(0.60 * performance + 0.40 * low_risk_component, 2)


def positive_probability(model, X_new):
    """Return probability for class 1 even when class order is not guaranteed."""
    proba = model.predict_proba(X_new)[0]
    classes = list(model.classes_)
    if 1 in classes:
        return float(proba[classes.index(1)])
    return float(proba[-1])


def calculate_hybrid_score(activity_key, fuzzy_score, rf_positive_probability):
    weights = HYBRID_WEIGHTS.get(activity_key, HYBRID_WEIGHTS["pooled"])
    rf_score = float(rf_positive_probability) * 100
    hybrid_score = weights["fuzzy"] * float(fuzzy_score) + weights["rf"] * rf_score
    return round(hybrid_score, 2), round(rf_score, 2), weights


class ActivityFuzzySystem:
    """Manual fuzzy inference system without scikit-fuzzy.control.

    This replacement keeps the same public interface as the old ctrl-based
    ActivityFuzzySystem, but computes rule activations directly using the
    SimpleFuzz membership functions above. It avoids importing skfuzzy/ctrl,
    which caused the Mac environment to hang/crash.
    """

    OUTPUT_CENTERS = {"low": 25.0, "medium": 50.0, "high": 80.0}

    def __init__(self, activity_key, adaptive_params):
        self.activity_key = activity_key
        self.cfg = ACTIVITIES[activity_key]
        self.params = adaptive_params

    def _degrees(self, feature, value):
        """Return low/medium/high membership degrees for one feature value."""
        x = np.arange(0, 101, 1)
        p = self.params[feature]
        return {
            "low": fuzz.interp_membership(x, fuzz.trapmf(x, p["low"]), value),
            "medium": fuzz.interp_membership(x, fuzz.trimf(x, p["medium"]), value),
            "high": fuzz.interp_membership(x, fuzz.trapmf(x, p["high"]), value),
        }

    @staticmethod
    def _and(*vals):
        return min(vals) if vals else 0.0

    @staticmethod
    def _or(*vals):
        return max(vals) if vals else 0.0

    def _score_from_activations(self, activations):
        """Defuzzify low/medium/high activations using weighted average."""
        total = sum(float(v) for v in activations.values())
        if total <= 0:
            return 50.0
        return sum(self.OUTPUT_CENTERS[k] * float(v) for k, v in activations.items()) / total

    def _add_activation(self, store, category_name, value, weight=1.0):
        # weight expresses FGD/data-informed rule priority while preserving
        # Mamdani-style max aggregation.
        weighted_value = float(value) * float(weight)
        store[category_name] = max(store.get(category_name, 0.0), weighted_value)

    def _infer(self, m):
        """Apply the same IF-THEN logic as the old ctrl.Rule block."""
        perf = {"low": 0.0, "medium": 0.0, "high": 0.0}
        risk = {"low": 0.0, "medium": 0.0, "high": 0.0}
        A = self._and
        O = self._or
        add = self._add_activation
        k = self.activity_key

        if k == "seaweed":
            # FGD/data-informed priority: disease and weather risks dominate,
            # while market/labor/drying are retained with lower rule priority.
            add(perf, "high", A(m["seed_quality"]["high"], m["water_condition"]["high"], m["disease_risk"]["low"]), weight=1.10)
            add(perf, "medium", A(m["seed_quality"]["medium"], m["water_condition"]["medium"]), weight=1.00)
            add(perf, "low", O(m["seed_quality"]["low"], m["water_condition"]["low"], m["disease_risk"]["high"]), weight=1.15)
            add(perf, "high", A(m["drying_condition"]["high"], m["market_price"]["high"]), weight=0.65)
            add(perf, "low", A(m["labor_availability"]["low"], m["weather_risk"]["high"]), weight=0.70)
            add(risk, "high", O(m["disease_risk"]["high"], m["weather_risk"]["high"]), weight=1.25)
            add(risk, "medium", O(m["disease_risk"]["medium"], m["weather_risk"]["medium"]), weight=1.10)
            add(risk, "low", A(m["disease_risk"]["low"], m["weather_risk"]["low"], m["water_condition"]["high"]), weight=1.15)
            add(risk, "high", A(m["drying_condition"]["low"], m["weather_risk"]["high"]), weight=0.75)
            add(perf, "medium", m["market_price"]["medium"], weight=0.50)

        elif k == "net_fishing":
            # FGD/data-informed priority: fishing ground potential, sea condition,
            # and crew availability receive stronger weights.
            add(perf, "high", A(m["sea_condition"]["high"], m["fishing_ground_potential"]["high"], m["net_condition"]["high"]), weight=1.25)
            add(perf, "medium", A(m["sea_condition"]["medium"], m["fishing_ground_potential"]["medium"]), weight=1.05)
            add(perf, "low", O(m["sea_condition"]["low"], m["net_condition"]["low"], m["fishing_ground_potential"]["low"]), weight=1.20)
            add(perf, "high", A(m["market_demand"]["high"], m["fishing_ground_potential"]["high"]), weight=0.70)
            add(perf, "low", A(m["fuel_cost"]["high"], m["market_demand"]["low"]), weight=0.75)
            add(risk, "high", O(m["sea_condition"]["low"], m["fuel_cost"]["high"]), weight=1.10)
            add(risk, "medium", O(m["sea_condition"]["medium"], m["fuel_cost"]["medium"]), weight=1.00)
            add(risk, "low", A(m["sea_condition"]["high"], m["fuel_cost"]["low"], m["ice_availability"]["high"]), weight=1.00)
            add(risk, "high", A(m["crew_availability"]["low"], m["sea_condition"]["low"]), weight=1.15)
            add(perf, "medium", m["fishing_ground_potential"]["medium"], weight=1.10)

        elif k == "floating_liftnet":
            # FGD/data-informed priority: depth and moon phase are emphasized;
            # operational safety remains controlled by current/wave/wind/weather.
            add(perf, "high", A(m["depth_condition"]["high"], m["moon_phase_suitability"]["high"]), weight=1.30)
            add(perf, "high", A(m["depth_condition"]["high"], m["moon_phase_suitability"]["high"], m["water_color"]["high"]), weight=1.25)
            add(perf, "high", A(m["current_suitability"]["high"], m["wave_condition"]["high"], m["wind_condition"]["high"], m["moon_phase_suitability"]["high"]), weight=1.00)
            add(perf, "high", A(m["water_clarity"]["high"], m["water_color"]["high"], m["weather_condition"]["high"]), weight=1.05)
            add(perf, "high", A(m["sst_suitability"]["high"], m["depth_condition"]["high"], m["salinity_stability"]["high"]), weight=0.70)
            add(perf, "medium", A(m["current_suitability"]["medium"], m["wave_condition"]["medium"], m["moon_phase_suitability"]["medium"]), weight=1.00)
            add(perf, "low", O(m["current_suitability"]["low"], m["wave_condition"]["low"], m["wind_condition"]["low"], m["weather_condition"]["low"]), weight=1.20)
            add(risk, "high", O(m["current_suitability"]["low"], m["wave_condition"]["low"], m["wind_condition"]["low"]), weight=1.20)
            add(risk, "high", A(m["moon_phase_suitability"]["low"], m["water_clarity"]["low"]), weight=0.95)
            add(risk, "medium", O(m["weather_condition"]["medium"], m["salinity_stability"]["medium"], m["sst_suitability"]["medium"]), weight=0.70)
            add(risk, "low", A(m["current_suitability"]["high"], m["wave_condition"]["high"], m["wind_condition"]["high"], m["weather_condition"]["high"]), weight=1.10)
            add(perf, "medium", A(m["depth_condition"]["medium"], m["water_color"]["medium"]), weight=1.10)

        elif k == "hook_fishing":
            add(perf, "high", A(m["sea_condition"]["high"], m["fish_presence_sign"]["high"], m["bait_availability"]["high"], m["hook_line_condition"]["high"]))
            add(perf, "high", A(m["time_suitability"]["high"], m["current_suitability"]["high"], m["market_demand"]["high"]))
            add(perf, "medium", A(m["sea_condition"]["medium"], m["fish_presence_sign"]["medium"], m["bait_availability"]["medium"]))
            add(perf, "low", O(m["sea_condition"]["low"], m["hook_line_condition"]["low"], m["bait_availability"]["low"]))
            add(risk, "high", O(m["sea_condition"]["low"], m["fuel_cost"]["high"], m["hook_line_condition"]["low"]))
            add(risk, "medium", O(m["current_suitability"]["medium"], m["time_suitability"]["medium"]))
            add(risk, "low", A(m["sea_condition"]["high"], m["hook_line_condition"]["high"], m["fuel_cost"]["low"]))
            add(perf, "medium", m["fish_presence_sign"]["medium"])
            add(perf, "high", A(m["market_demand"]["high"], m["fish_presence_sign"]["high"]))
            add(risk, "medium", A(m["fuel_cost"]["medium"], m["fish_presence_sign"]["low"]))

        else:  # processing
            add(perf, "high", A(m["raw_freshness"]["high"], m["cold_chain_access"]["high"], m["hygiene_condition"]["high"]))
            add(perf, "medium", A(m["raw_freshness"]["medium"], m["hygiene_condition"]["medium"]))
            add(perf, "low", O(m["raw_freshness"]["low"], m["hygiene_condition"]["low"], m["sorting_quality"]["low"]))
            add(perf, "high", A(m["packaging_condition"]["high"], m["sorting_quality"]["high"]))
            add(perf, "low", A(m["cold_chain_access"]["low"], m["transport_distance"]["high"]))
            add(risk, "high", O(m["raw_freshness"]["low"], m["handling_delay"]["high"]))
            add(risk, "high", A(m["cold_chain_access"]["low"], m["transport_distance"]["high"]))
            add(risk, "medium", O(m["handling_delay"]["medium"], m["transport_distance"]["medium"]))
            add(risk, "low", A(m["raw_freshness"]["high"], m["cold_chain_access"]["high"], m["handling_delay"]["low"]))
            add(perf, "medium", m["raw_freshness"]["medium"])

        return risk, perf

    def evaluate(self, values):
        memberships = {}
        for feature in self.cfg["features"]:
            value = float(values[feature])
            if value < 0 or value > 100:
                raise ValueError(f"{self.cfg['features'][feature]['label']} harus berada antara 0 dan 100.")
            memberships[feature] = self._degrees(feature, value)

        risk_act, perf_act = self._infer(memberships)
        risk = self._score_from_activations(risk_act)
        performance = self._score_from_activations(perf_act)

        base_result = {
            "risk_score": round(risk, 2),
            "risk_category": category(risk),
            "performance_score": round(performance, 2),
            "performance_category": category(performance),
            "recommendation": self.recommend(values, risk, performance),
        }
        fuzzy_score = calculate_fuzzy_decision_score(base_result)
        base_result["fuzzy_score"] = fuzzy_score
        base_result["fuzzy_category"] = decision_category(fuzzy_score)
        return base_result

    def recommend(self, values, risk, performance):
        if self.activity_key == "seaweed":
            if risk >= 70:
                return "Risiko budidaya rumput laut tinggi. Prioritaskan pengendalian penyakit, pantau kondisi perairan, tunda panen/penanaman jika cuaca dan pengeringan buruk."
            if performance >= 70:
                return "Kelayakan budidaya tinggi. Aktivitas dapat dilanjutkan; jaga kualitas bibit, pengeringan, dan manfaatkan harga pasar yang baik."
            return "Kondisi budidaya sedang. Lanjutkan dengan pemantauan penyakit, cuaca, kualitas bibit, dan harga pasar."

        if self.activity_key == "net_fishing":
            if risk >= 70:
                return "Risiko penangkapan dengan jaring tinggi. Tunda trip bila laut buruk atau biaya BBM tinggi; cek jaring dan es sebelum berangkat."
            if performance >= 70:
                return "Kelayakan trip jaring tinggi. Penangkapan dapat dilakukan terutama bila potensi lokasi dan permintaan pasar tinggi."
            return "Kondisi trip jaring sedang. Pertimbangkan kondisi laut, biaya BBM, ABK, kondisi jaring, dan permintaan pasar."

        if self.activity_key == "floating_liftnet":
            if risk >= 70:
                return "Risiko operasi bagan apung tinggi. Hindari operasi saat arus, gelombang, angin, atau cuaca tidak mendukung. Utamakan bulan gelap, cuaca stabil, dan perairan yang cukup jernih."
            if performance >= 70:
                return "Kelayakan operasi bagan apung tinggi. Kondisi arus, gelombang, angin, fase bulan, dan kualitas perairan mendukung pengumpulan ikan oleh cahaya lampu."
            return "Kondisi operasi bagan apung sedang. Pantau arus lemah-sedang, gelombang/angin, fase bulan, kecerahan dan warna air, suhu, kedalaman, salinitas, serta cuaca sebelum operasi."

        if self.activity_key == "hook_fishing":
            if risk >= 70:
                return "Risiko operasi pancing tinggi. Tunda trip jika laut buruk, alat pancing belum siap, umpan kurang baik, atau biaya operasional terlalu tinggi."
            if performance >= 70:
                return "Kelayakan trip pancing tinggi. Operasi dapat dilanjutkan bila tanda keberadaan ikan kuat, umpan dan alat siap, waktu tangkap sesuai, dan kondisi laut mendukung."
            return "Kondisi trip pancing sedang. Pertimbangkan kondisi laut, arus, ketersediaan umpan, kesiapan alat, waktu penangkapan, dan permintaan pasar."

        if risk >= 70:
            return "Risiko kehilangan mutu tinggi. Segera gunakan es, lakukan sortasi, perbaiki higienitas, kurangi penundaan, dan perbaiki kemasan."
        if performance >= 70:
            return "Kualitas pengolahan tinggi. Produk dapat diarahkan ke pasar premium, pengolahan bernilai tambah, atau pembeli dengan standar lebih tinggi."
        return "Kondisi pengolahan sedang. Tingkatkan rantai dingin, sortasi, higienitas, kemasan, dan percepat penanganan."


# ============================================================
# 4. VISUALIZATION
# ============================================================

def _chart_exists(filename):
    return (STATIC_DIR / filename).exists()


def membership_plot(activity_key, params, selected_feature):
    x = np.arange(0, 101, 1)
    valid_features = list(ACTIVITIES[activity_key]["features"].keys())
    if selected_feature not in params or selected_feature not in valid_features:
        selected_feature = valid_features[0]
    p = params[selected_feature]

    param_digest = hashlib.md5(json.dumps(p, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    filename = f"membership_{activity_key}_{selected_feature}_{param_digest}.png"
    path = STATIC_DIR / filename
    if path.exists():
        return filename

    low = fuzz.trapmf(x, p["low"])
    med = fuzz.trimf(x, p["medium"])
    high = fuzz.trapmf(x, p["high"])

    label = ACTIVITIES[activity_key]["features"][selected_feature]["label"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, low, label="Rendah/Buruk")
    ax.plot(x, med, label="Sedang")
    ax.plot(x, high, label="Tinggi/Baik")
    ax.set_title(f"Adaptive Membership: {label}")
    ax.set_xlabel("Skor 0–100")
    ax.set_ylabel("Derajat Keanggotaan")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return filename


def result_plot(cfg, fuzzy_result, ml_probability):
    labels = [
        cfg["risk_label"],
        cfg["performance_label"],
        "Skor Fuzzy",
        "Probabilitas Rekomendasi Positif RF",
        "Skor Hybrid",
    ]
    values = [
        fuzzy_result["risk_score"],
        fuzzy_result["performance_score"],
        fuzzy_result.get("fuzzy_score", 50),
        round(ml_probability * 100, 2),
        fuzzy_result.get("hybrid_score", 50),
    ]
    digest = hashlib.md5(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    filename = f"result_chart_{digest}.png"
    path = STATIC_DIR / filename
    if path.exists():
        return filename

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.bar(labels, values)
    ax.set_title("Indikator AI-Based Fuzzy Expert System")
    ax.set_ylabel("Skor / Probabilitas (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return filename

def feature_importance_plot(activity_key, importance_df):
    imp_records = importance_df[["feature", "importance"]].round(8).to_dict("records")
    imp_digest = hashlib.md5(json.dumps(imp_records, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    filename = f"feature_importance_{activity_key}_{imp_digest}.png"
    path = STATIC_DIR / filename
    if path.exists():
        return filename

    df = importance_df.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.barh(df["label"], df["importance"])
    ax.set_title("Machine Learning Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return filename


# ============================================================
# 5. APP CACHE
# ============================================================

AI_CACHE = {}
for key in ACTIVITIES:
    df = ensure_activity_sample_data(key)
    params = build_adaptive_params(df, key)
    model, acc, imp = train_activity_ml_model(df, key)
    q = build_q_policy(df, key)
    AI_CACHE[key] = {
        "data": df,
        "params": params,
        "model": model,
        "accuracy": acc,
        "importance": imp,
        "q_policy": q,
        "fuzzy": ActivityFuzzySystem(key, params),
    }


# ============================================================
# 6. FLASK ROUTES
# ============================================================

app = Flask(__name__)


@app.route("/health")
def health():
    return "OK - Flask app is running"


@app.route("/", methods=["GET", "POST"])
def index():
    activity_key = request.form.get("activity_key", "seaweed")
    if activity_key not in ACTIVITIES:
        activity_key = "seaweed"

    cfg = ACTIVITIES[activity_key]
    cache = AI_CACHE[activity_key]
    features = list(cfg["features"].keys())

    form_values = {f: 50 for f in features}
    selected_feature = request.form.get("selected_feature", features[0])
    if selected_feature not in features:
        selected_feature = features[0]

    result = None
    ml_result = None
    rl_result = None
    error = None
    charts = {}

    charts["membership"] = membership_plot(activity_key, cache["params"], selected_feature)
    charts["importance"] = feature_importance_plot(activity_key, cache["importance"])

    if request.method == "POST":
        action = request.form.get("action", "evaluate")

        if action == "change_activity":
            activity_key = request.form.get("activity_key", "seaweed")
            if activity_key not in ACTIVITIES:
                activity_key = "seaweed"
            cfg = ACTIVITIES[activity_key]
            cache = AI_CACHE[activity_key]
            features = list(cfg["features"].keys())
            form_values = {f: 50 for f in features}
            selected_feature = features[0]
            charts["membership"] = membership_plot(activity_key, cache["params"], selected_feature)
            charts["importance"] = feature_importance_plot(activity_key, cache["importance"])
        else:
            try:
                form_values = {f: float(request.form.get(f, 50)) for f in features}
                selected_feature = request.form.get("selected_feature", features[0])
                if selected_feature not in features:
                    selected_feature = features[0]

                result = cache["fuzzy"].evaluate(form_values)

                X_new = pd.DataFrame([form_values])[features]
                ml_pred = int(cache["model"].predict(X_new)[0])
                ml_prob = positive_probability(cache["model"], X_new)
                hybrid_score, rf_score, hybrid_weights = calculate_hybrid_score(
                    activity_key, result["fuzzy_score"], ml_prob
                )
                result["rf_score"] = rf_score
                result["rf_category"] = decision_category(rf_score)
                result["hybrid_score"] = hybrid_score
                result["hybrid_category"] = decision_category(hybrid_score)
                result["hybrid_weight_fuzzy"] = hybrid_weights["fuzzy"]
                result["hybrid_weight_rf"] = hybrid_weights["rf"]

                state, action_suggestion, q_value = suggest_action(cache["q_policy"], activity_key, form_values)

                ml_result = {
                    "prediction": "direkomendasikan" if ml_pred == 1 else "tidak direkomendasikan",
                    "target": "positive_reward",
                    "probability": round(ml_prob, 3),
                    "probability_percent": round(ml_prob * 100, 2),
                    "rf_score": rf_score,
                    "rf_category": decision_category(rf_score),
                    "hybrid_score": hybrid_score,
                    "hybrid_category": decision_category(hybrid_score),
                    "hybrid_formula": f"{hybrid_weights['fuzzy']:.3f} × fuzzy + {hybrid_weights['rf']:.3f} × RF",
                    "accuracy": round(cache["accuracy"], 3),
                }

                rl_result = {
                    "state": state,
                    "action": action_suggestion,
                    "q_value": q_value,
                }

                charts["membership"] = membership_plot(activity_key, cache["params"], selected_feature)
                charts["importance"] = feature_importance_plot(activity_key, cache["importance"])
                charts["result"] = result_plot(cfg, result, ml_prob)

            except Exception as exc:
                error = f"Kesalahan sistem: {exc}"

    return render_template(
        "index.html",
        activities=ACTIVITIES,
        activity_key=activity_key,
        cfg=cfg,
        features=features,
        form_values=form_values,
        selected_feature=selected_feature,
        adaptive_params=cache["params"],
        result=result,
        ml_result=ml_result,
        rl_result=rl_result,
        error=error,
        charts=charts,
    )


# ============================================================
# 7. ICE-ICE EARLY WARNING FEATURE FOR SEAWEED
# ============================================================

ICE_ICE_INPUTS = {
    "warna_thallus": {
        "label": "Warna Thallus",
        "help": "0 = normal mengilap, 5 = pucat/kusam, 10 = putih bening/transparan",
    },
    "lendir": {
        "label": "Lendir pada Thallus",
        "help": "0 = tidak ada, 5 = sedikit, 10 = banyak/berbau",
    },
    "bercak": {
        "label": "Bercak/Bintik pada Thallus",
        "help": "0 = tidak ada, 5 = 1–5% rumpun, 10 = lebih dari 5–10% rumpun",
    },
    "kerapuhan": {
        "label": "Kerapuhan Thallus",
        "help": "0 = kuat, 5 = mulai rapuh, 10 = banyak putus/keropos",
    },
    "stres_salinitas": {
        "label": "Stres Salinitas",
        "help": "0 = stabil, 5 = turun setelah hujan, 10 = turun tajam/limpasan air tawar",
    },
    "arus_lemah": {
        "label": "Kelemahan Arus",
        "help": "0 = arus baik, 5 = sedang, 10 = sangat lemah/kotoran mudah menempel",
    },
    "epifit_lumpur": {
        "label": "Epifit/Lumpur Menempel",
        "help": "0 = bersih, 5 = mulai tampak, 10 = banyak menempel",
    },
    "pertumbuhan_lambat": {
        "label": "Pertumbuhan Melambat",
        "help": "0 = normal, 5 = melambat, 10 = stagnan/menurun",
    },
}


def _trapmf_value(x, abcd):
    a, b, c, d = [float(v) for v in abcd]
    x = float(x)
    if b <= x <= c:
        return 1.0
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    if c < x < d:
        return (d - x) / (d - c) if d != c else 1.0
    return 0.0


def _trimf_value(x, abc):
    a, b, c = [float(v) for v in abc]
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 1.0
    if b < x < c:
        return (c - x) / (c - b) if c != b else 1.0
    return 0.0


def ice_ice_membership(value):
    """Membership for 0-10 field scores."""
    return {
        "rendah": _trapmf_value(value, [0, 0, 2, 4]),
        "sedang": _trimf_value(value, [2, 5, 8]),
        "tinggi": _trapmf_value(value, [6, 8, 10, 10]),
    }


def evaluate_ice_ice(values, symptom_days):
    """
    Fuzzy + IF-THEN rule base for early detection of ice-ice.
    Inputs are 0-10. Output is 0-100 risk score plus practical actions.
    """
    m = {k: ice_ice_membership(v) for k, v in values.items()}

    rules = []
    add = rules.append

    # Aman: visual thallus normal and environment stable.
    add((min(m["warna_thallus"]["rendah"], m["lendir"]["rendah"], m["bercak"]["rendah"],
             m["stres_salinitas"]["rendah"], m["arus_lemah"]["rendah"]), "rendah"))

    # Waspada: early visual change but environmental stress is not yet severe.
    add((min(m["warna_thallus"]["sedang"], max(m["lendir"]["sedang"], m["bercak"]["sedang"])), "sedang"))
    add((min(m["pertumbuhan_lambat"]["sedang"], max(m["warna_thallus"]["sedang"], m["bercak"]["sedang"])), "sedang"))
    add((max(m["stres_salinitas"]["sedang"], m["arus_lemah"]["sedang"], m["epifit_lumpur"]["sedang"]), "sedang"))

    # Risiko tinggi: symptoms plus one environmental trigger.
    add((min(m["warna_thallus"]["sedang"], m["lendir"]["sedang"],
             max(m["stres_salinitas"]["tinggi"], m["arus_lemah"]["tinggi"])), "tinggi"))
    add((min(m["bercak"]["tinggi"], max(m["kerapuhan"]["sedang"], m["lendir"]["sedang"])), "tinggi"))
    add((min(m["warna_thallus"]["tinggi"], max(m["lendir"]["sedang"], m["bercak"]["sedang"])), "tinggi"))
    add((min(m["epifit_lumpur"]["tinggi"], m["arus_lemah"]["tinggi"]), "tinggi"))

    # Sangat tinggi: strong early-warning combination or serious tissue damage.
    add((min(m["warna_thallus"]["tinggi"], m["lendir"]["tinggi"]), "sangat_tinggi"))
    add((min(m["bercak"]["tinggi"], m["kerapuhan"]["tinggi"]), "sangat_tinggi"))
    add((min(m["warna_thallus"]["sedang"], m["lendir"]["tinggi"],
             max(m["stres_salinitas"]["tinggi"], m["arus_lemah"]["tinggi"])), "sangat_tinggi"))
    add((min(m["warna_thallus"]["tinggi"], m["bercak"]["tinggi"], m["kerapuhan"]["tinggi"]), "sangat_tinggi"))

    # Temporal rule: repeated symptoms over 2-3 days increase risk.
    if symptom_days >= 2:
        repeated_strength = min(
            max(m["warna_thallus"]["sedang"], m["warna_thallus"]["tinggi"]),
            max(m["lendir"]["sedang"], m["lendir"]["tinggi"]),
            max(m["stres_salinitas"]["sedang"], m["stres_salinitas"]["tinggi"], m["arus_lemah"]["tinggi"]),
        )
        add((repeated_strength, "sangat_tinggi" if symptom_days >= 3 else "tinggi"))

    output_center = {
        "rendah": 20,
        "sedang": 50,
        "tinggi": 75,
        "sangat_tinggi": 92,
    }

    active_rules = [(strength, label) for strength, label in rules if strength > 0]
    if not active_rules:
        score = 50.0
    else:
        numerator = sum(strength * output_center[label] for strength, label in active_rules)
        denominator = sum(strength for strength, _ in active_rules)
        score = numerator / denominator if denominator else 50.0

    if score < 35:
        status = "Aman"
        category_label = "rendah"
    elif score < 60:
        status = "Waspada"
        category_label = "sedang"
    elif score < 80:
        status = "Pra-serangan ice-ice"
        category_label = "tinggi"
    else:
        status = "Serangan tinggi / tindakan segera"
        category_label = "sangat tinggi"

    recommendations = ice_ice_recommendations(score, values, symptom_days)

    dominant_rules = sorted(active_rules, key=lambda item: item[0], reverse=True)[:5]
    dominant_rules = [
        {"strength": round(float(strength), 3), "output": label.replace("_", " ")}
        for strength, label in dominant_rules
    ]

    return {
        "score": round(score, 2),
        "status": status,
        "category": category_label,
        "recommendations": recommendations,
        "dominant_rules": dominant_rules,
    }


def ice_ice_recommendations(score, values, symptom_days):
    actions = []
    if score < 35:
        actions.append("Lanjutkan monitoring pagi hari pada 5–10 titik tali.")
        actions.append("Catat warna thallus, lendir, bercak, suhu, salinitas, arus, kejernihan, dan epifit/lumpur.")
    elif score < 60:
        actions.append("Tingkatkan pemeriksaan menjadi harian dan fokus pada ujung, pangkal ikatan, serta thallus muda.")
        actions.append("Bersihkan epifit dan lumpur yang mulai menempel agar thallus tidak makin stres.")
        actions.append("Amati ulang dalam 24 jam; jika warna makin pucat atau lendir bertambah, naikkan status menjadi pra-serangan.")
    elif score < 80:
        actions.append("Anggap kebun masuk fase pra-serangan ice-ice dan lakukan pencegahan segera.")
        actions.append("Kurangi kepadatan ikatan agar sirkulasi air membaik.")
        actions.append("Angkat dan buang bagian thallus yang pucat, berlendir, atau mulai rapuh.")
        actions.append("Bersihkan epifit/lumpur dan pindahkan tali ke area berarus lebih baik bila memungkinkan.")
    else:
        actions.append("Lakukan pemilahan segera; buang bagian putih/transparan, rapuh, atau keropos.")
        actions.append("Pisahkan rumpun sakit agar fragmen rapuh tidak menyebar ke tali lain.")
        actions.append("Pindahkan ke lokasi berarus lebih baik dan atur kedalaman tali sesuai cahaya/suhu.")
        actions.append("Pertimbangkan panen dini pada bagian yang masih layak bila serangan mulai menyebar.")

    if values.get("stres_salinitas", 0) >= 7:
        actions.append("Karena salinitas turun/berubah tajam, waspadai dampak hujan deras, banjir sungai, atau limpasan air tawar.")
    if values.get("arus_lemah", 0) >= 7:
        actions.append("Karena arus lemah, prioritaskan lokasi dengan pertukaran air lebih baik.")
    if values.get("epifit_lumpur", 0) >= 7:
        actions.append("Karena epifit/lumpur tinggi, lakukan pembersihan tali dan tanaman lebih intensif.")
    if symptom_days >= 2:
        actions.append("Karena gejala muncul minimal 2 hari berturut-turut, perlakukan kondisi ini sebagai peringatan dini, bukan gejala biasa.")
    return actions


def ice_ice_plot(result):
    path = STATIC_DIR / "ice_ice_risk.png"
    labels = ["Risiko Ice-Ice"]
    values = [result["score"]]
    plt.figure(figsize=(6, 3.5))
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("Skor Risiko 0–100")
    plt.title("Fuzzy Early Warning: Ice-Ice")
    plt.grid(axis="y", alpha=0.3)
    plt.text(0, values[0] + 2, str(values[0]), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return "ice_ice_risk.png"


@app.route("/ice-ice", methods=["GET", "POST"])
def ice_ice_page():
    form_values = {k: 5 for k in ICE_ICE_INPUTS}
    symptom_days = 1
    result = None
    error = None
    chart = None

    if request.method == "POST":
        try:
            form_values = {}
            for key in ICE_ICE_INPUTS:
                value = float(request.form.get(key, 5))
                if value < 0 or value > 10:
                    raise ValueError(f"{ICE_ICE_INPUTS[key]['label']} harus berada antara 0 dan 10.")
                form_values[key] = value
            symptom_days = int(request.form.get("symptom_days", 1))
            if symptom_days < 1:
                symptom_days = 1
            if symptom_days > 7:
                symptom_days = 7
            result = evaluate_ice_ice(form_values, symptom_days)
            chart = ice_ice_plot(result)
        except Exception as exc:
            error = f"Kesalahan sistem ice-ice: {exc}"

    return render_template(
        "ice_ice.html",
        inputs=ICE_ICE_INPUTS,
        form_values=form_values,
        symptom_days=symptom_days,
        result=result,
        error=error,
        chart=chart,
    )


# ============================================================
# 8. NET FISHING NATURAL CONDITION FEATURE
# ============================================================

NET_NATURE_INPUTS = {
    "arus": {
        "label": "Kondisi Arus",
        "help": "0 = sangat lemah, 5 = sedang dan stabil/ideal, 10 = sangat kuat/deras",
    },
    "gelombang_angin": {
        "label": "Gelombang dan Angin",
        "help": "0 = tenang/aman, 5 = sedang, 10 = gelombang besar atau angin kuat",
    },
    "kejernihan_air": {
        "label": "Warna/Kejernihan Air",
        "help": "0 = terlalu keruh/lumpur, 5 = agak keruh alami/ideal, 10 = terlalu jernih",
    },
    "perubahan_suhu": {
        "label": "Perubahan Suhu/Massa Air",
        "help": "0 = stabil/sesuai, 5 = ada batas massa air ringan, 10 = perubahan ekstrem/ikan berpindah",
    },
    "tanda_biologis": {
        "label": "Tanda Biologis Ikan",
        "help": "0 = tidak ada tanda, 5 = beberapa tanda, 10 = banyak burung/ikan kecil/riak/gerombolan",
    },
    "waktu_tangkap": {
        "label": "Waktu Penangkapan",
        "help": "0 = siang kurang ideal, 5 = pagi/sore, 10 = malam/pergantian pasang-surut ideal",
    },
    "fase_bulan": {
        "label": "Fase Bulan untuk Jaring",
        "help": "0 = bulan sangat terang, 5 = sedang, 10 = bulan gelap/lebih menguntungkan untuk jaring pasif",
    },
    "kontur_kedalaman": {
        "label": "Kedalaman dan Kontur Dasar",
        "help": "0 = datar/tidak ada struktur, 5 = cukup potensial, 10 = tepi karang/rumpon/muara/pertemuan arus",
    },
    "kesesuaian_jaring": {
        "label": "Kesesuaian Jenis Jaring dengan Lokasi",
        "help": "0 = tidak sesuai/risiko tersangkut tinggi, 5 = cukup sesuai, 10 = sangat sesuai dengan arus, dasar, dan target ikan",
    },
}

NET_TYPES = {
    "gillnet": "Jaring Insang / Gillnet",
    "purse_seine": "Jaring Lingkar / Purse Seine Sederhana",
    "bottom_net": "Jaring Dasar",
    "general": "Jaring Umum / Belum Ditentukan",
}


def _medium_ideal_membership(value):
    """For variables where middle value is best: arus and water clarity."""
    return {
        "rendah": _trapmf_value(value, [0, 0, 2, 4]),
        "ideal": _trimf_value(value, [3, 5, 7]),
        "tinggi": _trapmf_value(value, [6, 8, 10, 10]),
    }


def _good_high_membership(value):
    """For variables where higher value means better suitability."""
    return {
        "buruk": _trapmf_value(value, [0, 0, 2, 4]),
        "sedang": _trimf_value(value, [2, 5, 8]),
        "baik": _trapmf_value(value, [6, 8, 10, 10]),
    }


def _risk_high_membership(value):
    """For variables where higher value means higher operational/environmental risk."""
    return {
        "rendah": _trapmf_value(value, [0, 0, 2, 4]),
        "sedang": _trimf_value(value, [2, 5, 8]),
        "tinggi": _trapmf_value(value, [6, 8, 10, 10]),
    }


def evaluate_net_nature(values, net_type):
    """
    Fuzzy + IF-THEN rule base for suitability of natural conditions for net fishing.
    Inputs are 0-10. Output is 0-100 suitability score plus practical actions.
    """
    arus = _medium_ideal_membership(values["arus"])
    kejernihan = _medium_ideal_membership(values["kejernihan_air"])
    gelombang = _risk_high_membership(values["gelombang_angin"])
    suhu = _risk_high_membership(values["perubahan_suhu"])
    tanda = _good_high_membership(values["tanda_biologis"])
    waktu = _good_high_membership(values["waktu_tangkap"])
    bulan = _good_high_membership(values["fase_bulan"])
    kontur = _good_high_membership(values["kontur_kedalaman"])
    cocok_jaring = _good_high_membership(values["kesesuaian_jaring"])

    rules = []
    def add(strength, label, explanation):
        if strength > 0:
            rules.append((float(strength), label, explanation))

    # Arus is the strongest condition.
    add(arus["ideal"], "tinggi", "IF arus sedang dan stabil THEN kondisi penangkapan jaring tinggi.")
    add(max(arus["rendah"], arus["tinggi"]), "rendah", "IF arus terlalu lemah atau terlalu kuat THEN kondisi penangkapan jaring rendah.")

    # Safety/weather rules.
    add(gelombang["tinggi"], "sangat_rendah", "IF gelombang besar atau angin kuat THEN operasi jaring sebaiknya ditunda.")
    add(min(gelombang["rendah"], arus["ideal"]), "tinggi", "IF gelombang rendah dan arus ideal THEN operasi lebih aman.")

    # Visibility/turbidity rules.
    add(kejernihan["ideal"], "tinggi", "IF air agak keruh alami THEN jaring tidak terlalu terlihat dan kondisi lebih sesuai.")
    add(max(kejernihan["rendah"], kejernihan["tinggi"]), "sedang", "IF air terlalu jernih atau terlalu berlumpur THEN efektivitas jaring menurun.")

    # Temperature/mass water rules.
    add(suhu["tinggi"], "rendah", "IF perubahan suhu/massa air ekstrem THEN ikan cenderung berpindah dan risiko meningkat.")
    add(min(suhu["sedang"], tanda["baik"]), "tinggi", "IF ada batas massa air ringan dan tanda ikan kuat THEN lokasi potensial.")

    # Biological signs and productive time.
    add(tanda["baik"], "tinggi", "IF banyak burung, ikan kecil, riak, gelembung, atau gerombolan ikan THEN peluang tangkapan tinggi.")
    add(tanda["buruk"], "rendah", "IF tanda biologis tidak terlihat THEN potensi lokasi rendah.")
    add(waktu["baik"], "tinggi", "IF waktu malam/pagi/sore/pergantian pasang-surut THEN kondisi lebih produktif.")

    # Moon phase; stronger effect for passive/gillnet.
    if net_type == "gillnet":
        add(bulan["baik"], "tinggi", "IF jaring insang digunakan saat bulan gelap THEN ikan lebih sulit melihat jaring.")
        add(bulan["buruk"], "sedang", "IF jaring insang digunakan saat bulan terang THEN ikan lebih mudah melihat jaring.")
    else:
        add(bulan["baik"], "sedang", "IF bulan gelap THEN kondisi dapat mendukung beberapa operasi jaring pasif.")

    # Depth/structure and net suitability.
    add(kontur["baik"], "tinggi", "IF ada kontur, rumpon, tepi karang, muara, lamun, atau pertemuan arus THEN lokasi potensial.")
    add(cocok_jaring["baik"], "tinggi", "IF jenis jaring sesuai dengan arus, dasar, dan target ikan THEN kelayakan operasi meningkat.")
    add(cocok_jaring["buruk"], "rendah", "IF jenis jaring tidak sesuai lokasi THEN risiko jaring rusak, hilang, atau hasil rendah meningkat.")

    # Gear-specific rules.
    if net_type == "gillnet":
        add(min(arus["ideal"], kejernihan["ideal"], max(waktu["sedang"], waktu["baik"])), "sangat_tinggi", "IF gillnet + arus sedang + air agak keruh + waktu pagi/sore/malam THEN sangat sesuai.")
        add(max(arus["tinggi"], gelombang["tinggi"]), "rendah", "IF gillnet dipasang saat arus kuat/gelombang besar THEN jaring mudah hanyut, terlipat, atau hilang.")
    elif net_type == "purse_seine":
        add(min(tanda["baik"], gelombang["rendah"], cocok_jaring["baik"]), "sangat_tinggi", "IF purse seine + ikan bergerombol jelas + laut tenang THEN sangat sesuai.")
        add(tanda["buruk"], "rendah", "IF purse seine tetapi ikan menyebar/tidak ada tanda permukaan THEN kurang sesuai.")
    elif net_type == "bottom_net":
        add(min(kontur["sedang"], cocok_jaring["baik"], max(arus["ideal"], arus["rendah"])), "tinggi", "IF jaring dasar + dasar cukup bersih + arus tidak terlalu kuat THEN sesuai untuk target demersal.")
        add(min(cocok_jaring["buruk"], kontur["baik"]), "rendah", "IF jaring dasar di daerah banyak karang tajam/tidak sesuai THEN risiko robek/tersangkut tinggi.")

    output_center = {
        "sangat_rendah": 18,
        "rendah": 32,
        "sedang": 55,
        "tinggi": 76,
        "sangat_tinggi": 92,
    }

    if not rules:
        score = 50.0
    else:
        numerator = sum(strength * output_center[label] for strength, label, _ in rules)
        denominator = sum(strength for strength, _, _ in rules)
        score = numerator / denominator if denominator else 50.0

    # Safety override: strong waves/wind should reduce final suitability.
    if values["gelombang_angin"] >= 8:
        score = min(score, 45.0)
    if values["arus"] >= 8.5:
        score = min(score, 50.0)

    if score < 35:
        status = "Tidak cocok / sebaiknya tunda"
        category = "rendah"
    elif score < 60:
        status = "Kurang ideal / perlu hati-hati"
        category = "sedang"
    elif score < 80:
        status = "Cocok untuk operasi jaring"
        category = "tinggi"
    else:
        status = "Sangat cocok / peluang tinggi"
        category = "sangat tinggi"

    dominant_rules = sorted(rules, key=lambda item: item[0], reverse=True)[:6]
    dominant_rules = [
        {"strength": round(strength, 3), "output": label.replace("_", " "), "rule": explanation}
        for strength, label, explanation in dominant_rules
    ]

    return {
        "score": round(score, 2),
        "status": status,
        "category": category,
        "recommendations": net_nature_recommendations(score, values, net_type),
        "dominant_rules": dominant_rules,
        "net_type_label": NET_TYPES.get(net_type, NET_TYPES["general"]),
    }


def net_nature_recommendations(score, values, net_type):
    actions = []
    if score < 35:
        actions.append("Sebaiknya tunda operasi jaring sampai arus, gelombang, dan angin lebih aman.")
        actions.append("Prioritaskan keselamatan perahu dan alat; jangan memaksakan trip saat jaring sulit dikontrol.")
    elif score < 60:
        actions.append("Operasi masih mungkin, tetapi lakukan dengan hati-hati dan pilih lokasi yang lebih terlindung.")
        actions.append("Kurangi durasi pemasangan dan cek posisi jaring lebih sering agar tidak terlipat, hanyut, atau tersangkut.")
    elif score < 80:
        actions.append("Kondisi cukup cocok. Operasi dapat dilakukan dengan tetap memantau perubahan arus, angin, dan tanda ikan.")
        actions.append("Pasang jaring saat arus mulai bergerak, misalnya setelah pasang tertinggi atau setelah surut terendah.")
    else:
        actions.append("Kondisi sangat mendukung. Prioritaskan area dengan tanda biologis kuat dan kontur/pertemuan arus.")
        actions.append("Catat jam, lokasi GPS, fase bulan, arus, dan hasil tangkapan agar pola lokasi produktif dapat divalidasi.")

    if values.get("arus", 5) >= 8:
        actions.append("Arus sangat kuat: risiko jaring hanyut, terlipat, hilang, atau tersangkut meningkat.")
    elif values.get("arus", 5) <= 2:
        actions.append("Arus sangat lemah: ikan mungkin kurang bergerak mengikuti aliran; pertimbangkan waktu saat arus mulai berjalan.")
    if values.get("gelombang_angin", 0) >= 7:
        actions.append("Gelombang/angin tinggi: hindari operasi karena risiko keselamatan dan kontrol jaring menurun.")
    if values.get("kejernihan_air", 5) >= 8:
        actions.append("Air terlalu jernih: untuk gillnet, pertimbangkan operasi malam atau saat cahaya rendah.")
    if values.get("kejernihan_air", 5) <= 2:
        actions.append("Air terlalu keruh/lumpur: ikan dapat menyebar atau berpindah; cari batas warna air yang lebih stabil.")
    if values.get("tanda_biologis", 0) >= 7:
        actions.append("Tanda biologis kuat: fokus pada area burung menyambar, ikan kecil melompat, riak, pusaran, atau gerombolan permukaan.")

    if net_type == "gillnet":
        actions.append("Untuk jaring insang: kondisi terbaik biasanya arus sedang, air agak keruh, dan waktu malam/pagi/sore.")
    elif net_type == "purse_seine":
        actions.append("Untuk jaring lingkar: operasi paling sesuai saat gerombolan ikan terlihat jelas dan laut cukup tenang.")
    elif net_type == "bottom_net":
        actions.append("Untuk jaring dasar: hindari karang tajam atau dasar banyak penghalang agar jaring tidak robek/tersangkut.")
    return actions


def net_nature_plot(result):
    path = STATIC_DIR / "net_nature_suitability.png"
    labels = ["Kelayakan Kondisi Alam"]
    values = [result["score"]]
    plt.figure(figsize=(6, 3.5))
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("Skor Kelayakan 0–100")
    plt.title("Fuzzy Suitability: Penangkapan Jaring")
    plt.grid(axis="y", alpha=0.3)
    plt.text(0, values[0] + 2, str(values[0]), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return "net_nature_suitability.png"


@app.route("/net-nature", methods=["GET", "POST"])
def net_nature_page():
    form_values = {k: 5 for k in NET_NATURE_INPUTS}
    net_type = "gillnet"
    result = None
    error = None
    chart = None

    if request.method == "POST":
        try:
            form_values = {}
            for key in NET_NATURE_INPUTS:
                value = float(request.form.get(key, 5))
                if value < 0 or value > 10:
                    raise ValueError(f"{NET_NATURE_INPUTS[key]['label']} harus berada antara 0 dan 10.")
                form_values[key] = value
            net_type = request.form.get("net_type", "gillnet")
            if net_type not in NET_TYPES:
                net_type = "general"
            result = evaluate_net_nature(form_values, net_type)
            chart = net_nature_plot(result)
        except Exception as exc:
            error = f"Kesalahan sistem kondisi alam jaring: {exc}"

    return render_template(
        "net_nature.html",
        inputs=NET_NATURE_INPUTS,
        net_types=NET_TYPES,
        net_type=net_type,
        form_values=form_values,
        result=result,
        error=error,
        chart=chart,
    )



# ============================================================
# 9. FEATURE: KONDISI LAUT BAGAN APUNG
# ============================================================

LIFTNET_NATURE_INPUTS = {
    "arus": {
        "label": "Arus",
        "help": "0 = arus sangat kuat/tidak aman, 5 = sedang, 10 = lemah-sedang stabil/ideal untuk bagan apung",
    },
    "gelombang": {
        "label": "Gelombang",
        "help": "0 = gelombang tinggi, 5 = sedang, 10 = tenang-rendah dan aman",
    },
    "angin": {
        "label": "Angin",
        "help": "0 = kuat/berubah mendadak, 5 = sedang, 10 = lemah-sedang stabil",
    },
    "kecerahan_air": {
        "label": "Kecerahan Air",
        "help": "0 = sangat keruh, 5 = sedang, 10 = jernih/visibilitas mendukung penyebaran cahaya lampu",
    },
    "warna_air": {
        "label": "Warna Air",
        "help": "0 = cokelat-kuning/keruh, 5 = cukup baik, 10 = hijau kebiruan/stabil",
    },
    "fase_bulan": {
        "label": "Fase Bulan",
        "help": "0 = bulan terang, 5 = sedang, 10 = bulan gelap/bulan baru",
    },
    "suhu_permukaan": {
        "label": "Suhu Permukaan Laut",
        "help": "0 = ekstrem/tidak stabil, 5 = cukup sesuai, 10 = stabil dan sesuai ikan target, misalnya 27–30°C untuk banyak pelagis kecil tropis",
    },
    "kedalaman": {
        "label": "Kedalaman",
        "help": "0 = terlalu dangkal/jaring menyentuh dasar, 5 = cukup, 10 = sesuai ukuran jaring dan lokasi bagan",
    },
    "salinitas": {
        "label": "Salinitas",
        "help": "0 = tidak stabil/limpasan air tawar tinggi, 5 = cukup stabil, 10 = stabil",
    },
    "cuaca": {
        "label": "Cuaca dan Visibilitas",
        "help": "0 = hujan lebat/petir/visibilitas buruk, 5 = cukup aman, 10 = tidak hujan lebat, tidak petir, visibilitas baik",
    },
}


def evaluate_liftnet_nature(values):
    """Fuzzy + IF-THEN untuk identifikasi kondisi laut bagan apung berbasis skala 0-10."""
    m = {k: _good_high_membership(v) for k, v in values.items()}
    rules = []

    def add(strength, label, explanation):
        if strength > 0:
            rules.append((float(strength), label, explanation))

    add(m["arus"]["baik"], "tinggi", "IF arus lemah-sedang dan stabil THEN operasi bagan apung mendukung.")
    add(m["arus"]["buruk"], "rendah", "IF arus kuat THEN jaring miring, lampu kurang efektif, dan hauling lebih berat.")
    add(m["gelombang"]["baik"], "tinggi", "IF gelombang tenang-rendah THEN posisi bagan dan bukaan jaring lebih stabil.")
    add(m["gelombang"]["buruk"], "sangat_rendah", "IF gelombang tinggi THEN operasi bagan sebaiknya ditunda karena risiko keselamatan dan efektivitas turun.")
    add(m["angin"]["baik"], "tinggi", "IF angin lemah-sedang dan stabil THEN posisi lampu dan bagan lebih terkendali.")
    add(m["angin"]["buruk"], "rendah", "IF angin kuat/berubah mendadak THEN bagan bergeser dan keselamatan menurun.")
    add(m["kecerahan_air"]["baik"], "tinggi", "IF kecerahan air sedang-jernih THEN cahaya lampu menyebar lebih efektif.")
    add(m["kecerahan_air"]["buruk"], "rendah", "IF air sangat keruh THEN efektivitas cahaya lampu menurun.")
    add(m["warna_air"]["baik"], "tinggi", "IF warna air hijau kebiruan THEN perairan cenderung stabil dan cocok untuk fishing with light.")
    add(m["fase_bulan"]["baik"], "tinggi", "IF bulan gelap/bulan baru THEN cahaya lampu bagan lebih dominan untuk mengumpulkan ikan.")
    add(m["fase_bulan"]["buruk"], "sedang", "IF bulan terang THEN ikan dapat lebih menyebar dan lampu bagan kurang dominan.")
    add(m["suhu_permukaan"]["baik"], "tinggi", "IF suhu permukaan stabil dan sesuai ikan target THEN peluang agregasi ikan lebih baik.")
    add(m["suhu_permukaan"]["buruk"], "rendah", "IF suhu berubah ekstrem THEN ikan pelagis kecil dapat berpindah.")
    add(m["kedalaman"]["baik"], "tinggi", "IF kedalaman cukup dan jaring tidak menyentuh dasar THEN operasi bagan lebih aman dan efektif.")
    add(m["salinitas"]["baik"], "tinggi", "IF salinitas stabil THEN ikan tidak terdorong menyebar akibat limpasan air tawar.")
    add(m["salinitas"]["buruk"], "rendah", "IF salinitas berubah drastis setelah hujan besar/muara THEN ikan dapat menyebar atau pindah.")
    add(m["cuaca"]["baik"], "tinggi", "IF cuaca aman, tanpa petir/hujan lebat, dan visibilitas baik THEN operasi lebih layak.")
    add(m["cuaca"]["buruk"], "sangat_rendah", "IF hujan lebat, petir, atau visibilitas buruk THEN operasi harus ditunda demi keselamatan.")

    add(min(m["arus"]["baik"], m["gelombang"]["baik"], m["angin"]["baik"], m["cuaca"]["baik"]), "sangat_tinggi", "IF arus, gelombang, angin, dan cuaca aman THEN kelayakan operasi bagan sangat tinggi.")
    add(min(m["kecerahan_air"]["baik"], m["warna_air"]["baik"], m["fase_bulan"]["baik"]), "sangat_tinggi", "IF air cukup jernih, warna hijau kebiruan, dan bulan gelap THEN lampu bagan sangat efektif.")
    add(min(m["suhu_permukaan"]["baik"], m["salinitas"]["baik"], m["kedalaman"]["baik"]), "tinggi", "IF suhu, salinitas, dan kedalaman sesuai THEN habitat mendukung ikan target.")

    output_center = {
        "sangat_rendah": 18,
        "rendah": 32,
        "sedang": 55,
        "tinggi": 76,
        "sangat_tinggi": 92,
    }
    if not rules:
        score = 50.0
    else:
        denominator = sum(s for s, _, _ in rules)
        score = sum(s * output_center[label] for s, label, _ in rules) / denominator if denominator else 50.0

    # Safety overrides.
    if values["cuaca"] <= 2 or values["gelombang"] <= 2:
        score = min(score, 35.0)
    if values["arus"] <= 2 or values["angin"] <= 2:
        score = min(score, 45.0)

    if score < 35:
        status = "Tidak layak / tunda operasi"
        category = "rendah"
    elif score < 60:
        status = "Kurang ideal / operasi berisiko"
        category = "sedang"
    elif score < 80:
        status = "Layak untuk operasi bagan apung"
        category = "tinggi"
    else:
        status = "Sangat layak / kondisi mendukung"
        category = "sangat tinggi"

    dominant_rules = sorted(rules, key=lambda item: item[0], reverse=True)[:7]
    dominant_rules = [
        {"strength": round(strength, 3), "output": label.replace("_", " "), "rule": explanation}
        for strength, label, explanation in dominant_rules
    ]

    return {
        "score": round(score, 2),
        "status": status,
        "category": category,
        "recommendations": liftnet_nature_recommendations(score, values),
        "dominant_rules": dominant_rules,
    }


def liftnet_nature_recommendations(score, values):
    actions = []
    if score < 35:
        actions.append("Tunda operasi bagan apung. Utamakan keselamatan karena kondisi laut/cuaca tidak mendukung.")
        actions.append("Operasi dengan lampu kurang efektif bila arus kuat, gelombang tinggi, angin kuat, atau hujan/petir.")
    elif score < 60:
        actions.append("Operasi masih berisiko. Pilih waktu lebih aman dan pantau perubahan arus, angin, gelombang, dan cuaca.")
        actions.append("Kurangi durasi operasi dan pastikan posisi bagan, jangkar, lampu, serta jaring stabil.")
    elif score < 80:
        actions.append("Kondisi cukup layak. Operasikan bagan saat arus lemah-sedang, cuaca aman, dan cahaya bulan rendah.")
        actions.append("Catat fase bulan, suhu, kecerahan air, warna air, salinitas, dan hasil tangkapan untuk validasi pola lokal.")
    else:
        actions.append("Kondisi sangat mendukung. Prioritaskan operasi pada bulan gelap/bulan baru dan perairan hijau kebiruan yang cukup jernih.")
        actions.append("Pertahankan stabilitas posisi bagan dan pastikan lampu, hauling, jaring, serta penyimpanan hasil siap.")

    if values.get("arus", 5) <= 3:
        actions.append("Arus kurang sesuai/terlalu kuat: risiko jaring miring, lampu kurang efektif, dan hauling berat meningkat.")
    if values.get("gelombang", 5) <= 3:
        actions.append("Gelombang kurang aman: tunggu kondisi lebih tenang agar bagan dan jaring stabil.")
    if values.get("angin", 5) <= 3:
        actions.append("Angin kuat/berubah mendadak: risiko bagan bergeser dan posisi lampu terganggu.")
    if values.get("fase_bulan", 5) >= 7:
        actions.append("Fase bulan mendukung: cahaya lampu lebih dominan sehingga ikan lebih mudah terkonsentrasi.")
    if values.get("kecerahan_air", 5) <= 3:
        actions.append("Air sangat keruh: efektivitas penyebaran cahaya turun; pertimbangkan lokasi dengan transparansi lebih baik.")
    if values.get("cuaca", 5) <= 3:
        actions.append("Cuaca buruk/petir/hujan lebat: jangan operasi karena risiko keselamatan tinggi.")
    return actions


def liftnet_nature_plot(result):
    path = STATIC_DIR / "liftnet_nature_suitability.png"
    plt.figure(figsize=(6, 3.5))
    plt.bar(["Kelayakan Kondisi Laut Bagan"], [result["score"]])
    plt.ylim(0, 100)
    plt.ylabel("Skor Kelayakan 0–100")
    plt.title("Fuzzy Suitability: Bagan Apung")
    plt.grid(axis="y", alpha=0.3)
    plt.text(0, result["score"] + 2, str(result["score"]), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return "liftnet_nature_suitability.png"


@app.route("/liftnet-nature", methods=["GET", "POST"])
def liftnet_nature_page():
    form_values = {k: 5 for k in LIFTNET_NATURE_INPUTS}
    result = None
    error = None
    chart = None

    if request.method == "POST":
        try:
            form_values = {}
            for key in LIFTNET_NATURE_INPUTS:
                value = float(request.form.get(key, 5))
                if value < 0 or value > 10:
                    raise ValueError(f"{LIFTNET_NATURE_INPUTS[key]['label']} harus berada antara 0 dan 10.")
                form_values[key] = value
            result = evaluate_liftnet_nature(form_values)
            chart = liftnet_nature_plot(result)
        except Exception as exc:
            error = f"Kesalahan sistem kondisi laut bagan apung: {exc}"

    return render_template(
        "liftnet_nature.html",
        inputs=LIFTNET_NATURE_INPUTS,
        form_values=form_values,
        result=result,
        error=error,
        chart=chart,
    )


# ============================================================
# 10. FEATURE: KONDISI LAUT DAN TEKNIK PANCING
# ============================================================

HOOK_NATURE_INPUTS = {
    "arus": {"label": "Arus", "help": "0 = terlalu kuat/mati total, 5 = cukup, 10 = lemah-sedang/arus makan aktif"},
    "gelombang": {"label": "Gelombang", "help": "0 = tinggi/tidak aman, 5 = sedang, 10 = rendah-sedang dan aman"},
    "angin": {"label": "Angin", "help": "0 = kencang/tidak stabil, 5 = sedang, 10 = stabil dan tidak kencang"},
    "kejernihan_air": {"label": "Kecerahan Air", "help": "0 = sangat keruh, 5 = sedang, 10 = jernih-sedang sesuai predator visual"},
    "pasang_surut": {"label": "Pasang-Surut / Arus Makan", "help": "0 = air mati total, 5 = cukup, 10 = menjelang pasang/surut atau arus mulai jalan"},
    "suhu_permukaan": {"label": "Suhu Permukaan", "help": "0 = ekstrem/berubah mendadak, 5 = cukup stabil, 10 = stabil dan sesuai target"},
    "struktur_perairan": {"label": "Struktur Perairan", "help": "0 = tidak ada struktur, 5 = cukup potensial, 10 = karang/tubir/rumpon/muara/pertemuan arus"},
    "tanda_ikan": {"label": "Tanda Lokasi Potensial", "help": "0 = tidak ada tanda, 5 = beberapa tanda, 10 = burung makan/ikan kecil lompat/riak/rumpon/drop-off aktif"},
    "umpan": {"label": "Kesesuaian Umpan", "help": "0 = tidak sesuai target, 5 = cukup, 10 = sangat sesuai kebiasaan makan ikan target"},
    "waktu": {"label": "Waktu Memancing", "help": "0 = kurang ideal, 5 = cukup, 10 = pagi buta/sore/malam/saat arus mulai jalan"},
    "alat": {"label": "Kesesuaian Alat Pancing", "help": "0 = tidak sesuai/rusak, 5 = cukup, 10 = sesuai jenis pancing dan target ikan"},
}

HOOK_TYPES = {
    "bottom": "Pancing Dasar",
    "handline_pelagic": "Pancing Ulur Tuna/Cakalang/Tongkol",
    "trolling": "Trolling",
    "casting": "Casting",
    "squid_jigging": "Squid Jigging",
    "sabiki": "Pancing Kecil / Sabiki",
    "general": "Pancing Umum / Belum Ditentukan",
}

FISH_TARGETS = {
    "demersal": "Kakap, Kerapu, Lencam, Kurisi / Ikan Demersal",
    "large_pelagic": "Tuna, Cakalang, Tongkol, Tenggiri / Pelagis Besar",
    "predator_reef": "Kuwe, Barakuda, Talang-talang / Predator Karang-Pantai",
    "squid": "Cumi",
    "small_pelagic": "Ikan Pelagis Kecil",
    "general": "Target Umum / Belum Ditentukan",
}


def evaluate_hook_nature(values, hook_type, fish_target):
    """Fuzzy + IF-THEN untuk kelayakan kondisi pancing berbasis skala 0-10."""
    m = {k: _good_high_membership(v) for k, v in values.items()}
    rules = []

    def add(strength, label, explanation):
        if strength > 0:
            rules.append((float(strength), label, explanation))

    add(m["arus"]["baik"], "tinggi", "IF arus lemah-sedang dan arus makan aktif THEN umpan lebih stabil dan ikan lebih aktif.")
    add(m["arus"]["buruk"], "rendah", "IF arus terlalu kuat atau terlalu mati THEN umpan sulit berada di kedalaman target atau ikan kurang aktif.")
    add(m["gelombang"]["buruk"], "sangat_rendah", "IF gelombang tinggi THEN memancing dari perahu kecil tidak aman dan kontrol tali menurun.")
    add(m["angin"]["buruk"], "rendah", "IF angin kencang THEN perahu hanyut terlalu cepat dan posisi pancing sulit dikontrol.")
    add(m["kejernihan_air"]["baik"], "tinggi", "IF air jernih-sedang THEN predator visual lebih mudah melihat umpan.")
    add(m["pasang_surut"]["baik"], "tinggi", "IF menjelang pasang/surut atau arus mulai jalan THEN makanan bergerak dan ikan predator lebih aktif.")
    add(m["suhu_permukaan"]["baik"], "tinggi", "IF suhu stabil THEN ikan lebih mungkin tetap berada di lokasi/kolom air target.")
    add(m["struktur_perairan"]["baik"], "tinggi", "IF ada karang, tubir, rumpon, muara, atau pertemuan arus THEN peluang ikan berkumpul meningkat.")
    add(m["tanda_ikan"]["baik"], "tinggi", "IF burung menyambar, ikan kecil melompat, perbedaan warna air, buih, atau rumpon aktif THEN lokasi potensial.")
    add(m["umpan"]["baik"], "tinggi", "IF umpan sesuai ikan target THEN peluang sambaran meningkat.")
    add(m["alat"]["baik"], "tinggi", "IF alat pancing sesuai teknik dan target THEN operasi lebih efektif.")
    add(m["waktu"]["baik"], "tinggi", "IF pagi buta, sore, malam, atau arus mulai jalan THEN waktu memancing lebih produktif.")

    if hook_type == "bottom" or fish_target == "demersal":
        add(min(m["struktur_perairan"]["baik"], m["arus"]["baik"], m["umpan"]["baik"], m["alat"]["baik"]), "sangat_tinggi", "IF pancing dasar + struktur karang/tubir + arus lemah-sedang + umpan sesuai THEN sangat cocok untuk ikan demersal.")
        add(m["arus"]["buruk"], "rendah", "IF pancing dasar saat arus terlalu kuat THEN pemberat terseret dan umpan tidak tepat di dasar.")
    if hook_type == "handline_pelagic" or fish_target == "large_pelagic":
        add(min(m["struktur_perairan"]["baik"], m["tanda_ikan"]["baik"], m["umpan"]["baik"]), "sangat_tinggi", "IF pancing ulur pelagis + rumpon/drop-off/pertemuan arus + tanda ikan kuat + umpan sesuai THEN sangat potensial.")
    if hook_type == "trolling":
        add(min(m["tanda_ikan"]["baik"], m["kejernihan_air"]["baik"], m["struktur_perairan"]["baik"], m["waktu"]["baik"]), "sangat_tinggi", "IF trolling di jalur ikan kecil/rumpon/drop-off dengan tanda ikan kuat THEN sangat cocok.")
    if hook_type == "casting":
        add(min(m["arus"]["baik"], m["struktur_perairan"]["baik"], m["tanda_ikan"]["baik"]), "tinggi", "IF casting saat ada arus masuk/keluar, struktur pantai, dan tanda ikan kecil panik THEN cocok untuk predator.")
    if hook_type == "squid_jigging" or fish_target == "squid":
        add(min(m["kejernihan_air"]["baik"], m["waktu"]["baik"], m["alat"]["baik"]), "tinggi", "IF squid jigging pada air cukup jernih dan waktu malam/dekat lampu THEN peluang cumi meningkat.")
    if hook_type == "sabiki" or fish_target == "small_pelagic":
        add(min(m["struktur_perairan"]["baik"], m["tanda_ikan"]["baik"], m["alat"]["baik"]), "tinggi", "IF sabiki/pancing kecil di sekitar rumpon/lampu/perairan pantai dengan tanda pelagis kecil THEN cocok.")

    output_center = {"sangat_rendah": 18, "rendah": 32, "sedang": 55, "tinggi": 76, "sangat_tinggi": 92}
    if not rules:
        score = 50.0
    else:
        denominator = sum(s for s, _, _ in rules)
        score = sum(s * output_center[label] for s, label, _ in rules) / denominator if denominator else 50.0

    # Safety overrides.
    if values["gelombang"] <= 2 or values["angin"] <= 2:
        score = min(score, 40.0)
    if values["arus"] <= 2 and hook_type in {"bottom", "general"}:
        score = min(score, 50.0)

    if score < 35:
        status = "Tidak layak / tunda atau pindah lokasi"
        category = "rendah"
    elif score < 60:
        status = "Kurang ideal / perlu hati-hati"
        category = "sedang"
    elif score < 80:
        status = "Layak untuk memancing"
        category = "tinggi"
    else:
        status = "Sangat layak / peluang tinggi"
        category = "sangat tinggi"

    dominant_rules = sorted(rules, key=lambda item: item[0], reverse=True)[:7]
    dominant_rules = [
        {"strength": round(strength, 3), "output": label.replace("_", " "), "rule": explanation}
        for strength, label, explanation in dominant_rules
    ]

    return {
        "score": round(score, 2),
        "status": status,
        "category": category,
        "recommendations": hook_nature_recommendations(score, values, hook_type, fish_target),
        "dominant_rules": dominant_rules,
        "hook_type_label": HOOK_TYPES.get(hook_type, HOOK_TYPES["general"]),
        "fish_target_label": FISH_TARGETS.get(fish_target, FISH_TARGETS["general"]),
    }


def hook_nature_recommendations(score, values, hook_type, fish_target):
    actions = []
    if score < 35:
        actions.append("Tunda memancing atau pindah lokasi. Kondisi laut, tanda ikan, alat, umpan, atau waktu belum mendukung.")
        actions.append("Utamakan keselamatan bila gelombang tinggi, angin kencang, petir/hujan ekstrem, atau banyak sampah terapung.")
    elif score < 60:
        actions.append("Kondisi kurang ideal. Pilih lokasi dengan struktur lebih jelas, arus makan mulai jalan, dan tanda ikan lebih kuat.")
        actions.append("Sesuaikan kedalaman umpan, jenis umpan, dan teknik pancing dengan target ikan.")
    elif score < 80:
        actions.append("Kondisi layak. Fokus pada rumpon, tubir/drop-off, karang, muara, pertemuan arus, atau tanda burung/ikan kecil.")
        actions.append("Catat jam, lokasi, arus, umpan, teknik, dan hasil tangkapan agar pola lokal bisa divalidasi.")
    else:
        actions.append("Kondisi sangat mendukung. Prioritaskan area dengan struktur kuat dan rantai makanan aktif.")
        actions.append("Gunakan umpan/alat sesuai target ikan dan manfaatkan waktu pagi buta, sore, malam, atau saat arus mulai jalan.")

    if values.get("gelombang", 5) <= 3:
        actions.append("Gelombang tinggi: tunda terutama untuk perahu kecil.")
    if values.get("angin", 5) <= 3:
        actions.append("Angin kencang: perahu mudah hanyut dan posisi pancing sulit dikontrol.")
    if values.get("arus", 5) <= 3:
        actions.append("Arus tidak ideal: umpan bisa tidak stabil, sulit mencapai target, atau ikan kurang aktif.")
    if values.get("kejernihan_air", 5) <= 3:
        actions.append("Air sangat keruh setelah hujan besar: predator visual lebih sulit melihat umpan; cari perairan lebih stabil.")
    if values.get("tanda_ikan", 5) >= 7:
        actions.append("Tanda ikan kuat: perhatikan burung menyambar, ikan kecil melompat, riak, buih, atau batas warna air.")

    if hook_type == "bottom" or fish_target == "demersal":
        actions.append("Untuk pancing dasar: targetkan karang, pasir-karang, tubir, kapal karam, atau rumpon dasar; jaga umpan dekat dasar tetapi hindari terlalu masuk karang.")
    elif hook_type == "handline_pelagic" or fish_target == "large_pelagic":
        actions.append("Untuk pancing ulur pelagis: targetkan rumpon, drop-off, pertemuan arus, area burung makan, dan sesuaikan kedalaman umpan.")
    elif hook_type == "trolling":
        actions.append("Untuk trolling: susuri sisi rumpon, tubir karang, jalur ikan kecil, atau garis pertemuan warna air dengan kecepatan stabil.")
    elif hook_type == "casting":
        actions.append("Untuk casting: cari arus masuk/keluar, karang tepi, dermaga, muara, batuan, atau ikan kecil panik di permukaan.")
    elif hook_type == "squid_jigging" or fish_target == "squid":
        actions.append("Untuk cumi: pilih air cukup jernih, dekat lampu, lamun/karang dangkal, dan waktu malam.")
    return actions


def hook_nature_plot(result):
    path = STATIC_DIR / "hook_nature_suitability.png"
    plt.figure(figsize=(6, 3.5))
    plt.bar(["Kelayakan Kondisi Pancing"], [result["score"]])
    plt.ylim(0, 100)
    plt.ylabel("Skor Kelayakan 0–100")
    plt.title("Fuzzy Suitability: Alat Pancing")
    plt.grid(axis="y", alpha=0.3)
    plt.text(0, result["score"] + 2, str(result["score"]), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    return "hook_nature_suitability.png"


@app.route("/hook-nature", methods=["GET", "POST"])
def hook_nature_page():
    form_values = {k: 5 for k in HOOK_NATURE_INPUTS}
    hook_type = "general"
    fish_target = "general"
    result = None
    error = None
    chart = None

    if request.method == "POST":
        try:
            form_values = {}
            for key in HOOK_NATURE_INPUTS:
                value = float(request.form.get(key, 5))
                if value < 0 or value > 10:
                    raise ValueError(f"{HOOK_NATURE_INPUTS[key]['label']} harus berada antara 0 dan 10.")
                form_values[key] = value
            hook_type = request.form.get("hook_type", "general")
            fish_target = request.form.get("fish_target", "general")
            if hook_type not in HOOK_TYPES:
                hook_type = "general"
            if fish_target not in FISH_TARGETS:
                fish_target = "general"
            result = evaluate_hook_nature(form_values, hook_type, fish_target)
            chart = hook_nature_plot(result)
        except Exception as exc:
            error = f"Kesalahan sistem kondisi pancing: {exc}"

    return render_template(
        "hook_nature.html",
        inputs=HOOK_NATURE_INPUTS,
        hook_types=HOOK_TYPES,
        fish_targets=FISH_TARGETS,
        hook_type=hook_type,
        fish_target=fish_target,
        form_values=form_values,
        result=result,
        error=error,
        chart=chart,
    )


if __name__ == "__main__":
    app.run(debug=True)
