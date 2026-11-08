"""Konfigurasi bersama: jalur berkas, nama kolom, label, dan palet grafik."""
from pathlib import Path

APP_NAME = "GlycoTree"  # satu-satunya tempat nama aplikasi; tampil di semua halaman

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "pima-indians-diabetes.csv"
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
IMAGE_DIR = BASE_DIR / "static" / "img"

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
FEATURES = COLUMNS[:-1]
TARGET = "Outcome"
# Kolom yang nilai 0-nya mustahil secara medis, jadi 0 dianggap data hilang.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
CLASS_NAMES = ["Tidak diabetes", "Diabetes"]
LABELS = {
    "Pregnancies": "Jumlah kehamilan",
    "Glucose": "Glukosa",
    "BloodPressure": "Tekanan darah",
    "SkinThickness": "Tebal lipatan kulit",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Riwayat keluarga",
    "Age": "Usia",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
PRUNED_DEPTH = 3
DEPTHS_TO_COMPARE = [1, 2, 3, 4, 5, 6, 8, 10, None]
CV_FOLDS = 5

# Palet grafik. Harus selaras dengan token warna di static/style.css.
THEMES = {
    "light": {"bg": "#FFFFFF", "text": "#1B1740", "grid": "#DEDCF2", "muted": "#6F6C99",
              "negative": "#0C9A8C", "positive": "#E8336F", "accent": "#5B4BDB",
              "heat": ["#F7FBFF", "#6BAED6", "#08306B"]},
    "dark": {"bg": "#1A1736", "text": "#EEEDFF", "grid": "#332F5E", "muted": "#8D8ABB",
             "negative": "#3FD6C6", "positive": "#FF6B9D", "accent": "#8E80FF",
             "heat": ["#232A5C", "#3E73D6", "#A7CEFF"]},
}
