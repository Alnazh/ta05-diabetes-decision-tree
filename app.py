"""Aplikasi web Flask: prediksi risiko diabetes dengan Decision Tree."""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, render_template, request

import train_model
from tree_view import build_tree_view
from config import APP_NAME, CLASS_NAMES, FEATURES, LABELS, METRICS_PATH, MODEL_PATH

# Spesifikasi formulir. min/max adalah batas wajar untuk validasi keras.
# required=False berarti boleh dikosongkan; nilai kosong diisi median data latih.
FIELDS = [
    {"name": "Pregnancies", "label": "Jumlah kehamilan", "unit": "kali",
     "min": 0, "max": 20, "integer": True, "required": True, "group": "Riwayat",
     "hint": "Isi 0 jika belum pernah hamil.",
     "why": "Jumlah kehamilan yang tercatat di data. Menjadi salah satu faktor yang dipelajari model."},
    {"name": "Age", "label": "Usia", "unit": "tahun",
     "min": 10, "max": 100, "integer": True, "required": True, "group": "Riwayat",
     "hint": "Data latih mencakup usia 21 sampai 81 tahun.",
     "why": "Risiko diabetes cenderung naik seiring bertambahnya usia."},
    {"name": "DiabetesPedigreeFunction", "label": "Skor riwayat keluarga", "unit": "",
     "min": 0.05, "max": 2.5, "integer": False, "required": False, "group": "Riwayat",
     "hint": "Skor risiko keturunan dari dataset (0,08 sampai 2,42). Kosongkan jika tidak tahu.",
     "why": "Skor kecenderungan diabetes dalam keluarga. Makin tinggi, makin besar kecenderungannya."},
    {"name": "Glucose", "label": "Glukosa plasma", "unit": "mg/dL",
     "min": 40, "max": 300, "integer": False, "required": True, "group": "Pemeriksaan",
     "hint": "Kadar glukosa 2 jam setelah tes toleransi glukosa.",
     "why": "Kadar gula darah 2 jam setelah tes toleransi glukosa. Nilai tinggi adalah tanda utama diabetes."},
    {"name": "BMI", "label": "Indeks massa tubuh (BMI)", "unit": "kg/m²",
     "min": 10, "max": 70, "integer": False, "required": True, "group": "Pemeriksaan",
     "hint": "Berat (kg) dibagi tinggi (m) kuadrat.",
     "why": "Perbandingan berat dan tinggi badan. BMI tinggi berkaitan dengan risiko diabetes yang lebih besar."},
    {"name": "BloodPressure", "label": "Tekanan darah diastolik", "unit": "mmHg",
     "min": 30, "max": 200, "integer": False, "required": False, "group": "Pemeriksaan",
     "hint": "Kosongkan jika tidak diukur.",
     "why": "Tekanan darah diastolik. Tekanan darah tinggi sering menyertai diabetes."},
    {"name": "SkinThickness", "label": "Tebal lipatan kulit trisep", "unit": "mm",
     "min": 5, "max": 100, "integer": False, "required": False, "group": "Pemeriksaan",
     "hint": "Kosongkan jika tidak diukur.",
     "why": "Tebal lipatan kulit trisep, perkiraan kasar jumlah lemak tubuh."},
    {"name": "Insulin", "label": "Insulin serum 2 jam", "unit": "µU/mL",
     "min": 10, "max": 900, "integer": False, "required": False, "group": "Pemeriksaan",
     "hint": "Kosongkan jika tidak diukur.",
     "why": "Hormon pengatur gula darah. Kadar yang tidak wajar dapat menandakan gangguan."},
]
UNITS = {field["name"]: field["unit"] for field in FIELDS}
FORM_GROUPS = ["Riwayat", "Pemeriksaan"]

# Contoh pasien (baris pertama dataset, insulin sengaja kosong). Dipakai untuk kartu di
# beranda dan untuk tombol "Isi contoh" di formulir.
EXAMPLE_PATIENT = {
    "Pregnancies": "6", "Age": "50", "DiabetesPedigreeFunction": "0,627",
    "Glucose": "148", "BMI": "33,6", "BloodPressure": "72",
    "SkinThickness": "35", "Insulin": "",
}

NAV_ITEMS = [
    ("index", "Beranda"),
    ("predict_page", "Prediksi"),
    ("model_page", "Model"),
    ("performance_page", "Performa"),
    ("dataset_page", "Dataset"),
]

app = Flask(__name__)


# ------------------------------------------------------------------ model
def load_artifacts():
    """Muat model dan metrik. Latih ulang jika belum ada atau tidak cocok."""
    try:
        model, metrics = _read_artifacts()
        if _artifacts_match(model, metrics):
            return model, metrics
    except Exception:  # berkas hilang, rusak, atau tidak kompatibel: latih ulang
        pass
    train_model.train()
    return _read_artifacts()


def _read_artifacts():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    return joblib.load(MODEL_PATH), metrics


def _artifacts_match(model, metrics):
    """Model dan metrik harus dari scikit-learn yang sama dan berpasangan satu sama lain."""
    same_version = metrics.get("sklearn_version") == sklearn.__version__
    same_tree = model.named_steps["tree"].tree_.node_count == len(metrics["node_stats"])
    return same_version and same_tree


MODEL, METRICS = load_artifacts()
TREE = MODEL.named_steps["tree"]
IMPUTER = MODEL.named_steps["imputer"]
NODE_STATS = METRICS["node_stats"]
TREE_VIEW = build_tree_view(TREE, NODE_STATS)

# Dengan class_weight="balanced", sebuah daun dinyatakan berisiko bila proporsi pasien
# diabetes di daun itu melebihi proporsi diabetes pada seluruh data (sekitar 35%).
_counts = METRICS["dataset"]["class_counts"]
RISK_THRESHOLD = _counts["1"] / (_counts["0"] + _counts["1"])


# ------------------------------------------------------- filter template
@app.template_filter("num")
def format_number(value, digits=1):
    """Angka gaya Indonesia tanpa nol di belakang koma: 127.50 -> '127,5'."""
    if value is None:
        return "-"
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".") if digits else f"{value:.0f}"
    return text.replace(".", ",")


@app.template_filter("fixed")
def format_fixed(value, digits=2):
    """Angka dengan jumlah desimal tetap: 0.8 -> '0,80'."""
    return f"{value:.{digits}f}".replace(".", ",")


@app.template_filter("pct")
def format_percent(value, digits=0):
    return f"{value * 100:.{digits}f}".replace(".", ",") + "%"


# ------------------------------------------------------------- validasi
def parse_number(raw):
    """Kembalikan (angka, galat). Galat 'blank' berarti kolom dikosongkan."""
    text = (raw or "").strip().replace(",", ".")
    if not text:
        return None, "blank"
    try:
        value = float(text)
    except ValueError:
        return None, "Masukkan angka, misalnya 120 atau 33,6."
    if math.isnan(value) or math.isinf(value):
        return None, "Masukkan angka yang valid."
    return value, None


def validate_field(field, raw):
    """Kembalikan (nilai, galat, peringatan) untuk satu kolom formulir."""
    value, error = parse_number(raw)
    if error == "blank":
        return None, ("Wajib diisi." if field["required"] else None), None
    if error:
        return None, error, None
    if field["integer"] and value != int(value):
        return None, "Harus bilangan bulat.", None
    if not field["min"] <= value <= field["max"]:
        return None, _range_message(field), None

    low, high = METRICS["ranges"][field["name"]]
    warning = None
    if not low <= value <= high:
        warning = (f"Di luar rentang data latih ({format_number(low, 2)} sampai "
                   f"{format_number(high, 2)}); hasil kurang dapat diandalkan.")
    return value, None, warning


def _range_message(field):
    message = (f"Nilai harus antara {format_number(field['min'], 2)} dan "
               f"{format_number(field['max'], 2)} {field['unit']}.").replace(" .", ".")
    if not field["required"]:
        message += " Kosongkan jika tidak diketahui."
    return message


def validate_form(form):
    values, errors, warnings = {}, {}, {}
    for field in FIELDS:
        name = field["name"]
        value, error, warning = validate_field(field, form.get(name))
        values[name] = value
        if error:
            errors[name] = error
        if warning:
            warnings[name] = warning
    return values, errors, warnings


# ------------------------------------------------------------- penjelasan
def reference_category(feature, value):
    """Kategori acuan umum (bukan diagnosis) untuk beberapa indikator. Kembalikan (teks, nada)."""
    if feature == "Glucose":  # glukosa plasma 2 jam setelah tes toleransi glukosa
        if value < 140:
            return "Rentang normal", "ok"
        return ("Rentang prediabetes", "warn") if value < 200 else ("Rentang diabetes", "high")
    if feature == "BMI":
        if value < 18.5:
            return "Kurus", "warn"
        if value < 25:
            return "Normal", "ok"
        return ("Kelebihan berat badan", "warn") if value < 30 else ("Obesitas", "high")
    if feature == "BloodPressure":  # tekanan diastolik
        if value < 80:
            return "Normal", "ok"
        return ("Di atas normal", "warn") if value < 90 else ("Tinggi", "high")
    return None


def describe_inputs(filled, skipped):
    """Nilai tiap indikator dibandingkan dengan acuan umum dan median data latih."""
    rows = []
    for index, feature in enumerate(FEATURES):
        value = float(filled[index])
        median = METRICS["medians"][feature]
        reference = None if feature in skipped else reference_category(feature, value)
        if abs(value - median) < 1e-9:
            position = "sama dengan median"
        else:
            position = "di atas median" if value > median else "di bawah median"
        rows.append({
            "label": LABELS[feature], "unit": UNITS[feature], "value": value,
            "median": median, "position": position, "imputed": feature in skipped,
            "category": reference[0] if reference else None,
            "tone": reference[1] if reference else "none",
        })
    return rows


def build_advice(positive, inputs):
    """Penjelasan arti hasil dan langkah yang disarankan."""
    high = [row["label"] for row in inputs if row["tone"] == "high"]
    if positive:
        text = ("Model menilai pola data ini mirip dengan pasien diabetes pada data latih. "
                "Sebaiknya lakukan pemeriksaan lanjutan ke dokter atau fasilitas kesehatan, "
                "misalnya glukosa puasa atau HbA1c, untuk memastikan.")
    else:
        text = ("Model menilai pola data ini lebih mirip pasien tanpa diabetes. Tetap jaga pola "
                "makan, aktivitas fisik, dan berat badan, serta periksa berkala. Hasil rendah "
                "bukan jaminan bebas diabetes.")
    return {"text": text, "high": high}


# ------------------------------------------------------------- prediksi
def predict(values):
    """Prediksi satu pasien dan susun jalur keputusannya dari akar ke daun."""
    row = pd.DataFrame([[np.nan if values[f] is None else values[f] for f in FEATURES]],
                       columns=FEATURES)
    filled = IMPUTER.transform(row)[0]
    skipped = {f for f in FEATURES if values[f] is None}

    node_path = TREE.decision_path(filled.reshape(1, -1)).indices
    steps = [_describe_step(node, filled, skipped)
             for node in node_path if TREE.tree_.children_left[node] != -1]
    inputs = describe_inputs(filled, skipped)
    leaf = int(node_path[-1])

    predicted = int(MODEL.predict(row)[0])
    stats = NODE_STATS[leaf]
    return {
        "positive": predicted == 1,
        "label": CLASS_NAMES[predicted],
        "steps": steps,
        "leaf_n": stats["n"],
        "leaf_pos": stats["pos"],
        "leaf_share": stats["pos"] / stats["n"] if stats["n"] else 0.0,
        "imputed": [LABELS[f] for f in FEATURES if f in skipped],
        "path": [int(node) for node in node_path],
        "factors": list(dict.fromkeys(step["feature"] for step in steps)),
        "inputs": inputs,
        "advice": build_advice(predicted == 1, inputs),
    }


def _describe_step(node, filled, skipped):
    structure = TREE.tree_
    feature_index = structure.feature[node]
    feature = FEATURES[feature_index]
    threshold = float(structure.threshold[node])
    value = float(filled[feature_index])
    unit = UNITS[feature]
    left = value <= threshold
    relation = "sama dengan atau di bawah" if left else "di atas"
    return {
        "feature": LABELS[feature],
        "value": value,
        "threshold": threshold,
        "left": left,
        "imputed": feature in skipped,
        "unit": unit,
        "sentence": (f"{LABELS[feature]} {format_number(value, 1)} {unit} {relation} batas "
                     f"{format_number(threshold, 2)} {unit}, jadi pohon memilih cabang "
                     f"{'kiri (Ya)' if left else 'kanan (Tidak)'}.").replace("  ", " "),
    }


# ---------------------------------------------------------------- rute
@app.context_processor
def shared_context():
    return {"app_name": APP_NAME, "nav_items": NAV_ITEMS, "active": request.endpoint,
            "m": METRICS, "labels": LABELS, "raw_threshold": RISK_THRESHOLD,
            "tree_view": TREE_VIEW}


@app.route("/")
def index():
    return render_template("index.html", fields=FIELDS, importance={
        item["label"]: item["value"] for item in METRICS["importance"]})


@app.route("/prediksi", methods=["GET", "POST"])
def predict_page():
    context = {"fields": FIELDS, "groups": FORM_GROUPS, "example": EXAMPLE_PATIENT,
               "form": {}, "errors": {}, "warnings": {}, "result": None}
    if request.method == "POST":
        values, errors, warnings = validate_form(request.form)
        context.update(form=request.form, errors=errors, warnings=warnings,
                       result=None if errors else predict(values))
    return render_template("predict.html", **context)


@app.route("/model")
def model_page():
    return render_template("model.html")


@app.route("/performa")
def performance_page():
    return render_template("performance.html")


@app.route("/dataset")
def dataset_page():
    return render_template("dataset.html", fields=FIELDS)


@app.route("/health")
def health():
    return {"status": "ok", "sklearn": sklearn.__version__}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
