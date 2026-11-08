import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import figures
from config import (CLASS_NAMES, COLUMNS, CV_FOLDS, DATA_PATH, DEPTHS_TO_COMPARE, FEATURES,
                    LABELS, METRICS_PATH, MODEL_PATH, PRUNED_DEPTH, RANDOM_STATE, TARGET,
                    TEST_SIZE, ZERO_AS_MISSING)


# --------------------------------------------------------------------- data
def load_raw_data():
    """Baca CSV. Berkas ini tidak punya baris header, jadi nama kolom diberikan manual."""
    return pd.read_csv(DATA_PATH, header=None, names=COLUMNS)


def clean_data(raw):
    """Ubah angka 0 yang mustahil menjadi NaN. Pengisian median dilakukan di Pipeline."""
    cleaned = raw.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, np.nan)
    return cleaned


def make_pipeline(max_depth):
    """Pengisi median lalu pohon keputusan. Hanya max_depth yang membedakan Model A dan B."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(max_depth=max_depth,
                                        class_weight="balanced",
                                        random_state=RANDOM_STATE)),
    ])


def format_number(value):
    """Format angka gaya Indonesia: 127.5 -> '127,5' dan 30.0 -> '30'."""
    if abs(value - round(value)) < 1e-9:
        text = f"{value:.0f}"
    else:
        text = f"{value:.3f}".rstrip("0")
    return text.replace(".", ",")


# ------------------------------------------------------------ interpretation
def compute_node_stats(model, X, y):
    """Jumlah pasien latih asli (tanpa pembobotan) yang melewati setiap simpul."""
    tree = model.named_steps["tree"]
    imputed = model.named_steps["imputer"].transform(X)
    path = tree.decision_path(imputed)  # matriks sampel x simpul
    totals = np.asarray(path.sum(axis=0)).ravel()
    positives = np.asarray(path.T @ np.asarray(y)).ravel()
    return [{"n": int(n), "pos": int(p)} for n, p in zip(totals, positives)]


def extract_rules(tree, stats):
    """Ubah setiap daun pohon menjadi aturan jika-maka."""
    structure = tree.tree_
    rules = []

    def walk(node, conditions):
        if structure.children_left[node] == -1:
            rules.append(_leaf_rule(node, conditions, structure, stats))
            return
        name = LABELS[FEATURES[structure.feature[node]]]
        threshold = format_number(structure.threshold[node])
        walk(structure.children_left[node], conditions + [f"{name} \u2264 {threshold}"])
        walk(structure.children_right[node], conditions + [f"{name} > {threshold}"])

    walk(0, [])
    return rules


def _leaf_rule(node, conditions, structure, stats):
    samples, positives = stats[node]["n"], stats[node]["pos"]
    predicted = int(np.argmax(structure.value[node][0]))
    return {
        "node": int(node),
        "conditions": conditions,
        "prediction": CLASS_NAMES[predicted],
        "samples": samples,
        "positives": positives,
        "p_diabetes": round(positives / samples, 3) if samples else 0.0,
    }


def pick_highlight_rules(rules):
    """Tiga jalur untuk dijelaskan: daun tidak-diabetes terbesar dan dua daun diabetes terbesar."""
    by_size = sorted(rules, key=lambda rule: -rule["samples"])
    negatives = [r for r in by_size if r["prediction"] != "Diabetes"]
    positives = [r for r in by_size if r["prediction"] == "Diabetes"]
    return [negatives[0], positives[0], positives[1] if len(positives) > 1 else negatives[1]]


# --------------------------------------------------------------- evaluation
def evaluate(model, X_train, y_train, X_test, y_test):
    predictions = model.predict(X_test)
    tree = model.named_steps["tree"]
    return {
        "max_depth": tree.get_params()["max_depth"],
        "actual_depth": int(tree.get_depth()),
        "n_leaves": int(tree.get_n_leaves()),
        "train_accuracy": round(float(model.score(X_train, y_train)), 4),
        "test_accuracy": round(float(model.score(X_test, y_test)), 4),
        "auc": round(float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "report": classification_report(y_test, predictions,
                                        target_names=CLASS_NAMES, output_dict=True),
    }


def cross_validate_depth(max_depth, X_train, y_train):
    """Rata-rata skor validasi silang untuk satu nilai max_depth."""
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    result = cross_validate(make_pipeline(max_depth), X_train, y_train, cv=folds,
                            scoring=["accuracy", "recall", "roc_auc"])
    return {
        "accuracy": float(result["test_accuracy"].mean()),
        "accuracy_std": float(result["test_accuracy"].std()),
        "recall": float(result["test_recall"].mean()),
        "auc": float(result["test_roc_auc"].mean()),
    }


def compare_depths(X_train, y_train, X_test, y_test):
    rows = []
    for depth in DEPTHS_TO_COMPARE:
        model = make_pipeline(depth).fit(X_train, y_train)
        cv = cross_validate_depth(depth, X_train, y_train)
        rows.append({
            "depth": depth,
            "train": round(float(model.score(X_train, y_train)), 4),
            "test": round(float(model.score(X_test, y_test)), 4),
            "cv": round(cv["accuracy"], 4),
            "cv_recall": round(cv["recall"], 4),
            "cv_auc": round(cv["auc"], 4),
        })
    return rows


def evaluate_with_cv(model, max_depth, splits):
    X_train, X_test, y_train, y_test = splits
    result = evaluate(model, X_train, y_train, X_test, y_test)
    cv = cross_validate_depth(max_depth, X_train, y_train)
    result.update(cv_mean=round(cv["accuracy"], 4), cv_std=round(cv["accuracy_std"], 4),
                  cv_recall=round(cv["recall"], 4), cv_auc=round(cv["auc"], 4))
    return result


# --------------------------------------------------------------------- main
def build_metrics(raw, cleaned, splits, model_a, model_b, sweep, stats):
    X_train, X_test, _, _ = splits
    class_counts = cleaned[TARGET].value_counts()
    tree_b = model_b.named_steps["tree"]
    importance = sorted(zip(FEATURES, tree_b.feature_importances_), key=lambda pair: -pair[1])
    rules = extract_rules(tree_b, stats)
    medians = model_b.named_steps["imputer"].statistics_
    correlation = cleaned[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(ascending=False)

    return {
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "labels": LABELS,
        "dataset": {
            "rows": int(len(cleaned)),
            "class_counts": {"0": int(class_counts[0]), "1": int(class_counts[1])},
            "zero_as_missing": {c: int(raw[c].eq(0).sum()) for c in ZERO_AS_MISSING},
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        "medians": dict(zip(FEATURES, map(float, medians))),
        "ranges": {f: [float(cleaned[f].min()), float(cleaned[f].max())] for f in FEATURES},
        "model_a": evaluate_with_cv(model_a, None, splits),
        "model_b": evaluate_with_cv(model_b, PRUNED_DEPTH, splits),
        "depth_sweep": sweep,
        "correlation": [{"label": LABELS[f], "value": round(float(v), 3)}
                        for f, v in correlation.items()],
        "importance": [{"feature": f, "label": LABELS[f], "value": round(float(v), 4)}
                       for f, v in importance],
        "node_stats": stats,
        "rules": rules,
        "highlight_rules": pick_highlight_rules(rules),
    }


def train(make_figures=True):
    """Latih Model A dan B, simpan Model B, metrik, dan grafik. Mengembalikan metrik."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    cleaned = clean_data(raw)
    X, y = cleaned[FEATURES], cleaned[TARGET]
    splits = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_train, X_test, y_train, y_test = splits

    model_a = make_pipeline(None).fit(X_train, y_train)           # tanpa batasan
    model_b = make_pipeline(PRUNED_DEPTH).fit(X_train, y_train)   # dipangkas

    stats = compute_node_stats(model_b, X_train, y_train)
    sweep = compare_depths(X_train, y_train, X_test, y_test)
    metrics = build_metrics(raw, cleaned, splits, model_a, model_b, sweep, stats)

    joblib.dump(model_b, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if make_figures:
        figures.save_all(model_a, model_b, sweep, cleaned, X_test, y_test)

    for name, key in (("Model A", "model_a"), ("Model B", "model_b")):
        r = metrics[key]
        print(f"{name}  latih {r['train_accuracy']:.3f}  uji {r['test_accuracy']:.3f}  "
              f"CV {r['cv_mean']:.3f}")
    print("Model B disimpan ke", MODEL_PATH)
    return metrics


if __name__ == "__main__":
    train()
