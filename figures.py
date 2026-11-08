"""Grafik untuk aplikasi, dibuat dalam dua versi (terang dan gelap)."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from config import (CLASS_NAMES, FEATURES, IMAGE_DIR, LABELS, PRUNED_DEPTH, TARGET,
                    THEMES)


def apply_theme(theme):
    """Terapkan palet tema ke Matplotlib dan kembalikan warna-warnanya."""
    colors = THEMES[theme]
    plt.rcParams.update({
        "figure.facecolor": colors["bg"], "axes.facecolor": colors["bg"],
        "savefig.facecolor": colors["bg"], "text.color": colors["text"],
        "axes.labelcolor": colors["text"], "axes.edgecolor": colors["grid"],
        "xtick.color": colors["text"], "ytick.color": colors["text"],
        "axes.titlecolor": colors["text"], "font.size": 12,
    })
    return colors


def readable_text_color(background):
    """Hitam atau putih, mana yang lebih terbaca di atas warna latar tertentu."""
    r, g, b = background[:3]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0B1020" if luminance > 0.5 else "#FFFFFF"


def save(fig, name, theme):
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / f"{name}_{theme}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------- performa model
def plot_depth_comparison(sweep, theme):
    colors = apply_theme(theme)
    labels = [str(row["depth"]) if row["depth"] else "None" for row in sweep]
    positions = np.arange(len(sweep))
    chosen = labels.index(str(PRUNED_DEPTH))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(positions, [r["train"] for r in sweep], marker="o",
            color=colors["positive"], label="Akurasi latih")
    ax.plot(positions, [r["cv"] for r in sweep], marker="s",
            color=colors["negative"], label="Akurasi validasi silang")
    ax.plot(positions, [r["cv_auc"] for r in sweep], marker="^", ls=":",
            color=colors["accent"], label="AUC validasi silang")
    ax.axvline(chosen, color=colors["grid"], ls="--")
    ax.text(chosen + 0.12, 0.96, f"Model B (max_depth={PRUNED_DEPTH})",
            color=colors["text"], fontsize=10, fontweight="bold")
    ax.set_xticks(positions, labels)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Skor")
    ax.set_ylim(0.6, 1.02)
    ax.grid(color=colors["grid"], alpha=0.6)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
              fontsize=10, handlelength=1.6, columnspacing=1.4)
    clean_axes(ax)
    save(fig, "depth_sweep", theme)


def plot_confusion_matrices(models, X_test, y_test, theme):
    """Dua heatmap biru berdampingan, sama dengan notebook (Blues, skala per model)."""
    colors = apply_theme(theme)
    cmap = LinearSegmentedColormap.from_list("heat", colors["heat"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    for ax, (title, model) in zip(axes, models):
        matrix = confusion_matrix(y_test, model.predict(X_test))
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=matrix.max())
        for row in range(2):
            for col in range(2):
                shade = cmap(matrix[row, col] / matrix.max())
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                        fontsize=22, fontweight="bold", color=readable_text_color(shade))
        ax.set_xticks([0, 1], ["Prediksi\ntidak", "Prediksi\ndiabetes"], fontsize=10)
        ax.set_yticks([0, 1], ["Aktual\ntidak", "Aktual\ndiabetes"], fontsize=10)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    save(fig, "confusion_matrix", theme)


def plot_roc(models, X_test, y_test, theme):
    colors = apply_theme(theme)
    line_colors = [colors["positive"], colors["accent"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    for (title, model), color in zip(models, line_colors):
        probability = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probability)
        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f"{title} (AUC = {roc_auc_score(y_test, probability):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color=colors["muted"], label="Tebakan acak")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(color=colors["grid"], alpha=0.6)
    ax.legend(frameon=False, loc="lower right")
    clean_axes(ax)
    save(fig, "roc", theme)


# --------------------------------------------------------------- dataset
def plot_class_distribution(data, theme):
    colors = apply_theme(theme)
    counts = data[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    bars = ax.bar(CLASS_NAMES, counts.values, color=[colors["negative"], colors["positive"]],
                  width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 8,
                f"{value} ({value / counts.sum():.1%})".replace(".", ","),
                ha="center", fontweight="bold")
    ax.set_ylabel("Jumlah pasien")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", color=colors["grid"], alpha=0.6)
    ax.set_axisbelow(True)
    clean_axes(ax)
    save(fig, "class_distribution", theme)


def plot_correlation(data, theme):
    colors = apply_theme(theme)
    names = [LABELS[f] for f in FEATURES] + ["Diabetes"]
    matrix = data[FEATURES + [TARGET]].corr().values
    cmap = LinearSegmentedColormap.from_list(
        "corr", [colors["accent"], colors["bg"], colors["positive"]])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1)
    for row in range(len(names)):
        for col in range(len(names)):
            ax.text(col, row, f"{matrix[row, col]:.2f}".replace(".", ","), ha="center",
                    va="center", fontsize=9, color=colors["text"])
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.8)
    colorbar.outline.set_visible(False)
    save(fig, "correlation", theme)


def plot_feature_distributions(data, theme):
    colors = apply_theme(theme)
    features = ["Glucose", "BMI", "Age"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, feature in zip(axes, features):
        values = data[feature].dropna()
        bins = np.histogram_bin_edges(values, bins=22)
        for outcome, name, color in ((0, CLASS_NAMES[0], colors["negative"]),
                                     (1, CLASS_NAMES[1], colors["positive"])):
            subset = data.loc[data[TARGET] == outcome, feature].dropna()
            ax.hist(subset, bins=bins, density=True, alpha=0.45, color=color, label=name)
            ax.hist(subset, bins=bins, density=True, histtype="step", color=color, lw=1.8)
        ax.set_title(LABELS[feature], fontweight="bold")
        ax.set_yticks([])
        clean_axes(ax)
    axes[0].legend(frameon=False, fontsize=9)
    save(fig, "feature_distributions", theme)


# ----------------------------------------------------------------- semua
def save_all(model_a, model_b, sweep, data, X_test, y_test):
    """Buat semua grafik untuk kedua tema."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    models = [("Model A", model_a), ("Model B", model_b)]
    for theme in THEMES:
        plot_depth_comparison(sweep, theme)
        plot_confusion_matrices(models, X_test, y_test, theme)
        plot_roc(models, X_test, y_test, theme)
        plot_class_distribution(data, theme)
        plot_correlation(data, theme)
        plot_feature_distributions(data, theme)
