"""
Evaluation & Reporting — AI Email Risk Analyzer
=================================================
Loads every saved model, produces confusion-matrix heatmaps, a model-
comparison chart, ROC curves, and a full JSON metrics report.

Usage
-----
    python evaluate.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from utils.preprocessing import clean_text
from utils.feature_engineering import (
    build_tfidf_features,
    build_combined_features,
    load_vectorizer,
)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = 'data/processed/emails_dataset.csv'
BASELINE_DIR = 'models/baseline'
DL_DIR       = 'models/deep_learning'
FIGURES_DIR  = 'reports/figures'
METRICS_DIR  = 'reports/metrics'

# ── Label Config ───────────────────────────────────────────────────────────
LABEL_MAP   = {'safe': 0, 'spam': 1, 'phishing': 2, 'promotion': 3, 'suspicious': 4}
LABEL_NAMES = {0: 'Safe', 1: 'Spam', 2: 'Phishing', 3: 'Promotion', 4: 'Suspicious'}
CLASS_NAMES = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
NUM_CLASSES = len(CLASS_NAMES)

# ── Visual Style ───────────────────────────────────────────────────────────
PALETTE = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_dirs():
    for d in [FIGURES_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)


def _encode_labels(labels):
    encoded = labels.str.strip().str.lower().map(LABEL_MAP)
    return encoded.astype(int)


def _print_section(title):
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {title}")
    print(f"{'=' * w}")


# ── Confusion Matrix Heatmap ──────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir=FIGURES_DIR):
    """Save a professional confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor='white',
        cbar_kws={'shrink': 0.8},
        ax=ax,
    )
    ax.set_title(f'Confusion Matrix — {model_name}', fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    path = os.path.join(save_dir, f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"    📊  Confusion matrix → {path}")
    return cm


# ── Per-class Metrics ─────────────────────────────────────────────────────

def compute_per_class_metrics(y_true, y_pred, model_name):
    """Return a dict of per-class precision / recall / f1."""
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        digits=4,
        zero_division=0,
    )
    print(f"\n    Per-class metrics — {model_name}:")
    header = f"    {'Class':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    print(header)
    print(f"    {'─' * (len(header) - 4)}")
    for cls in CLASS_NAMES:
        m = report[cls]
        print(
            f"    {cls:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1-score']:>10.4f} {int(m['support']):>10}"
        )
    return report


# ── Model Comparison Bar Chart ────────────────────────────────────────────

def plot_model_comparison(all_metrics, save_dir=FIGURES_DIR):
    """Bar chart comparing accuracy & F1 across models."""
    names = list(all_metrics.keys())
    accuracies = [all_metrics[n]['accuracy'] for n in names]
    f1_macros  = [all_metrics[n]['f1_macro'] for n in names]
    f1_weights = [all_metrics[n]['f1_weighted'] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, accuracies,  width, label='Accuracy',    color='#2196F3', edgecolor='white')
    bars2 = ax.bar(x,         f1_macros,   width, label='F1 Macro',    color='#FF5722', edgecolor='white')
    bars3 = ax.bar(x + width, f1_weights,  width, label='F1 Weighted', color='#4CAF50', edgecolor='white')

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
            )

    ax.set_title('Model Comparison — Email Risk Classifier', fontweight='bold', pad=12)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1.12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n📊  Model comparison chart → {path}")


# ── ROC Curves ────────────────────────────────────────────────────────────

def plot_roc_curves(y_true, y_prob, model_name, save_dir=FIGURES_DIR):
    """Plot one-vs-rest ROC curves for the best model."""
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, color=PALETTE[i],
                label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_title(f'ROC Curves (One-vs-Rest) — {model_name}', fontweight='bold', pad=12)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"📊  ROC curves → {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    _ensure_dirs()

    # ── 1. Load & Preprocess ──────────────────────────────────────────────
    _print_section("1  LOADING & PREPROCESSING DATA")

    if not os.path.exists(DATA_PATH):
        print(f"❌  Dataset not found at '{DATA_PATH}'. Exiting.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    texts_original = df['text'].astype(str).tolist()
    texts_clean    = [clean_text(t) for t in texts_original]
    y = _encode_labels(df['label'])

    # Same split as training (same random_state=42)
    X_train_clean, X_test_clean, y_train, y_test, \
        X_train_orig, X_test_orig = train_test_split(
            texts_clean, y, texts_original,
            test_size=0.20, random_state=42, stratify=y,
        )
    y_train = np.array(y_train)
    y_test  = np.array(y_test)
    X_test_clean = list(X_test_clean)
    X_test_orig  = list(X_test_orig)
    X_train_clean = list(X_train_clean)
    X_train_orig  = list(X_train_orig)

    print(f"    Dataset : {len(df):,} emails")
    print(f"    Test set: {len(X_test_clean):,} emails")

    # ── 2. Build Features ─────────────────────────────────────────────────
    _print_section("2  BUILDING FEATURES")

    tfidf_vec = load_vectorizer(os.path.join(BASELINE_DIR, 'tfidf_vectorizer.pkl'))

    # Combined features for LR / RF / XGB
    X_train_combined, _ = build_combined_features(
        X_train_clean, X_train_orig,
        tfidf_vectorizer=tfidf_vec, fit=False,
    )
    X_test_combined, _ = build_combined_features(
        X_test_clean, X_test_orig,
        tfidf_vectorizer=tfidf_vec, fit=False,
    )

    # TF-IDF only for MultinomialNB
    X_test_tfidf, _ = build_tfidf_features(
        X_test_clean, fit=False, vectorizer=tfidf_vec,
    )

    print(f"    Combined features shape : {X_test_combined.shape}")

    # ── 3. Load Models ────────────────────────────────────────────────────
    _print_section("3  LOADING SAVED MODELS")

    baseline_models = {
        'MultinomialNB':      ('multinomial_nb.pkl',      X_test_tfidf),
        'LogisticRegression': ('logistic_regression.pkl', X_test_combined),
        'RandomForest':       ('random_forest.pkl',       X_test_combined),
        'XGBoost':            ('xgboost.pkl',             X_test_combined),
    }

    loaded_models = {}
    for name, (filename, _) in baseline_models.items():
        path = os.path.join(BASELINE_DIR, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
            print(f"    ✓ {name} loaded")
        else:
            print(f"    ✗ {name} not found — skipping")

    # ── 4. Evaluate Each Model ────────────────────────────────────────────
    _print_section("4  EVALUATING MODELS")

    all_metrics = {}
    all_reports = {}
    best_acc = -1
    best_model_name = None
    best_y_pred = None
    best_y_prob = None
    best_X_test = None

    for name, model in loaded_models.items():
        _, X_test_for_model = baseline_models[name]
        y_pred = model.predict(X_test_for_model)

        acc = accuracy_score(y_test, y_pred)
        f1_m = f1_score(y_test, y_pred, average='macro')
        f1_w = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)

        all_metrics[name] = {
            'accuracy':    round(acc, 4),
            'f1_macro':    round(f1_m, 4),
            'f1_weighted': round(f1_w, 4),
            'precision':   round(prec, 4),
            'recall':      round(rec, 4),
        }

        # Confusion matrix heatmap
        plot_confusion_matrix(y_test, y_pred, name)

        # Per-class table
        report = compute_per_class_metrics(y_test, y_pred, name)
        all_reports[name] = report

        # Track best model (by accuracy)
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_y_pred = y_pred
            best_X_test = X_test_for_model
            # Attempt to get probabilities
            if hasattr(model, 'predict_proba'):
                best_y_prob = model.predict_proba(X_test_for_model)

    # ── Deep Learning Model ──────────────────────────────────────────────
    dl_model_path = os.path.join(DL_DIR, 'bilstm_model.keras')
    if os.path.exists(dl_model_path):
        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            from tensorflow.keras.preprocessing.sequence import pad_sequences

            _print_section("4+  EVALUATING Bi-LSTM")

            dl_model = load_model(dl_model_path)
            print(f"    ✓ Bi-LSTM model loaded")

            with open(os.path.join(DL_DIR, 'tokenizer.json'), 'r', encoding='utf-8') as f:
                tokenizer = tokenizer_from_json(f.read())

            X_test_seq = pad_sequences(
                tokenizer.texts_to_sequences(X_test_clean), maxlen=200,
            )

            y_prob_dl = dl_model.predict(X_test_seq, verbose=0)
            y_pred_dl = np.argmax(y_prob_dl, axis=1)

            acc_dl = accuracy_score(y_test, y_pred_dl)
            f1_m_dl = f1_score(y_test, y_pred_dl, average='macro')
            f1_w_dl = f1_score(y_test, y_pred_dl, average='weighted')
            prec_dl = precision_score(y_test, y_pred_dl, average='macro', zero_division=0)
            rec_dl  = recall_score(y_test, y_pred_dl, average='macro', zero_division=0)

            all_metrics['BiLSTM'] = {
                'accuracy':    round(acc_dl, 4),
                'f1_macro':    round(f1_m_dl, 4),
                'f1_weighted': round(f1_w_dl, 4),
                'precision':   round(prec_dl, 4),
                'recall':      round(rec_dl, 4),
            }

            plot_confusion_matrix(y_test, y_pred_dl, 'BiLSTM')
            compute_per_class_metrics(y_test, y_pred_dl, 'BiLSTM')

            # Check if BiLSTM is best
            if acc_dl > best_acc:
                best_acc = acc_dl
                best_model_name = 'BiLSTM'
                best_y_pred = y_pred_dl
                best_y_prob = y_prob_dl

        except ImportError:
            print("    ⚠️  TensorFlow not available — skipping Bi-LSTM evaluation.")
        except Exception as e:
            print(f"    ⚠️  Bi-LSTM evaluation failed: {e}")
    else:
        print(f"\n    ℹ️  Bi-LSTM model not found at {dl_model_path} — skipping.")

    # ── 5. Comparison Charts ──────────────────────────────────────────────
    _print_section("5  GENERATING COMPARISON CHARTS")

    if all_metrics:
        plot_model_comparison(all_metrics)

    # ROC curves for best model
    if best_y_prob is not None and best_model_name is not None:
        print(f"\n    Best model: {best_model_name} (Accuracy = {best_acc:.4f})")
        plot_roc_curves(y_test, best_y_prob, best_model_name)
    else:
        print("    ⚠️  No probability predictions available — skipping ROC curves.")

    # ── 6. Save Full Report ───────────────────────────────────────────────
    _print_section("6  SAVING EVALUATION REPORT")

    report_payload = {
        'dataset_size': len(df),
        'test_size': len(y_test),
        'label_mapping': LABEL_NAMES,
        'model_metrics': all_metrics,
        'best_model': best_model_name,
        'best_accuracy': best_acc,
        'per_class_reports': {},
    }
    for name, report in all_reports.items():
        report_payload['per_class_reports'][name] = {
            cls: {
                'precision': round(report[cls]['precision'], 4),
                'recall':    round(report[cls]['recall'], 4),
                'f1_score':  round(report[cls]['f1-score'], 4),
                'support':   int(report[cls]['support']),
            }
            for cls in CLASS_NAMES if cls in report
        }

    report_path = os.path.join(METRICS_DIR, 'evaluation_results.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, indent=2)
    print(f"    📁  Evaluation report → {report_path}")

    # ── 7. Final Summary ─────────────────────────────────────────────────
    _print_section("FINAL SUMMARY")

    header = f"{'Model':<22} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12} {'Precision':>10} {'Recall':>10}"
    print(header)
    print('─' * len(header))
    for name, m in all_metrics.items():
        print(
            f"{name:<22} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
            f"{m['f1_weighted']:>12.4f} {m['precision']:>10.4f} {m['recall']:>10.4f}"
        )
    print()
    if best_model_name:
        print(f"🏆  Best Model: {best_model_name} with Accuracy = {best_acc:.4f}")
    print("\n✅  Evaluation complete.")


if __name__ == '__main__':
    main()
