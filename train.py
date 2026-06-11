"""
Training Pipeline — AI Email Risk Analyzer
============================================
Trains baseline ML models (Phase 1) and a Bi-LSTM deep learning model
(Phase 2) for multi-class email risk classification.

Usage
-----
    python train.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')                    # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from utils.preprocessing import clean_text
from utils.feature_engineering import (
    build_tfidf_features,
    build_meta_features,
    build_combined_features,
    save_vectorizer,
)

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH           = 'data/processed/emails_dataset.csv'
BASELINE_DIR        = 'models/baseline'
DL_DIR              = 'models/deep_learning'
FIGURES_DIR         = 'reports/figures'
METRICS_DIR         = 'reports/metrics'

# ── Label Mapping ──────────────────────────────────────────────────────────
LABEL_MAP = {
    'safe': 0,
    'spam': 1,
    'phishing': 2,
    'promotion': 3,
    'suspicious': 4,
}
LABEL_NAMES = {v: k.title() for k, v in LABEL_MAP.items()}

# ── Helpers ────────────────────────────────────────────────────────────────

def _ensure_dirs():
    for d in [BASELINE_DIR, DL_DIR, FIGURES_DIR, METRICS_DIR]:
        os.makedirs(d, exist_ok=True)


def _encode_labels(labels):
    """Map string labels → integer codes using LABEL_MAP."""
    encoded = labels.str.strip().str.lower().map(LABEL_MAP)
    if encoded.isna().any():
        unknown = labels[encoded.isna()].unique().tolist()
        raise ValueError(f"Unknown labels encountered: {unknown}")
    return encoded.astype(int)


def _print_section(title):
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _evaluate_model(model, X_test, y_test, model_name):
    """Predict, print metrics, return dict of key scores."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    print(f"\n📊  {model_name} Results")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1 Macro : {f1_macro:.4f}")
    print(f"    F1 Weighted : {f1_weighted:.4f}")
    print()
    print(classification_report(
        y_test, y_pred,
        target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        digits=4,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        'accuracy': round(acc, 4),
        'f1_macro': round(f1_macro, 4),
        'f1_weighted': round(f1_weighted, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    _ensure_dirs()
    start_time = time.time()

    # ── 1. Load Data ──────────────────────────────────────────────────────
    _print_section("1  LOADING DATA")

    if not os.path.exists(DATA_PATH):
        print(f"❌  Dataset not found at '{DATA_PATH}'.")
        print("    Run the data‑preparation step first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    print(f"    Loaded {len(df):,} emails  |  Columns: {list(df.columns)}")
    print(f"    Label distribution:\n{df['label'].value_counts().to_string()}")

    # ── 2. Text Preprocessing ────────────────────────────────────────────
    _print_section("2  TEXT PREPROCESSING")

    texts_original = df['text'].astype(str).tolist()
    texts_clean = [clean_text(t) for t in texts_original]
    print(f"    Cleaned {len(texts_clean):,} email texts.")

    # ── 3. Encode Labels & Split ─────────────────────────────────────────
    _print_section("3  ENCODING LABELS & TRAIN/TEST SPLIT")

    y = _encode_labels(df['label'])
    X_train_clean, X_test_clean, y_train, y_test, \
        X_train_orig, X_test_orig = train_test_split(
            texts_clean, y, texts_original,
            test_size=0.20,
            random_state=42,
            stratify=y,
        )

    # Convert to numpy/list for safety
    y_train = np.array(y_train)
    y_test  = np.array(y_test)
    X_train_clean = list(X_train_clean)
    X_test_clean  = list(X_test_clean)
    X_train_orig  = list(X_train_orig)
    X_test_orig   = list(X_test_orig)

    print(f"    Train : {len(X_train_clean):,}  |  Test : {len(X_test_clean):,}")

    # ── 4. Feature Engineering ───────────────────────────────────────────
    _print_section("4  FEATURE ENGINEERING")

    # Combined features (TF-IDF + meta)
    X_train_combined, tfidf_vec = build_combined_features(
        X_train_clean, X_train_orig, fit=True,
    )
    X_test_combined, _ = build_combined_features(
        X_test_clean, X_test_orig,
        tfidf_vectorizer=tfidf_vec, fit=False,
    )

    # TF-IDF only (for MultinomialNB — no negative values)
    X_train_tfidf, _ = build_tfidf_features(
        X_train_clean, fit=True, vectorizer=None,
    )
    # Reuse the *same* vectorizer for test
    X_test_tfidf, _ = build_tfidf_features(
        X_test_clean, fit=False, vectorizer=tfidf_vec,
    )

    save_vectorizer(tfidf_vec, os.path.join(BASELINE_DIR, 'tfidf_vectorizer.pkl'))

    print(f"    Combined features shape : {X_train_combined.shape}")
    print(f"    TF-IDF–only shape       : {X_train_tfidf.shape}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1 — BASELINE ML MODELS
    # ══════════════════════════════════════════════════════════════════════
    _print_section("PHASE 1 — BASELINE ML MODELS")

    results = {}

    # ── 4a. Multinomial Naive Bayes (TF-IDF only) ────────────────────────
    _print_section("4a  MultinomialNB (TF-IDF only)")
    mnb = MultinomialNB()
    mnb.fit(X_train_tfidf, y_train)
    results['MultinomialNB'] = _evaluate_model(mnb, X_test_tfidf, y_test, 'MultinomialNB')
    joblib.dump(mnb, os.path.join(BASELINE_DIR, 'multinomial_nb.pkl'))
    print(f"    💾  Saved → {BASELINE_DIR}/multinomial_nb.pkl")

    # ── 4b. Logistic Regression ──────────────────────────────────────────
    _print_section("4b  Logistic Regression")
    lr = LogisticRegression(
        max_iter=1000, solver='lbfgs',
    )
    lr.fit(X_train_combined, y_train)
    results['LogisticRegression'] = _evaluate_model(lr, X_test_combined, y_test, 'LogisticRegression')
    joblib.dump(lr, os.path.join(BASELINE_DIR, 'logistic_regression.pkl'))
    print(f"    💾  Saved → {BASELINE_DIR}/logistic_regression.pkl")

    # ── 4c. Random Forest ────────────────────────────────────────────────
    _print_section("4c  Random Forest")
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_combined, y_train)
    results['RandomForest'] = _evaluate_model(rf, X_test_combined, y_test, 'RandomForest')
    joblib.dump(rf, os.path.join(BASELINE_DIR, 'random_forest.pkl'))
    print(f"    💾  Saved → {BASELINE_DIR}/random_forest.pkl")

    # ── 4d. XGBoost ──────────────────────────────────────────────────────
    _print_section("4d  XGBoost")
    xgb = XGBClassifier(
        n_estimators=200,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
    )
    xgb.fit(X_train_combined, y_train)
    results['XGBoost'] = _evaluate_model(xgb, X_test_combined, y_test, 'XGBoost')
    joblib.dump(xgb, os.path.join(BASELINE_DIR, 'xgboost.pkl'))
    print(f"    💾  Saved → {BASELINE_DIR}/xgboost.pkl")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2 — DEEP LEARNING (Bi-LSTM)
    # ══════════════════════════════════════════════════════════════════════
    try:
        _print_section("PHASE 2 — Bi-LSTM DEEP LEARNING MODEL")

        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (
            Embedding, Bidirectional, LSTM, Dropout, Dense,
        )
        from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.utils import to_categorical
        from sklearn.utils.class_weight import compute_class_weight

        NUM_WORDS  = 10_000
        MAX_LEN    = 200
        EMBED_DIM  = 64
        EPOCHS     = 20
        BATCH_SIZE = 32

        # Tokenize
        tokenizer = KerasTokenizer(num_words=NUM_WORDS)
        tokenizer.fit_on_texts(X_train_clean)

        X_train_seq = pad_sequences(
            tokenizer.texts_to_sequences(X_train_clean), maxlen=MAX_LEN,
        )
        X_test_seq = pad_sequences(
            tokenizer.texts_to_sequences(X_test_clean), maxlen=MAX_LEN,
        )

        num_classes = len(LABEL_MAP)
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat  = to_categorical(y_test, num_classes)

        # Class weights
        cw_values = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight = {i: w for i, w in enumerate(cw_values)}
        print(f"    Class weights: {class_weight}")

        # Build model
        model = Sequential([
            Embedding(NUM_WORDS, EMBED_DIM, input_length=MAX_LEN),
            Bidirectional(LSTM(64, return_sequences=True)),
            Bidirectional(LSTM(32)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax'),
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )
        model.summary()

        callbacks = [
            EarlyStopping(
                monitor='val_loss', patience=3,
                restore_best_weights=True, verbose=1,
            ),
            ReduceLROnPlateau(
                monitor='val_loss', patience=2,
                factor=0.5, min_lr=1e-6, verbose=1,
            ),
        ]

        history = model.fit(
            X_train_seq, y_train_cat,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.15,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate
        loss, acc = model.evaluate(X_test_seq, y_test_cat, verbose=0)
        y_pred_dl = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)
        f1_dl = f1_score(y_test, y_pred_dl, average='macro')

        print(f"\n📊  Bi-LSTM Results")
        print(f"    Test Loss    : {loss:.4f}")
        print(f"    Test Accuracy: {acc:.4f}")
        print(f"    F1 Macro     : {f1_dl:.4f}")
        print()
        print(classification_report(
            y_test, y_pred_dl,
            target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
            digits=4,
        ))

        results['BiLSTM'] = {
            'accuracy': round(acc, 4),
            'f1_macro': round(f1_dl, 4),
            'f1_weighted': round(
                f1_score(y_test, y_pred_dl, average='weighted'), 4,
            ),
        }

        # ── Save DL artefacts ────────────────────────────────────────────
        model.save(os.path.join(DL_DIR, 'bilstm_model.keras'))
        print(f"    💾  Model saved → {DL_DIR}/bilstm_model.keras")

        tokenizer_json = tokenizer.to_json()
        with open(os.path.join(DL_DIR, 'tokenizer.json'), 'w', encoding='utf-8') as f:
            f.write(tokenizer_json)
        print(f"    💾  Tokenizer saved → {DL_DIR}/tokenizer.json")

        np.save(
            os.path.join(DL_DIR, 'label_classes.npy'),
            np.array([LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]),
        )
        print(f"    💾  Label classes saved → {DL_DIR}/label_classes.npy")

        # ── Training History Plot ────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy',
                     linewidth=2, color='#2196F3')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',
                     linewidth=2, color='#FF5722', linestyle='--')
        axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(history.history['loss'], label='Train Loss',
                     linewidth=2, color='#2196F3')
        axes[1].plot(history.history['val_loss'], label='Val Loss',
                     linewidth=2, color='#FF5722', linestyle='--')
        axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Bi-LSTM Training History', fontsize=16, fontweight='bold',
                      y=1.02)
        plt.tight_layout()
        history_path = os.path.join(FIGURES_DIR, 'training_history.png')
        plt.savefig(history_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    📈  Training history plot → {history_path}")

    except ImportError as e:
        print(f"\n⚠️  TensorFlow not available — skipping deep learning phase.")
        print(f"    Error: {e}")
        results['BiLSTM'] = {
            'accuracy': None, 'f1_macro': None, 'f1_weighted': None,
            'error': str(e),
        }
    except Exception as e:
        print(f"\n⚠️  Deep learning training failed — skipping.")
        print(f"    Error: {e}")
        results['BiLSTM'] = {
            'accuracy': None, 'f1_macro': None, 'f1_weighted': None,
            'error': str(e),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    _print_section("MODEL COMPARISON")

    header = f"{'Model':<22} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}"
    print(header)
    print('─' * len(header))
    for name, scores in results.items():
        acc_str = f"{scores['accuracy']:.4f}" if scores['accuracy'] is not None else '  N/A'
        f1m_str = f"{scores['f1_macro']:.4f}" if scores['f1_macro'] is not None else '  N/A'
        f1w_str = f"{scores['f1_weighted']:.4f}" if scores['f1_weighted'] is not None else '  N/A'
        print(f"{name:<22} {acc_str:>10} {f1m_str:>10} {f1w_str:>12}")
    print()

    # Save results JSON
    results_path = os.path.join(METRICS_DIR, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"📁  Training results saved → {results_path}")

    elapsed = time.time() - start_time
    print(f"\n✅  Training complete in {elapsed / 60:.1f} minutes.")


if __name__ == '__main__':
    main()
