"""
Feature Engineering Module for Email Risk Analyzer
===================================================
Builds TF-IDF text features and handcrafted meta features,
then combines them into a single sparse feature matrix for
downstream ML models.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
import joblib
import os

from utils.preprocessing import extract_meta_features


# ---------------------------------------------------------------------------
# TF-IDF Features
# ---------------------------------------------------------------------------

def build_tfidf_features(texts, max_features=5000, ngram_range=(1, 2),
                         fit=True, vectorizer=None):
    """Build TF-IDF features from a collection of text documents.

    Parameters
    ----------
    texts : iterable of str
        Cleaned email texts.
    max_features : int
        Maximum number of TF-IDF features to keep.
    ngram_range : tuple (min_n, max_n)
        Range of n-gram sizes for the TF-IDF vectorizer.
    fit : bool
        If True, fit a new vectorizer on *texts*; if False, transform
        using the provided *vectorizer*.
    vectorizer : TfidfVectorizer or None
        Pre-fitted vectorizer (required when ``fit=False``).

    Returns
    -------
    sparse_matrix : scipy.sparse.csr_matrix
        TF-IDF feature matrix.
    vectorizer : TfidfVectorizer
        Fitted vectorizer (new or reused).
    """
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents='unicode',
            dtype=np.float64,
        )
        sparse_matrix = vectorizer.fit_transform(texts)
    else:
        if vectorizer is None:
            raise ValueError(
                "A fitted vectorizer must be provided when fit=False."
            )
        sparse_matrix = vectorizer.transform(texts)

    return sparse_matrix, vectorizer


# ---------------------------------------------------------------------------
# Handcrafted Meta Features
# ---------------------------------------------------------------------------

def build_meta_features(texts_original):
    """Extract handcrafted meta features from *original* (uncleaned) texts.

    Parameters
    ----------
    texts_original : iterable of str
        Raw email bodies before any text cleaning.

    Returns
    -------
    meta_df : pd.DataFrame
        DataFrame with columns:
        ``email_length``, ``word_count``, ``url_count``,
        ``email_addr_count``, ``special_char_count``,
        ``capital_ratio``, ``suspicious_word_count``,
        ``has_urgency``, ``has_money_ref``, ``has_url_shortener``
    """
    meta_records = []
    for text in texts_original:
        features = extract_meta_features(text)
        meta_records.append(features)

    meta_df = pd.DataFrame(meta_records)

    # Ensure expected column order
    expected_cols = [
        'email_length', 'word_count', 'url_count', 'email_addr_count',
        'special_char_count', 'capital_ratio', 'suspicious_word_count',
        'has_urgency', 'has_money_ref', 'has_url_shortener',
    ]
    # Keep only expected columns that exist, in order
    available = [c for c in expected_cols if c in meta_df.columns]
    meta_df = meta_df[available]

    return meta_df


# ---------------------------------------------------------------------------
# Combined Feature Matrix
# ---------------------------------------------------------------------------

def build_combined_features(texts_clean, texts_original,
                            tfidf_vectorizer=None, fit=True):
    """Combine TF-IDF and meta features into a single sparse matrix.

    Parameters
    ----------
    texts_clean : iterable of str
        Cleaned email texts (used for TF-IDF).
    texts_original : iterable of str
        Original email texts (used for meta features).
    tfidf_vectorizer : TfidfVectorizer or None
        Existing fitted vectorizer (pass when ``fit=False``).
    fit : bool
        Whether to fit a new TF-IDF vectorizer.

    Returns
    -------
    combined_features : scipy.sparse.csr_matrix
        Horizontally stacked [TF-IDF | meta] feature matrix.
    tfidf_vectorizer : TfidfVectorizer
        The fitted TF-IDF vectorizer.
    """
    # TF-IDF features
    tfidf_matrix, tfidf_vectorizer = build_tfidf_features(
        texts_clean, fit=fit, vectorizer=tfidf_vectorizer,
    )

    # Meta features
    meta_df = build_meta_features(texts_original)
    meta_sparse = csr_matrix(meta_df.values.astype(np.float64))

    # Combine
    combined_features = hstack([tfidf_matrix, meta_sparse], format='csr')

    return combined_features, tfidf_vectorizer


# ---------------------------------------------------------------------------
# Persistence Helpers
# ---------------------------------------------------------------------------

def save_vectorizer(vectorizer, path='models/baseline/tfidf_vectorizer.pkl'):
    """Persist a fitted TF-IDF vectorizer to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"[✓] Vectorizer saved → {path}")


def load_vectorizer(path='models/baseline/tfidf_vectorizer.pkl'):
    """Load a previously saved TF-IDF vectorizer."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer not found at {path}")
    vectorizer = joblib.load(path)
    print(f"[✓] Vectorizer loaded ← {path}")
    return vectorizer
