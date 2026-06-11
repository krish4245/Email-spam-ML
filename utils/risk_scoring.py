"""
Risk Scoring Engine for Email Risk Analyzer.

Computes a composite risk score (0–100) from four components:
  1. ML confidence (40%)   — model's predicted probability
  2. Keyword score (30%)   — density of suspicious/dangerous keywords
  3. URL score (15%)       — URL pattern suspiciousness
  4. Meta score (15%)      — textual meta-features (caps, specials, etc.)

Each sub-score is normalised to [0, 1] before weighting.
"""

import re
import math

from utils.explainability import (
    PHISHING_INDICATORS,
    SPAM_INDICATORS,
    SUSPICIOUS_INDICATORS,
)

# ──────────────────────────────────────────────────────────────
# Component scorers
# ──────────────────────────────────────────────────────────────

def calculate_keyword_score(text: str) -> float:
    """Score in ``[0, 1]`` reflecting density of suspicious / dangerous keywords.

    Walks all three indicator dictionaries and counts unique keyword hits.
    The raw count is passed through a log-saturating function so that
    diminishing returns kick in after ~10 hits.
    """
    text_lower = text.lower()
    all_keywords: list[str] = []
    for indicator_dict in (PHISHING_INDICATORS, SPAM_INDICATORS, SUSPICIOUS_INDICATORS):
        for keywords in indicator_dict.values():
            all_keywords.extend(keywords)

    hits = sum(1 for kw in set(all_keywords) if kw in text_lower)
    total = len(set(all_keywords))

    if total == 0:
        return 0.0

    # Log-saturating curve: rapid rise for first few hits, plateaus near 1.0
    raw_ratio = hits / total
    score = min(1.0, raw_ratio * 3)  # 33% keyword hit → score 1.0
    return round(score, 4)


def calculate_url_score(text: str) -> float:
    """Score in ``[0, 1]`` reflecting URL-related risk.

    Factors
    -------
    - Presence of any URL  (+0.1)
    - URL shortener        (+0.25 each, max 0.5)
    - IP-based URL         (+0.3)
    - HTTP-only link       (+0.15)
    - Count of URLs        (+0.05 per URL beyond 1, max 0.3)
    """
    text_lower = text.lower()
    score = 0.0

    urls = re.findall(r'https?://[^\s<>"\']+', text_lower)
    if not urls:
        return 0.0

    # Base presence
    score += 0.1

    # Shorteners
    shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly']
    shortener_hits = sum(
        1 for url in urls for s in shorteners if s in url
    )
    score += min(0.5, shortener_hits * 0.25)

    # IP-based
    if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', text_lower):
        score += 0.3

    # HTTP only
    http_only = [u for u in urls if u.startswith('http://')]
    if http_only:
        score += 0.15

    # URL count
    score += min(0.3, max(0, len(urls) - 1) * 0.05)

    return round(min(1.0, score), 4)


def calculate_meta_score(text: str) -> float:
    """Score in ``[0, 1]`` based on textual meta-features.

    Factors
    -------
    - ``capital_ratio`` — fraction of uppercase letters (spam often SHOUTS)
    - ``special_char_ratio`` — density of ``!``, ``$``, ``*``, ``%``
    - ``exclamation_count`` — raw count of ``!``
    - ``word_length_variance`` — unusually short or long average word length
    """
    if not text or not text.strip():
        return 0.0

    score = 0.0
    alpha_chars = [c for c in text if c.isalpha()]
    total_alpha = len(alpha_chars) or 1

    # Capital ratio
    caps = sum(1 for c in alpha_chars if c.isupper())
    cap_ratio = caps / total_alpha
    if cap_ratio > 0.5:
        score += 0.35
    elif cap_ratio > 0.3:
        score += 0.2
    elif cap_ratio > 0.15:
        score += 0.1

    # Special characters
    specials = sum(1 for c in text if c in '!$*%@#^&')
    special_ratio = specials / max(len(text), 1)
    if special_ratio > 0.05:
        score += 0.25
    elif special_ratio > 0.02:
        score += 0.15
    elif special_ratio > 0.01:
        score += 0.05

    # Exclamation marks
    excl_count = text.count('!')
    if excl_count > 5:
        score += 0.25
    elif excl_count > 2:
        score += 0.15
    elif excl_count > 0:
        score += 0.05

    # Suspicious word count from all dicts
    text_lower = text.lower()
    all_keywords: set[str] = set()
    for indicator_dict in (PHISHING_INDICATORS, SPAM_INDICATORS, SUSPICIOUS_INDICATORS):
        for keywords in indicator_dict.values():
            all_keywords.update(keywords)

    suspicious_word_count = sum(1 for kw in all_keywords if kw in text_lower)
    if suspicious_word_count > 8:
        score += 0.15
    elif suspicious_word_count > 4:
        score += 0.10
    elif suspicious_word_count > 1:
        score += 0.05

    return round(min(1.0, score), 4)


# ──────────────────────────────────────────────────────────────
# Composite risk score
# ──────────────────────────────────────────────────────────────

def get_risk_level(score: float) -> tuple[str, str, str]:
    """Map a 0–100 score to ``(level_name, color, emoji)``.

    Thresholds
    ----------
    0–30  → ``('Low',    'green',  '🟢')``
    31–60 → ``('Medium', 'yellow', '🟡')``
    61–100 → ``('High',   'red',    '🔴')``
    """
    if score <= 30:
        return ('Low', 'green', '🟢')
    elif score <= 60:
        return ('Medium', 'yellow', '🟡')
    else:
        return ('High', 'red', '🔴')


def calculate_risk_score(
    text: str,
    predicted_class: str,
    confidence: float,
    category_probabilities: dict[str, float] | None = None,
) -> dict:
    """Calculate a composite risk score and return a structured report.

    Formula
    -------
    ``risk = (ml_component × 40) + (keyword × 30) + (url × 15) + (meta × 15)``

    For the **safe** class the ML component is *inverted* (1 − confidence)
    so that high confidence in "safe" *lowers* the risk.  For **promotion**
    the confidence is scaled to 60 % to produce a moderate score.

    Parameters
    ----------
    text : str
        Raw email body.
    predicted_class : str
        Predicted label (``safe``, ``spam``, ``phishing``, ``promotion``, ``suspicious``).
    confidence : float
        Model confidence for the predicted class, in ``[0, 1]``.
    category_probabilities : dict, optional
        Full probability distribution across classes (used for display only).

    Returns
    -------
    dict
        ``{
            'score': int,          # 0–100
            'level': str,          # Low / Medium / High
            'color': str,          # green / yellow / red
            'emoji': str,          # 🟢 / 🟡 / 🔴
            'components': {
                'ml_confidence': float,
                'keyword_score': float,
                'url_score': float,
                'meta_score': float,
            },
        }``
    """
    pred_lower = predicted_class.lower()

    # --- ML confidence component (weight 40) ---
    if pred_lower == 'safe':
        # High confidence in "safe" should reduce risk
        ml_component = (1.0 - confidence)
    elif pred_lower == 'promotion':
        # Promotions are mildly risky — scale down
        ml_component = confidence * 0.6
    else:
        # spam / phishing / suspicious — use confidence directly
        ml_component = confidence

    # --- Sub-scores ---
    keyword = calculate_keyword_score(text)
    url = calculate_url_score(text)
    meta = calculate_meta_score(text)

    # --- Weighted sum ---
    raw_score = (
        ml_component * 40
        + keyword * 30
        + url * 15
        + meta * 15
    )
    score = int(round(min(100.0, max(0.0, raw_score))))

    level, color, emoji = get_risk_level(score)

    return {
        'score': score,
        'level': level,
        'color': color,
        'emoji': emoji,
        'components': {
            'ml_confidence': round(ml_component * 40, 2),
            'keyword_score': round(keyword * 30, 2),
            'url_score': round(url * 15, 2),
            'meta_score': round(meta * 15, 2),
        },
    }
