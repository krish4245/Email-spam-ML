"""
AI Email Risk Analyzer — Streamlit Web Application.

A premium, glassmorphism-styled dashboard for real-time email threat
classification, risk scoring, and explainable-AI explanations.

Run from project root:
    streamlit run app/streamlit_app.py
"""

# ── Path bootstrap ───────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Standard / third-party imports ───────────────────────────
import json
import pickle
import numpy as np
import streamlit as st

# ── Project imports ──────────────────────────────────────────
from utils.explainability import get_all_explanations
from utils.risk_scoring import calculate_risk_score, get_risk_level

# Attempt optional imports — fail gracefully
try:
    from utils.preprocessing import clean_text
except ImportError:
    def clean_text(text: str) -> str:  # type: ignore[misc]
        """Fallback cleaner when preprocessing module is unavailable."""
        return text.strip().lower()

# ── Page configuration ───────────────────────────────────────
st.set_page_config(
    page_title="AI Email Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═════════════════════════════════════════════════════════════
#  CUSTOM CSS — dark glassmorphism theme
# ═════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Root variables ───────────────────────────────────────── */
:root {
    --bg-primary: #0e1117;
    --bg-card: rgba(255, 255, 255, 0.04);
    --border-card: rgba(255, 255, 255, 0.08);
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-purple: #a855f7;
    --accent-blue: #3b82f6;
    --safe-green: #00c853;
    --spam-orange: #ff9100;
    --phishing-red: #ff1744;
    --promotion-blue: #2979ff;
    --suspicious-yellow: #ffd600;
    --glass-bg: rgba(17, 25, 40, 0.75);
    --glass-border: rgba(255, 255, 255, 0.12);
}

/* ── Global overrides ─────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .main {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary);
}

/* ── Animated gradient header ─────────────────────────────── */
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 30%, #3b82f6 60%, #06b6d4 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradient-shift 6s ease infinite;
    margin-bottom: 0;
    line-height: 1.15;
}
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
    font-weight: 400;
    letter-spacing: 0.01em;
}

/* ── Glass cards ──────────────────────────────────────────── */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(16px) saturate(180%);
    -webkit-backdrop-filter: blur(16px) saturate(180%);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(168, 85, 247, 0.12);
}

/* ── Classification badge ─────────────────────────────────── */
.class-badge {
    display: inline-block;
    padding: 0.55rem 1.4rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    animation: badge-pop 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}
@keyframes badge-pop {
    0%   { transform: scale(0.5); opacity: 0; }
    100% { transform: scale(1);   opacity: 1; }
}
.badge-safe        { background: rgba(0,200,83,0.18);  color: #00c853; border: 1px solid rgba(0,200,83,0.35); }
.badge-spam        { background: rgba(255,145,0,0.18); color: #ff9100; border: 1px solid rgba(255,145,0,0.35); }
.badge-phishing    { background: rgba(255,23,68,0.18); color: #ff1744; border: 1px solid rgba(255,23,68,0.35); }
.badge-promotion   { background: rgba(41,121,255,0.18);color: #2979ff; border: 1px solid rgba(41,121,255,0.35); }
.badge-suspicious  { background: rgba(255,214,0,0.18); color: #ffd600; border: 1px solid rgba(255,214,0,0.35); }

/* ── Risk score ring ──────────────────────────────────────── */
.risk-number {
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1;
    letter-spacing: -0.04em;
}
.risk-label {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 0.25rem;
}

/* ── Progress bars ────────────────────────────────────────── */
.component-bar-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
    margin: 4px 0 10px 0;
}
.component-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}

/* ── Explanation bullets ──────────────────────────────────── */
.explanation-item {
    padding: 0.5rem 0.75rem;
    border-left: 3px solid var(--accent-purple);
    margin-bottom: 0.45rem;
    font-size: 0.92rem;
    background: rgba(168,85,247,0.05);
    border-radius: 0 8px 8px 0;
    transition: background 0.2s ease;
}
.explanation-item:hover {
    background: rgba(168,85,247,0.12);
}

/* ── Warning items ────────────────────────────────────────── */
.warning-item {
    padding: 0.5rem 0.75rem;
    border-left: 3px solid #ff9100;
    margin-bottom: 0.45rem;
    font-size: 0.92rem;
    background: rgba(255,145,0,0.05);
    border-radius: 0 8px 8px 0;
}

/* ── Sidebar styling ──────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: rgba(14, 17, 23, 0.95) !important;
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    font-weight: 600;
}

/* ── Button styling ───────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 50%, #3b82f6 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    padding: 0.65rem 2rem !important;
    border: none !important;
    border-radius: 12px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    letter-spacing: 0.02em;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(99, 102, 241, 0.4) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Text area ────────────────────────────────────────────── */
.stTextArea textarea {
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 2px rgba(168, 85, 247, 0.2) !important;
}

/* ── Expander ─────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: var(--text-primary) !important;
}

/* ── Divider ──────────────────────────────────────────────── */
.subtle-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--glass-border), transparent);
    margin: 1.5rem 0;
}

/* ── Metric value overrides ───────────────────────────────── */
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 800 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
#  CONSTANTS
# ═════════════════════════════════════════════════════════════

CLASS_LABELS = ['safe', 'spam', 'phishing', 'promotion', 'suspicious']

CLASS_COLORS = {
    'safe': '#00c853',
    'spam': '#ff9100',
    'phishing': '#ff1744',
    'promotion': '#2979ff',
    'suspicious': '#ffd600',
}

CLASS_EMOJIS = {
    'safe': '✅',
    'spam': '📧',
    'phishing': '🎣',
    'promotion': '📢',
    'suspicious': '⚠️',
}

SAMPLE_EMAILS = {
    '✅ Safe — Meeting Reminder': (
        "Hi team, just a reminder that our quarterly review meeting is scheduled "
        "for Thursday at 2 PM in Conference Room B. Please bring your progress "
        "reports. Thanks, Sarah"
    ),
    '📧 Spam — Prize Scam': (
        "CONGRATULATIONS!!! You have been SELECTED as our GRAND PRIZE WINNER of "
        "$1,000,000! Click here NOW to claim your prize before it expires! This is "
        "NOT a joke! Act IMMEDIATELY!"
    ),
    '🎣 Phishing — PayPal Spoof': (
        "Dear Customer, We detected unusual activity on your PayPal account. Your "
        "account has been temporarily suspended. Please verify your identity by "
        "clicking the link below within 24 hours or your account will be permanently "
        "locked. http://paypa1-secure.verify-account.com/login"
    ),
    '📢 Promotion — Tech Sale': (
        "Hey there! 🎉 Our biggest sale of the year is HERE! Get up to 70% off on "
        "all electronics this weekend only. Use code MEGA70 at checkout. Free "
        "shipping on orders over $50. Shop now at TechDeals.com"
    ),
    '⚠️ Suspicious — Wire Transfer': (
        "Hello, I am writing regarding a confidential matter. I have a large sum "
        "that needs to be transferred urgently. I need your assistance and trust. "
        "Please keep this between us. Reply with your bank details for processing."
    ),
}

# ═════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═════════════════════════════════════════════════════════════

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')


@st.cache_resource(show_spinner=False)
def load_baseline_model():
    """Load the best available baseline model + TF-IDF vectorizer."""
    import joblib
    vectorizer_path = os.path.join(MODELS_DIR, 'baseline', 'tfidf_vectorizer.pkl')
    if not os.path.exists(vectorizer_path):
        return None, None

    # Try models in order of expected performance
    model_candidates = [
        'xgboost.pkl',
        'random_forest.pkl',
        'logistic_regression.pkl',
        'multinomial_nb.pkl',
    ]
    model_path = None
    for candidate in model_candidates:
        path = os.path.join(MODELS_DIR, 'baseline', candidate)
        if os.path.exists(path):
            model_path = path
            break

    if model_path is None:
        return None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


@st.cache_resource(show_spinner=False)
def load_deep_learning_model():
    """Load the Keras/TF deep-learning model + tokenizer."""
    model_path = os.path.join(MODELS_DIR, 'deep_learning', 'bilstm_model.keras')
    tokenizer_path = os.path.join(MODELS_DIR, 'deep_learning', 'tokenizer.json')

    # Also try .h5 extension
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, 'deep_learning', 'bilstm_model.h5')

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        return None, None

    try:
        from tensorflow.keras.models import load_model  # type: ignore[import]
        model = load_model(model_path)
        from tensorflow.keras.preprocessing.text import tokenizer_from_json  # type: ignore[import]
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            tokenizer = tokenizer_from_json(f.read())
        return model, tokenizer
    except Exception:
        return None, None


# Integer label -> string class name mapping (matches train.py LABEL_MAP)
_INT_TO_CLASS = {0: 'safe', 1: 'spam', 2: 'phishing', 3: 'promotion', 4: 'suspicious'}


def predict_baseline(text: str, model, vectorizer) -> tuple[str, float, dict[str, float]]:
    """Run baseline model prediction. Returns (class, confidence, probabilities)."""
    from utils.preprocessing import extract_meta_features
    from scipy.sparse import hstack, csr_matrix

    cleaned = clean_text(text)

    # Build TF-IDF features
    tfidf_features = vectorizer.transform([cleaned])

    # Build meta features (same as training) and combine
    meta = extract_meta_features(text)
    meta_values = np.array([[
        meta.get('email_length', 0),
        meta.get('word_count', 0),
        meta.get('url_count', 0),
        meta.get('email_addr_count', 0),
        meta.get('special_char_count', 0),
        meta.get('capital_ratio', 0.0),
        meta.get('suspicious_word_count', 0),
        meta.get('has_urgency', 0),
        meta.get('has_money_ref', 0),
        meta.get('has_url_shortener', 0),
    ]], dtype=np.float64)

    # For MultinomialNB use TF-IDF only (no negative values allowed)
    model_class_name = type(model).__name__
    if model_class_name == 'MultinomialNB':
        features = tfidf_features
    else:
        features = hstack([tfidf_features, csr_matrix(meta_values)], format='csr')

    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    pred_idx = int(np.argmax(probabilities))
    # Map integer label back to string class name
    raw_label = classes[pred_idx]
    predicted_class = _INT_TO_CLASS.get(int(raw_label), str(raw_label))
    confidence = float(probabilities[pred_idx])

    # Build probability dict with string class names
    prob_dict = {}
    for c, p in zip(classes, probabilities):
        label_str = _INT_TO_CLASS.get(int(c), str(c))
        prob_dict[label_str] = float(p)
    # Ensure all 5 classes present
    for label in CLASS_LABELS:
        prob_dict.setdefault(label, 0.0)

    return predicted_class, confidence, prob_dict


def predict_deep_learning(text: str, model, tokenizer) -> tuple[str, float, dict[str, float]]:
    """Run deep learning model prediction. Returns (class, confidence, probabilities)."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore[import]

    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    probabilities = model.predict(padded, verbose=0)[0]

    pred_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_LABELS[pred_idx]
    confidence = float(probabilities[pred_idx])

    prob_dict = {label: float(p) for label, p in zip(CLASS_LABELS, probabilities)}
    return predicted_class, confidence, prob_dict


def keyword_fallback_predict(text: str) -> tuple[str, float, dict[str, float]]:
    """Simple keyword-based fallback when no models are available."""
    from utils.explainability import (
        PHISHING_INDICATORS, SPAM_INDICATORS, SUSPICIOUS_INDICATORS,
        _match_indicators,
    )

    phishing_hits = sum(len(v) for v in _match_indicators(text, PHISHING_INDICATORS).values())
    spam_hits = sum(len(v) for v in _match_indicators(text, SPAM_INDICATORS).values())
    suspicious_hits = sum(len(v) for v in _match_indicators(text, SUSPICIOUS_INDICATORS).values())

    total = phishing_hits + spam_hits + suspicious_hits + 1  # +1 avoid div/0

    scores = {
        'phishing': phishing_hits / total,
        'spam': spam_hits / total,
        'suspicious': suspicious_hits / total,
        'promotion': 0.05,
        'safe': max(0.0, 1.0 - (phishing_hits + spam_hits + suspicious_hits) / max(total, 1)),
    }

    # Normalise
    total_score = sum(scores.values()) or 1.0
    scores = {k: v / total_score for k, v in scores.items()}

    predicted = max(scores, key=scores.get)  # type: ignore[arg-type]
    confidence = scores[predicted]
    return predicted, confidence, scores


# ======================================================================
#  MAIN CONTENT & TABS
# ======================================================================

# ── Sidebar Settings ──────────────────────────────────────────
with st.sidebar:
    st.markdown('<p style="font-size:1.6rem;font-weight:800;margin-bottom:0.5rem;color:var(--accent-purple);text-align:center;">🛡️ Analyzer Settings</p>', unsafe_allow_html=True)
    st.markdown('<div style="height:2px;background:linear-gradient(90deg,transparent,#a855f7,transparent);margin-bottom:1.5rem;"></div>', unsafe_allow_html=True)
    
    # Model selection
    st.markdown('<p style="font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;color:var(--text-primary);">🧠 Classification Engine</p>', unsafe_allow_html=True)
    model_choice = st.radio(
        "Choose Model Architecture:",
        ["Best Baseline Model (XGBoost/RF/LR)", "Deep Learning Model (Bi-LSTM)"],
        index=0,
        label_visibility="collapsed",
        key="sidebar_model_choice"
    )
    
    st.markdown('<div class="subtle-divider" style="margin:1rem 0;"></div>', unsafe_allow_html=True)
    
    # Sample templates selection
    st.markdown('<p style="font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;color:var(--text-primary);">📝 Load Preset Templates</p>', unsafe_allow_html=True)
    sample_options = ['— Select a sample —'] + list(SAMPLE_EMAILS.keys())
    selected_sample = st.selectbox(
        "Select a template to auto-fill:",
        options=sample_options,
        index=0,
        label_visibility="collapsed",
        key="sidebar_sample_select"
    )
    
    st.markdown('<div class="subtle-divider" style="margin:1rem 0;"></div>', unsafe_allow_html=True)
    
    # About Section
    st.markdown('<p style="font-weight:700;font-size:1.1rem;margin-bottom:0.5rem;color:var(--text-primary);">ℹ️ Project Overview</p>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.88rem;color:var(--text-secondary);line-height:1.5;">'
        'This dashboard classifies emails into 5 risk tiers (Safe, Spam, Phishing, '
        'Promotion, Suspicious), provides explainable AI indicators, calculates threat '
        'scores (0-100), and scans IMAP/Gmail inboxes.'
        '</div>',
        unsafe_allow_html=True
    )

# ── Hero header ──────────────────────────────────────────────
st.markdown(
    '<p class="hero-title">🛡️ AI Email Risk Analyzer</p>'
    '<p class="hero-subtitle">'
    'Real-time threat classification · Explainable AI · Active Inbox Monitoring'
    '</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

# Define Tabs
tab1, tab2 = st.tabs(["🔬 Single Email Analyzer", "📬 Active Inbox Scanner"])

# Helper function to run analysis pipeline
def run_analysis_pipeline(email_text):
    using_fallback = False
    
    if 'Baseline' in model_choice:
        model, vectorizer = load_baseline_model()
        if model is not None and vectorizer is not None:
            predicted_class, confidence, prob_dict = predict_baseline(
                email_text, model, vectorizer
            )
        else:
            using_fallback = True
    else:
        model, tokenizer = load_deep_learning_model()
        if model is not None and tokenizer is not None:
            predicted_class, confidence, prob_dict = predict_deep_learning(
                email_text, model, tokenizer
            )
        else:
            using_fallback = True

    if using_fallback:
        predicted_class, confidence, prob_dict = keyword_fallback_predict(email_text)

    # Risk score
    risk = calculate_risk_score(
        email_text, predicted_class, confidence, prob_dict
    )
    # Explanations
    explanations = get_all_explanations(email_text, predicted_class, confidence)
    
    return predicted_class, confidence, prob_dict, risk, explanations, using_fallback

# Helper function to render analysis results
def render_analysis_results(email_text, predicted_class, confidence, prob_dict, risk, explanations, is_fallback):
    if is_fallback:
        st.warning(
            "⚠️ Trained model not found — using keyword-based fallback. "
            "Train your models first for ML-powered predictions.",
            icon="⚠️",
        )
        
    st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)

    # ── Row 1: Classification + Risk Score ───────────────────
    col_class, col_risk = st.columns(2)

    with col_class:
        badge_class = f"badge-{predicted_class.lower()}"
        color = CLASS_COLORS.get(predicted_class.lower(), '#8b949e')
        emoji = CLASS_EMOJIS.get(predicted_class.lower(), '📧')

        st.markdown(
            '<div class="glass-card" style="text-align:center;">'
            f'<p style="font-size:0.8rem;color:var(--text-secondary);text-transform:uppercase;'
            f'letter-spacing:0.12em;font-weight:600;margin-bottom:0.6rem;">Classification</p>'
            f'<span class="class-badge {badge_class}">{emoji} {predicted_class.upper()}</span>'
            f'<p style="margin-top:1rem;font-size:2.2rem;font-weight:800;color:{color};">'
            f'{confidence * 100:.1f}%</p>'
            f'<p style="font-size:0.8rem;color:var(--text-secondary);">Model Confidence</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_risk:
        risk_color_map = {'green': '#00c853', 'yellow': '#ffd600', 'red': '#ff1744'}
        r_color = risk_color_map.get(risk['color'], '#8b949e')

        st.markdown(
            '<div class="glass-card" style="text-align:center;">'
            f'<p style="font-size:0.8rem;color:var(--text-secondary);text-transform:uppercase;'
            f'letter-spacing:0.12em;font-weight:600;margin-bottom:0.6rem;">Risk Assessment</p>'
            f'<p class="risk-number" style="color:{r_color};">{risk["emoji"]} {risk["score"]}</p>'
            f'<p class="risk-label" style="color:{r_color};">{risk["level"]} Risk</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Row 2: Risk component breakdown ──────────────────────
    st.markdown(
        '<div class="glass-card">'
        '<p style="font-weight:700;font-size:1.05rem;margin-bottom:1rem;">'
        '📊 Risk Score Breakdown</p>',
        unsafe_allow_html=True,
    )

    component_labels = {
        'ml_confidence': ('🧠 ML Confidence', 40),
        'keyword_score': ('🔑 Keyword Analysis', 30),
        'url_score': ('🔗 URL Patterns', 15),
        'meta_score': ('📊 Meta Features', 15),
    }

    component_colors = {
        'ml_confidence': '#a855f7',
        'keyword_score': '#f97316',
        'url_score': '#3b82f6',
        'meta_score': '#06b6d4',
    }

    cols_comp = st.columns(4)
    for idx, (key, (label, max_val)) in enumerate(component_labels.items()):
        val = risk['components'][key]
        pct = (val / max_val) * 100 if max_val else 0
        bar_color = component_colors[key]
        with cols_comp[idx]:
            st.markdown(
                f'<p style="font-size:0.78rem;color:var(--text-secondary);margin-bottom:2px;">{label}</p>'
                f'<p style="font-size:1.35rem;font-weight:800;color:{bar_color};margin:0;">'
                f'{val:.1f}<span style="font-size:0.7rem;color:var(--text-secondary);">/{max_val}</span></p>'
                f'<div class="component-bar-bg">'
                f'<div class="component-bar-fill" style="width:{pct:.0f}%;background:{bar_color};"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3: Probability distribution ──────────────────────
    with st.expander("📈 Class Probability Distribution", expanded=True):
        import plotly.graph_objects as go  # type: ignore[import]

        sorted_labels = sorted(prob_dict.keys(), key=lambda k: prob_dict[k], reverse=True)
        sorted_probs = [prob_dict[l] * 100 for l in sorted_labels]
        bar_colors = [CLASS_COLORS.get(l, '#8b949e') for l in sorted_labels]

        fig = go.Figure(
            go.Bar(
                x=[l.capitalize() for l in sorted_labels],
                y=sorted_probs,
                marker_color=bar_colors,
                marker_line=dict(width=0),
                text=[f'{p:.1f}%' for p in sorted_probs],
                textposition='outside',
                textfont=dict(family='Inter', size=13, color='#e6edf3'),
            )
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#e6edf3'),
            yaxis=dict(
                title='Probability (%)',
                range=[0, max(sorted_probs) * 1.25],
                gridcolor='rgba(255,255,255,0.06)',
                zeroline=False,
            ),
            xaxis=dict(
                title='',
                tickfont=dict(size=13, color='#e6edf3'),
            ),
            margin=dict(t=20, b=40, l=50, r=20),
            height=320,
            bargap=0.35,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"prob_chart_{hash(email_text) % 100000000}")

    # ── Row 4: Explanations ──────────────────────────────────
    with st.expander("🔍 Detailed Explanation", expanded=True):
        reasons = explanations.get('reasons', [])
        if reasons:
            for reason in reasons:
                st.markdown(
                    f'<div class="explanation-item">{reason}</div>',
                    unsafe_allow_html=True,
                )

        # URL warnings
        url_warns = explanations.get('url_warnings', [])
        if url_warns:
            st.markdown(
                '<p style="font-weight:600;margin-top:1rem;margin-bottom:0.3rem;">'
                '🔗 URL Warnings</p>',
                unsafe_allow_html=True,
            )
            for w in url_warns:
                st.markdown(f'<div class="warning-item">{w}</div>', unsafe_allow_html=True)

        # Attachment warnings
        attach_warns = explanations.get('attachment_warnings', [])
        if attach_warns:
            st.markdown(
                '<p style="font-weight:600;margin-top:1rem;margin-bottom:0.3rem;">'
                '📎 Attachment Warnings</p>',
                unsafe_allow_html=True,
            )
            for w in attach_warns:
                st.markdown(f'<div class="warning-item">{w}</div>', unsafe_allow_html=True)

        # Sender warnings
        sender_warns = explanations.get('sender_warnings', [])
        if sender_warns:
            st.markdown(
                '<p style="font-weight:600;margin-top:1rem;margin-bottom:0.3rem;">'
                '🔍 Sender Anomalies</p>',
                unsafe_allow_html=True,
            )
            for w in sender_warns:
                st.markdown(f'<div class="warning-item">{w}</div>', unsafe_allow_html=True)

        # Threat indicators
        indicators = explanations.get('threat_indicators', {})
        if indicators:
            st.markdown(
                '<p style="font-weight:600;margin-top:1rem;margin-bottom:0.3rem;">'
                '🚨 Threat Indicator Matches</p>',
                unsafe_allow_html=True,
            )
            for cat, keywords in indicators.items():
                cat_label = cat.replace('_', ' ').title()
                kw_list = ', '.join(keywords)
                st.markdown(
                    f'<div class="warning-item">'
                    f'<strong>{cat_label}:</strong> {kw_list}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ── Tab 1: Single Email Analyzer ─────────────────────────────
with tab1:
    default_text = ''
    if selected_sample and selected_sample != '— Select a sample —':
        default_text = SAMPLE_EMAILS[selected_sample]

    email_input = st.text_area(
        "📩 Paste or type an email to analyze",
        value=default_text,
        height=180,
        placeholder="Paste the email body here…",
        key="email_input_area_tab1",
    )

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyze_clicked = st.button("🔍 Analyze Email", use_container_width=True, type="primary", key="btn_analyze_tab1")

    if analyze_clicked and email_input.strip():
        with st.spinner("Analyzing email…"):
            p_class, conf, p_dict, r_score, expl, is_fb = run_analysis_pipeline(email_input)
            render_analysis_results(email_input, p_class, conf, p_dict, r_score, expl, is_fb)
    elif analyze_clicked:
        st.warning("Please enter an email to analyze.", icon="📩")

# ── Tab 2: Active Inbox Scanner ──────────────────────────────
with tab2:
    st.markdown("### 📬 Scan your Real Inbox (IMAP)")
    st.markdown("Connect your email client to scan recent incoming unread emails for threats.")
    
    col_imap, col_port = st.columns([3, 1])
    with col_imap:
        imap_presets = {
            "Gmail (imap.gmail.com)": "imap.gmail.com",
            "Outlook (outlook.office365.com)": "outlook.office365.com",
            "Yahoo Mail (imap.mail.yahoo.com)": "imap.mail.yahoo.com",
            "Custom IMAP Server": ""
        }
        selected_preset = st.selectbox("IMAP Server Provider", list(imap_presets.keys()))
        imap_server = imap_presets[selected_preset]
        if not imap_server:
            imap_server = st.text_input("Enter custom IMAP Server (e.g. imap.example.com)")
            
    with col_port:
        imap_port = st.number_input("IMAP SSL Port", value=993, min_value=1, max_value=65535)

    col_user, col_pass = st.columns(2)
    with col_user:
        email_user = st.text_input("Email Address", placeholder="yourname@gmail.com")
    with col_pass:
        email_pass = st.text_input("Password / App Password", type="password", 
                                  help="For Gmail, enable 2FA and use an 'App Password' from Google account security page.")

    fetch_limit = st.slider("Number of emails to fetch", min_value=1, max_value=20, value=5)

    col_scan_btn, _ = st.columns([1, 4])
    with col_scan_btn:
        scan_clicked = st.button("📥 Connect & Scan Inbox", use_container_width=True, type="primary", key="btn_scan_inbox")

    if scan_clicked:
        if not email_user or not email_pass or not imap_server:
            st.error("Please fill in all server and connection credentials.")
        else:
            with st.spinner("Connecting to mail server and downloading messages..."):
                import imaplib
                import email
                from email.header import decode_header
                
                try:
                    # Establish IMAP Connection
                    mail = imaplib.IMAP4_SSL(imap_server, imap_port)
                    mail.login(email_user, email_pass)
                    mail.select("inbox")
                    
                    # Search messages
                    status, messages = mail.search(None, "ALL")
                    if status == "OK" and messages[0]:
                        mail_ids = messages[0].split()
                        total_emails = len(mail_ids)
                        
                        st.success(f"Successfully connected! Found {total_emails} total emails in Inbox. Fetching last {fetch_limit}...", icon="✅")
                        
                        # Fetch and analyze loop
                        results_list = []
                        start_idx = max(0, total_emails - fetch_limit)
                        
                        for i in range(total_emails - 1, start_idx - 1, -1):
                            mail_id = mail_ids[i]
                            res, msg_data = mail.fetch(mail_id, "(RFC822)")
                            
                            for response_part in msg_data:
                                if isinstance(response_part, tuple):
                                    msg = email.message_from_bytes(response_part[1])
                                    
                                    # Parse Subject
                                    subj_val = msg["Subject"] or "[No Subject]"
                                    decoded_subj = ""
                                    for part, enc in decode_header(subj_val):
                                        if isinstance(part, bytes):
                                            try:
                                                decoded_subj += part.decode(enc or "utf-8", errors="ignore")
                                            except Exception:
                                                decoded_subj += str(part)
                                        else:
                                            decoded_subj += str(part)
                                            
                                    # Parse Sender
                                    from_sender = msg.get("From", "[Unknown Sender]")
                                    for part, enc in decode_header(from_sender):
                                        if isinstance(part, bytes):
                                            from_sender = part.decode(enc or "utf-8", errors="ignore")
                                            
                                    # Parse Date
                                    date_sent = msg.get("Date", "[Unknown Date]")
                                    
                                    # Parse Plaintext Body
                                    body_text = ""
                                    if msg.is_multipart():
                                        for part in msg.walk():
                                            c_type = part.get_content_type()
                                            c_disp = str(part.get("Content-Disposition"))
                                            if c_type == "text/plain" and "attachment" not in c_disp:
                                                payload = part.get_payload(decode=True)
                                                if payload:
                                                    body_text = payload.decode(errors="ignore")
                                                break
                                    else:
                                        payload = msg.get_payload(decode=True)
                                        if payload:
                                            body_text = payload.decode(errors="ignore")
                                            
                                    if not body_text.strip():
                                        body_text = decoded_subj # fallback to subject text for classification if body is empty HTML
                                        
                                    # Run Analyzer Pipeline
                                    p_class, conf, p_dict, r_score, expl, is_fb = run_analysis_pipeline(body_text)
                                    
                                    results_list.append({
                                        "Subject": decoded_subj,
                                        "Sender": from_sender,
                                        "Date": date_sent,
                                        "Class": f"{CLASS_EMOJIS.get(p_class, '📧')} {p_class.upper()}",
                                        "Risk": f"{r_score['emoji']} {r_score['score']} ({r_score['level']})",
                                        "body": body_text,
                                        "p_class": p_class,
                                        "conf": conf,
                                        "p_dict": p_dict,
                                        "r_score": r_score,
                                        "expl": expl,
                                        "is_fb": is_fb
                                    })
                        
                        mail.logout()
                        
                        # Display Results
                        if results_list:
                            # Save in session state to enable clicking
                            st.session_state['inbox_results'] = results_list
                            st.markdown("### 🔍 Scan Results Summary")
                            
                            # Render beautiful dashboard table
                            display_df = pd.DataFrame(results_list)[["Subject", "Sender", "Class", "Risk", "Date"]]
                            st.dataframe(display_df, use_container_width=True)
                            
                            st.markdown("---")
                            st.markdown("#### 🔬 Detailed Analysis Report Selector")
                            selected_index = st.selectbox(
                                "Select fetched email to view detailed threat explanation",
                                options=range(len(results_list)),
                                format_func=lambda idx: f"Email {idx+1}: {results_list[idx]['Subject'][:45]}... (Sender: {results_list[idx]['Sender'][:25]}...)"
                            )
                            
                            if selected_index is not None:
                                mail_info = results_list[selected_index]
                                st.markdown("##### 📝 Email Body:")
                                st.code(mail_info['body'], language='text')
                                
                                # Render reports
                                render_analysis_results(
                                    mail_info['body'],
                                    mail_info['p_class'],
                                    mail_info['conf'],
                                    mail_info['p_dict'],
                                    mail_info['r_score'],
                                    mail_info['expl'],
                                    mail_info['is_fb']
                                )
                        else:
                            st.warning("Connected but no emails could be retrieved.")
                            
                except Exception as e:
                    st.error(f"❌ Connection Failed: {e}. Make sure you are using a valid App Password (not your password) for Gmail, and IMAP SSL is enabled.")

# ── Footer ───────────────────────────────────────────────
st.markdown('<hr class="subtle-divider">', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align:center;font-size:0.78rem;color:var(--text-secondary);">'
    '🛡️ AI Email Risk Analyzer • Active Threat Protection • Built with Streamlit & Python'
    '</p>',
    unsafe_allow_html=True,
)
