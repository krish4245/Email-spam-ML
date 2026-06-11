"""
Explainable AI Module for Email Risk Analyzer.

Provides human-readable explanations for email classification decisions
by detecting phishing indicators, spam patterns, suspicious keywords,
URL anomalies, dangerous attachments, and sender spoofing signals.
"""

import re

# ──────────────────────────────────────────────────────────────
# Indicator dictionaries
# ──────────────────────────────────────────────────────────────

PHISHING_INDICATORS = {
    'account_request': [
        'account', 'verify', 'confirm', 'validate', 'update your',
    ],
    'credentials': [
        'password', 'login', 'credentials', 'ssn', 'social security',
        'credit card',
    ],
    'urgency': [
        'immediately', 'urgent', 'asap', 'right away', 'within 24 hours',
        'suspended', 'locked', 'expire',
    ],
    'threats': [
        'unauthorized', 'compromised', 'breach', 'fraud', 'illegal',
        'terminated',
    ],
    'brand_impersonation': [
        'paypal', 'apple', 'microsoft', 'amazon', 'netflix',
        'bank of america', 'wells fargo', 'chase',
    ],
    'action_request': [
        'click here', 'click below', 'click the link', 'download',
        'open attachment', 'sign in',
    ],
}

SPAM_INDICATORS = {
    'prize_scam': [
        'winner', 'congratulations', 'selected', 'lottery', 'prize',
        'jackpot',
    ],
    'money_lure': [
        'free', 'no cost', 'cash', 'million dollars', 'earn money',
        'make money',
    ],
    'pressure': [
        'limited time', 'act now', "don't miss", 'exclusive',
        'special offer', 'only today',
    ],
    'health_scam': [
        'weight loss', 'miracle', 'cure', 'pill', 'supplement', 'diet',
    ],
    'too_good': [
        'guaranteed', 'risk free', 'no obligation', 'incredible deal',
        'lowest price',
    ],
}

SUSPICIOUS_INDICATORS = {
    'financial_request': [
        'wire transfer', 'bank transfer', 'western union', 'bitcoin',
        'cryptocurrency',
    ],
    'secrecy': [
        'confidential', 'secret', "don't tell", 'private matter',
        'between us',
    ],
    'social_engineering': [
        'trust me', 'help me', 'personal favor', 'do me a favor',
    ],
    'unusual_sender': [
        'prince', 'diplomat', 'minister', 'barrister', 'beneficiary',
    ],
}

# Human-readable labels for each indicator category
_CATEGORY_LABELS = {
    # Phishing
    'account_request': 'Account-related request detected',
    'credentials': 'Credential / sensitive-data solicitation detected',
    'urgency': 'Urgency language found',
    'threats': 'Threatening language detected',
    'brand_impersonation': 'Possible brand impersonation',
    'action_request': 'Suspicious call-to-action detected',
    # Spam
    'prize_scam': 'Prize / lottery scam language detected',
    'money_lure': 'Money-lure language detected',
    'pressure': 'High-pressure sales language detected',
    'health_scam': 'Health-scam language detected',
    'too_good': '"Too good to be true" language detected',
    # Suspicious
    'financial_request': 'Financial transfer request detected',
    'secrecy': 'Secrecy / confidentiality request detected',
    'social_engineering': 'Social-engineering language detected',
    'unusual_sender': 'Unusual sender persona detected',
}

# ──────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────

def _match_indicators(text: str, indicator_dict: dict) -> dict[str, list[str]]:
    """Return ``{category: [matched_keywords]}`` for every category with ≥1 hit."""
    text_lower = text.lower()
    matches: dict[str, list[str]] = {}
    for category, keywords in indicator_dict.items():
        found = [kw for kw in keywords if kw in text_lower]
        if found:
            matches[category] = found
    return matches


# ──────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────

def explain_prediction(text: str, predicted_class: str, confidence: float) -> list[str]:
    """Generate human-readable explanations for why an email was classified.

    Parameters
    ----------
    text : str
        The raw or preprocessed email body.
    predicted_class : str
        One of ``'safe'``, ``'spam'``, ``'phishing'``, ``'promotion'``, ``'suspicious'``.
    confidence : float
        Model confidence in ``[0, 1]``.

    Returns
    -------
    list[str]
        Explanation strings prefixed with a ✓ icon.
    """
    explanations: list[str] = []
    predicted_lower = predicted_class.lower()

    # Always note the model confidence
    explanations.append(
        f"✓ Model confidence: {confidence * 100:.1f}% for class '{predicted_class}'"
    )

    # Check indicators relevant to the predicted class
    if predicted_lower == 'phishing':
        matches = _match_indicators(text, PHISHING_INDICATORS)
        for cat, keywords in matches.items():
            explanations.append(f"✓ {_CATEGORY_LABELS[cat]} — keywords: {', '.join(keywords)}")
        if not matches:
            explanations.append("✓ Phishing signals detected by ML model (no explicit keyword match)")

    elif predicted_lower == 'spam':
        matches = _match_indicators(text, SPAM_INDICATORS)
        for cat, keywords in matches.items():
            explanations.append(f"✓ {_CATEGORY_LABELS[cat]} — keywords: {', '.join(keywords)}")
        if not matches:
            explanations.append("✓ Spam patterns detected by ML model (no explicit keyword match)")

    elif predicted_lower == 'suspicious':
        matches = _match_indicators(text, SUSPICIOUS_INDICATORS)
        for cat, keywords in matches.items():
            explanations.append(f"✓ {_CATEGORY_LABELS[cat]} — keywords: {', '.join(keywords)}")
        if not matches:
            explanations.append("✓ Suspicious patterns detected by ML model (no explicit keyword match)")

    elif predicted_lower == 'promotion':
        # Promotions may share some spam-like pressure words
        matches = _match_indicators(text, SPAM_INDICATORS)
        for cat, keywords in matches.items():
            explanations.append(f"✓ Promotional language: {_CATEGORY_LABELS[cat]} — keywords: {', '.join(keywords)}")
        if not matches:
            explanations.append("✓ Promotional content detected by ML model")

    elif predicted_lower == 'safe':
        explanations.append("✓ No significant threat indicators detected")
        # Cross-check against all dictionaries for minor hits
        all_dicts = {**PHISHING_INDICATORS, **SPAM_INDICATORS, **SUSPICIOUS_INDICATORS}
        minor = _match_indicators(text, all_dicts)
        if minor:
            matched_kws = [kw for kws in minor.values() for kw in kws]
            explanations.append(
                f"✓ Minor keyword overlap ({', '.join(matched_kws[:5])}) — "
                "context indicates safe content"
            )

    return explanations


def detect_url_patterns(text: str) -> list[str]:
    """Detect suspicious URL patterns in the email text.

    Checks for:
    - URL shorteners (bit.ly, tinyurl, t.co, goo.gl, ow.ly, is.gd, buff.ly)
    - IP-based URLs (``http://192.168.x.x/…``)
    - Suspicious TLDs (``.ru``, ``.cn``, ``.tk``, ``.xyz``, ``.top``, ``.buzz``)
    - Plain HTTP (non-HTTPS) links
    - Excessive URL count (> 3)

    Returns
    -------
    list[str]
        Warning strings.
    """
    warnings: list[str] = []
    text_lower = text.lower()

    # Extract all URLs
    urls = re.findall(r'https?://[^\s<>"\']+', text_lower)

    if not urls:
        return warnings

    # URL shorteners
    shorteners = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'is.gd', 'buff.ly', 'rebrand.ly']
    for url in urls:
        for shortener in shorteners:
            if shortener in url:
                warnings.append(f"⚠ URL shortener detected ({shortener}) — may hide true destination")
                break

    # IP-based URLs
    ip_pattern = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    if ip_pattern.search(text_lower):
        warnings.append("⚠ IP-based URL detected — legitimate sites rarely use raw IPs")

    # Suspicious TLDs
    suspicious_tlds = ['.ru', '.cn', '.tk', '.xyz', '.top', '.buzz', '.club', '.work']
    for url in urls:
        for tld in suspicious_tlds:
            if url.rstrip('/').endswith(tld) or f'{tld}/' in url:
                warnings.append(f"⚠ Suspicious TLD detected ({tld}) in URL")
                break

    # HTTP-only (no HTTPS)
    http_only = [u for u in urls if u.startswith('http://')]
    if http_only:
        warnings.append(f"⚠ Non-secure HTTP link(s) found ({len(http_only)}) — legitimate sites use HTTPS")

    # Too many URLs
    if len(urls) > 3:
        warnings.append(f"⚠ High URL count ({len(urls)}) — may indicate phishing or spam")

    return warnings


def detect_attachment_risks(text: str) -> list[str]:
    """Detect mentions of dangerous file attachments in the email body.

    Flags references to: ``.exe``, ``.bat``, ``.scr``, ``.zip``, ``.rar``,
    ``.js``, ``.vbs``, ``.msi``, ``.cmd``, ``.pif``, ``.com``.

    Returns
    -------
    list[str]
        Warning strings for each detected dangerous extension.
    """
    warnings: list[str] = []
    text_lower = text.lower()

    dangerous_extensions = {
        '.exe': 'Executable file',
        '.bat': 'Batch script',
        '.scr': 'Screensaver (executable)',
        '.zip': 'Compressed archive (may contain malware)',
        '.rar': 'Compressed archive (may contain malware)',
        '.js': 'JavaScript file',
        '.vbs': 'VBScript file',
        '.msi': 'Windows installer',
        '.cmd': 'Command script',
        '.pif': 'Program Information File (executable)',
        '.com': 'DOS executable',
    }

    for ext, description in dangerous_extensions.items():
        # Match the extension when preceded by a word character (filename-like)
        pattern = re.compile(rf'\w{re.escape(ext)}(?:\s|$|[,;)\]\}}])', re.IGNORECASE)
        if pattern.search(text_lower):
            warnings.append(f"📎 Dangerous attachment type mentioned: {ext} — {description}")

    return warnings


def detect_sender_anomalies(text: str) -> list[str]:
    """Detect spoofed or suspicious sender patterns in the email text.

    Checks for:
    - Brand names embedded in random domains (``amazon-support@xyz.ru``)
    - Numeric-heavy sender names (``user38291@…``)
    - Suspicious TLDs in email addresses
    - Mismatched display-name vs. address domain

    Returns
    -------
    list[str]
        Warning strings.
    """
    warnings: list[str] = []
    text_lower = text.lower()

    # Extract email-like patterns
    emails = re.findall(r'[\w.+-]+@[\w.-]+\.\w+', text_lower)

    brands = ['paypal', 'apple', 'microsoft', 'amazon', 'netflix', 'google',
              'facebook', 'instagram', 'chase', 'wellsfargo', 'bankofamerica']

    suspicious_tlds = ['.ru', '.cn', '.tk', '.xyz', '.top', '.buzz']

    for email in emails:
        local_part, domain = email.rsplit('@', 1)

        # Brand in non-official domain
        for brand in brands:
            if brand in local_part and brand not in domain:
                warnings.append(
                    f"🔍 Possible brand spoofing: '{brand}' in sender name but domain is '{domain}'"
                )

        # Numeric-heavy local part (>50% digits)
        digit_ratio = sum(c.isdigit() for c in local_part) / max(len(local_part), 1)
        if digit_ratio > 0.5 and len(local_part) > 4:
            warnings.append(f"🔍 Numeric sender name detected ({email}) — often auto-generated")

        # Suspicious TLD
        for tld in suspicious_tlds:
            if domain.endswith(tld.lstrip('.')):
                warnings.append(f"🔍 Sender uses suspicious TLD: {domain}")
                break

    return warnings


def get_all_explanations(text: str, predicted_class: str, confidence: float) -> dict:
    """Aggregate all explanation signals into a single dictionary.

    Parameters
    ----------
    text : str
        Raw email body.
    predicted_class : str
        Predicted label.
    confidence : float
        Model confidence in ``[0, 1]``.

    Returns
    -------
    dict
        ``{
            'reasons': [str, ...],
            'threat_indicators': {category: [keywords]},
            'url_warnings': [str, ...],
            'attachment_warnings': [str, ...],
            'sender_warnings': [str, ...],
        }``
    """
    # Gather threat indicators across all dictionaries
    threat_indicators: dict[str, list[str]] = {}
    threat_indicators.update(_match_indicators(text, PHISHING_INDICATORS))
    threat_indicators.update(_match_indicators(text, SPAM_INDICATORS))
    threat_indicators.update(_match_indicators(text, SUSPICIOUS_INDICATORS))

    return {
        'reasons': explain_prediction(text, predicted_class, confidence),
        'threat_indicators': threat_indicators,
        'url_warnings': detect_url_patterns(text),
        'attachment_warnings': detect_attachment_risks(text),
        'sender_warnings': detect_sender_anomalies(text),
    }
