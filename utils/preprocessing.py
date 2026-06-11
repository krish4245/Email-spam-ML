import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data on import safely
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}. Preprocessing might fall back to basic text cleaning.")

SUSPICIOUS_KEYWORDS = {
    'phishing': ['verify', 'account', 'suspended', 'password', 'bank', 'login', 'security', 'confirm', 'credentials', 'unauthorized', 'ssn', 'social security', 'identity', 'expire', 'locked', 'compromised', 'paypal', 'apple id'],
    'spam': ['winner', 'free', 'prize', 'congratulations', 'lottery', 'cash', 'million', 'cheap', 'buy now', 'limited time', 'act now', 'no cost', 'guarantee', 'incredible deal', 'weight loss'],
    'suspicious': ['wire transfer', 'confidential', 'urgent request', 'attachment', 'bitcoin', 'cryptocurrency', 'secret', 'offshore', 'investment opportunity', 'classified'],
    'urgency': ['urgent', 'immediately', 'now', 'asap', 'hurry', 'deadline', 'expire', 'final notice', 'last chance', 'warning', 'alert']
}

def remove_urls(text):
    """Remove URLs from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)

def remove_emails(text):
    """Remove email addresses from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'\S+@\S+', ' ', text)

def remove_html_tags(text):
    """Remove HTML tags from text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'<.*?>', ' ', text)

def remove_punctuation(text):
    """Remove punctuation, keeping spaces."""
    if not isinstance(text, str):
        return ""
    # Create translation table to replace punctuation with space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)

def remove_stopwords(text):
    """Remove NLTK stopwords from text."""
    if not isinstance(text, str):
        return ""
    try:
        stop_words = set(stopwords.words('english'))
    except Exception:
        # Fallback if NLTK is not downloaded
        stop_words = {'the', 'is', 'are', 'at', 'to', 'for', 'in', 'on', 'of', 'and', 'a', 'an', 'this', 'that', 'it', 'you', 'your', 'we'}
    
    words = text.split()
    filtered_words = [w for w in words if w.lower() not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """Lemmatize words in text to their base form."""
    if not isinstance(text, str):
        return ""
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(w) for w in words]
        return ' '.join(lemmatized)
    except Exception:
        # Fallback if WordNet is not downloaded
        return text

def clean_text(text) -> str:
    """Clean the text using the full preprocessing pipeline."""
    if not isinstance(text, str):
        return ""
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def count_urls(text) -> int:
    """Count number of URLs in raw text."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'https?://\S+|www\.\S+', text))

def count_email_addresses(text) -> int:
    """Count number of email addresses in raw text."""
    if not isinstance(text, str):
        return 0
    return len(re.findall(r'\S+@\S+', text))

def detect_url_shorteners(text) -> bool:
    """Check if any common URL shorteners are present in text."""
    if not isinstance(text, str):
        return False
    shorteners = [r'bit\.ly', r'tinyurl\.com', r'goo\.gl', r't\.co', r'ow\.ly', r'is\.gd', r'buff\.ly']
    pattern = '|'.join(shorteners)
    return bool(re.search(pattern, text, re.IGNORECASE))

def count_suspicious_words(text) -> int:
    """Count occurrences of suspicious words in text."""
    if not isinstance(text, str):
        return 0
    text_lower = text.lower()
    count = 0
    # Combine all classes' keywords
    all_keywords = []
    for k_list in SUSPICIOUS_KEYWORDS.values():
        all_keywords.extend(k_list)
        
    for word in all_keywords:
        # Check as whole word or substring depending on length
        if len(word) > 4:
            count += len(re.findall(rf'\b{re.escape(word)}', text_lower))
        else:
            count += len(re.findall(rf'\b{re.escape(word)}\b', text_lower))
    return count

def extract_meta_features(text) -> dict:
    """Extract handcrafted meta features from raw email text."""
    if not isinstance(text, str):
        text = ""
        
    email_len = len(text)
    words = text.split()
    word_cnt = len(words)
    
    url_cnt = count_urls(text)
    email_cnt = count_email_addresses(text)
    
    # Capital letter ratio
    caps_cnt = sum(1 for c in text if c.isupper())
    cap_ratio = caps_cnt / email_len if email_len > 0 else 0.0
    
    # Special character count
    special_chars = sum(1 for c in text if c in string.punctuation)
    
    # Suspect keywords count
    susp_word_cnt = count_suspicious_words(text)
    
    # Specific flags
    text_lower = text.lower()
    
    has_urg = any(word in text_lower for word in SUSPICIOUS_KEYWORDS['urgency'])
    has_money = any(word in text_lower for word in ['free', 'prize', 'winner', 'cash', 'lottery', 'million', '$', '€', '£'])
    has_shortener = detect_url_shorteners(text)
    
    return {
        'email_length': email_len,
        'word_count': word_cnt,
        'url_count': url_cnt,
        'email_addr_count': email_cnt,
        'special_char_count': special_chars,
        'capital_ratio': cap_ratio,
        'suspicious_word_count': susp_word_cnt,
        'has_urgency': int(has_urg),
        'has_money_ref': int(has_money),
        'has_url_shortener': int(has_shortener)
    }
