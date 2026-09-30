"""Feature extraction: URL + keyword signals pulled out of raw email text."""
import re
import numpy as np

URL_RE = re.compile(r'(https?://[^\s<>"\')]+|www\.[^\s<>"\')]+)', re.I)
IP_URL_RE = re.compile(r'https?://\d{1,3}(?:\.\d{1,3}){3}')
SHORTENERS = ("bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly")

KEYWORDS = [
    "urgent", "verify", "suspended", "password", "account", "login", "confirm",
    "click", "bank", "security", "update", "limited", "immediately", "winner",
    "prize", "free", "invoice", "payment", "unusual", "expire", "locked",
    "credentials", "ssn", "refund", "gift card",
]

FEATURE_NAMES = [
    "num_urls", "has_ip_url", "max_url_len", "num_at_in_urls", "num_hyphens_in_urls",
    "num_shorteners", "num_http_only", "num_subdomain_dots", "keyword_hits",
    "num_exclaims", "upper_ratio", "text_len",
]


def extract_urls(text: str):
    return [u.rstrip(".,;") for u in URL_RE.findall(text or "")]


def keyword_hits(text: str):
    low = (text or "").lower()
    return [k for k in KEYWORDS if k in low]


def _one(text: str):
    text = text or ""
    urls = extract_urls(text)
    hosts = [re.sub(r"^https?://", "", u, flags=re.I).split("/")[0] for u in urls]
    return [
        len(urls),
        1 if IP_URL_RE.search(text) else 0,
        max((len(u) for u in urls), default=0),
        sum(u.count("@") for u in urls),
        sum(h.count("-") for h in hosts),
        sum(any(s in h.lower() for s in SHORTENERS) for h in hosts),
        sum(u.lower().startswith("http://") for u in urls),
        max((h.count(".") for h in hosts), default=0),
        len(keyword_hits(text)),
        text.count("!"),
        sum(c.isupper() for c in text) / max(len(text), 1),
        len(text),
    ]


def extract_features(texts):
    """Used inside the sklearn pipeline (FunctionTransformer)."""
    return np.array([_one(t) for t in texts], dtype=float)
