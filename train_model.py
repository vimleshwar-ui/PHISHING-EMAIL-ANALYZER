"""Train the phishing email classifier.

Usage:
    python train_model.py                      # uses data/emails.csv if present, else demo data
    python train_model.py --csv path/to.csv    # CSV with a text column and a label column
"""
import argparse, json, random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from features import extract_features

BASE = Path(__file__).parent
MODEL_PATH = BASE / "model.joblib"
METRICS_PATH = BASE / "metrics.json"
CM_PATH = BASE / "static" / "confusion_matrix.png"

TEXT_COLS = ["text", "email text", "email_text", "body", "message", "content", "email"]
LABEL_COLS = ["label", "email type", "email_type", "class", "target", "type"]
PHISH_WORDS = {"phishing", "phishing email", "spam", "1", "malicious"}


# ---------- data ----------
def make_demo_data(n=600, seed=42):
    """Synthetic fallback so the project runs out of the box. Replace with a real dataset!"""
    rnd = random.Random(seed)
    brands = ["PayPal", "Amazon", "Netflix", "Microsoft", "HDFC Bank", "Apple", "SBI"]
    bad_hosts = ["secure-{b}-login.com", "{b}-verify-account.net", "update-{b}.info",
                 "192.168.44.{n}", "bit.ly/{n}xQ", "{b}.account-check.ru"]
    phish = [
        "URGENT: Your {B} account has been suspended! Verify your password immediately: http://{h}/login",
        "Dear customer, unusual sign-in detected. Confirm your credentials now or your account will be locked: http://{h}",
        "Congratulations! You are a winner of a free gift card. Click here to claim your prize: http://{h}/claim",
        "Payment failed for your {B} invoice. Update your billing details within 24 hours: http://{h}/pay",
        "Security alert!!! Your {B} account will expire. Click http://{h} to verify and avoid limited access.",
        "Refund pending. Provide your bank details and SSN to receive your refund: http://{h}/refund",
    ]
    good_hosts = ["github.com/team/repo/pull/{n}", "docs.google.com/document/d/{n}",
                  "www.company.com/newsletter", "zoom.us/j/{n}", "www.wikipedia.org/wiki/Python"]
    safe = [
        "Hi team, the meeting notes from today are attached. Let's sync again on Thursday. Notes: https://{h}",
        "Thanks for your help with the report. Here is the draft for review: https://{h}. Let me know your comments.",
        "Reminder: the quarterly planning call is tomorrow at 10am. Agenda is here: https://{h}",
        "Hey, are we still on for lunch on Friday? Also, I shared the photos from the trip with you.",
        "Your order has shipped and will arrive on Monday. Track it from your account on our website.",
        "Weekly newsletter: new articles on data science and machine learning. Read online: https://{h}",
    ]
    rows = []
    for i in range(n):
        if i % 2 == 0:
            b = rnd.choice(brands)
            h = rnd.choice(bad_hosts).format(b=b.lower().replace(" ", ""), n=rnd.randint(2, 250))
            rows.append((rnd.choice(phish).format(B=b, h=h), "Phishing"))
        else:
            h = rnd.choice(good_hosts).format(n=rnd.randint(1000, 99999))
            rows.append((rnd.choice(safe).format(h=h), "Safe"))
    return pd.DataFrame(rows, columns=["text", "label"])


def load_data(csv_path=None):
    path = Path(csv_path) if csv_path else BASE / "data" / "emails.csv"
    if not path.exists():
        print("No dataset found -> using built-in demo data (put a real CSV in data/emails.csv).")
        return make_demo_data(), "demo"
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next(cols[c] for c in TEXT_COLS if c in cols)
    lcol = next(cols[c] for c in LABEL_COLS if c in cols)
    df = df[[tcol, lcol]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].astype(str).str.lower().str.strip().map(
        lambda v: "Phishing" if v in PHISH_WORDS else "Safe")
    print(f"Loaded {len(df)} emails from {path}")
    return df, path.name


# ---------- model ----------
def build_pipeline():
    features = ColumnTransformer([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english",
                                  ngram_range=(1, 2), min_df=2, max_features=20000), "text"),
        ("url_signals", Pipeline([("extract", FunctionTransformer(extract_features)),
                                  ("scale", StandardScaler())]), "text"),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    args = ap.parse_args()

    df, source = load_data(args.csv)
    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    model = build_pipeline().fit(X_train, y_train)
    pred = model.predict(X_test)

    labels = ["Safe", "Phishing"]
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=labels)
    print(f"\nAccuracy: {acc:.4f}\n")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=[f"actual {l}" for l in labels], columns=[f"pred {l}" for l in labels]))
    print("\n", classification_report(y_test, pred, digits=4))

    CM_PATH.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    fig.savefig(CM_PATH, dpi=150)

    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "accuracy": round(acc, 4), "confusion_matrix": cm.tolist(), "labels": labels,
        "train_size": len(X_train), "test_size": len(X_test), "source": source,
    }, indent=2))
    print(f"Saved model -> {MODEL_PATH.name}")


if __name__ == "__main__":
    main()
