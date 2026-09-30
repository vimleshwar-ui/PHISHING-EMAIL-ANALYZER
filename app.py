"""Flask website: paste an email, get Phishing / Safe + model metrics."""
import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from features import extract_urls, keyword_hits, extract_features  # noqa: F401 (needed to unpickle)

BASE = Path(__file__).parent
app = Flask(__name__)

model = joblib.load(BASE / "model.joblib")
metrics = json.loads((BASE / "metrics.json").read_text())


def classify(text: str):
    proba = model.predict_proba(pd.DataFrame({"text": [text]}))[0]
    classes = list(model.classes_)
    p_phish = float(proba[classes.index("Phishing")])
    return {
        "label": "Phishing" if p_phish >= 0.5 else "Safe",
        "phishing_probability": round(p_phish * 100, 1),
        "urls": extract_urls(text),
        "keywords": keyword_hits(text),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    text, result = "", None
    if request.method == "POST":
        text = request.form.get("email", "").strip()
        if text:
            result = classify(text)
    return render_template("index.html", text=text, result=result, metrics=metrics)


@app.post("/api/predict")
def api_predict():
    text = (request.get_json(silent=True) or {}).get("email", "").strip()
    if not text:
        return jsonify(error="Send JSON like {\"email\": \"...\"}"), 400
    return jsonify(classify(text))


if __name__ == "__main__":
    app.run(debug=True)
