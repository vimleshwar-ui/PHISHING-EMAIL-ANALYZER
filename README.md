# Phishing Email Detection (Scikit-learn + Flask)

1. Install: `pip install -r requirements.txt`
2. (Recommended) Download a real dataset, e.g. the Kaggle "Phishing Email Dataset",
   and save it as `data/emails.csv` (needs a text column and a label column).
3. Train: `python train_model.py`  -> prints accuracy + confusion matrix, saves model.joblib
4. Run the site: `python app.py`  -> open http://127.0.0.1:5000

API: `POST /api/predict` with JSON `{"email": "..."}`
