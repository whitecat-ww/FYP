# src/app.py
from flask import Flask, render_template, request
import joblib
from feature_extractor import extract_features, features_to_vector, FEATURE_ORDER
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "../templates"), static_folder=os.path.join(os.path.dirname(__file__), "../static"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/phishing_rf.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found. Train the model first by running: python src/train.py")

m = joblib.load(MODEL_PATH)
clf = m['model']
feat_order = m['feature_order']

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    details = {}
    if request.method == "POST":
        url = request.form.get("url")
        if url:
            feats = extract_features(url)
            vec = features_to_vector(feats)
            pred = clf.predict([vec])[0]
            prob = clf.predict_proba([vec])[0].max() if hasattr(clf, 'predict_proba') else None

            if pred == 1:
                label = "PHISHING"
            else:
                label = "LEGITIMATE"

            result = {
                'label': label,
                'probability': float(prob) if prob is not None else None
            }
            # include features for explainability
            details = {k: feats.get(k) for k in feat_order}
    return render_template("index.html", result=result, details=details)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
