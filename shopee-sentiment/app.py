"""
=============================================================
 ANALISIS SENTIMEN ULASAN E-COMMERCE SHOPEE
 Menggunakan Metode Naive Bayes Classifier
=============================================================
 Author  : [Kelompok 3]
 Dataset : Shopee Product Reviews (Kaggle)
 Model   : Multinomial Naive Bayes
=============================================================
"""

from flask import Flask, render_template, request, jsonify
import joblib
import os
import json

# Import modul analisis
from ml_model import (
    load_or_train_model,
    predict_sentiment,
    get_model_metrics,
    preprocess_text,
    get_wordcloud_data,
)

app = Flask(__name__)

# ── Load / train model saat aplikasi dimulai ──────────────
print("🔄  Memuat model Naive Bayes ...")
model, vectorizer, label_encoder = load_or_train_model()
metrics = get_model_metrics()
print("✅  Model siap digunakan!")


# ─────────────────────────────────────────────
#  ROUTE: Halaman Utama
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", metrics=metrics)


# ─────────────────────────────────────────────
#  ROUTE: Prediksi Sentimen
# ─────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    teks = data.get("text", "").strip()

    if not teks:
        return jsonify({"error": "Teks tidak boleh kosong!"}), 400

    result = predict_sentiment(model, vectorizer, label_encoder, teks)
    return jsonify(result)


# ─────────────────────────────────────────────
#  ROUTE: Data Metrik Model
# ─────────────────────────────────────────────
@app.route("/metrics")
def get_metrics():
    return jsonify(metrics)


# ─────────────────────────────────────────────
#  ROUTE: Data Word Cloud
# ─────────────────────────────────────────────
@app.route("/wordcloud/<sentiment>")
def wordcloud(sentiment):
    words = get_wordcloud_data(sentiment)
    return jsonify(words)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🛒  Shopee Sentiment Analyzer - Naive Bayes")
    print("  🌐  Buka browser: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host='127.0.0.1', port=8000, debug=False)