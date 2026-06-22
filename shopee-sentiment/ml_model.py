"""
=============================================================
 ml_model.py  –  Modul Machine Learning Naive Bayes
=============================================================
 Berisi:
   • Preprocessing teks (cleaning, stopwords, stemming)
   • Load dataset Shopee dari Kaggle
   • Training Multinomial Naive Bayes
   • Evaluasi model (accuracy, precision, recall, f1)
   • Fungsi prediksi sentimen
=============================================================
"""

import os
from pyexpat import model
import re
import json
import pickle
from matplotlib.pyplot import text
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# Inisialisasi stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def simple_stemmer(word):
    try:
        return stemmer.stem(word)
    except:
        return word

# ── NLP & Machine Learning ────────────────────────────────
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ── Direktori ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "dataset")
METRIC_FILE = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR,  exist_ok=True)

# ── Stopwords Bahasa Indonesia (built-in, tanpa NLTK) ─────
STOPWORDS_ID = {

    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan", "untuk",
    "pada", "adalah", "dalam", "ada", "juga", "saya", "aku",
    "kamu", "dia", "kami", "kita", "mereka", "sudah",
    "akan", "harus", "jika", "kalau", "karena", "tapi", "atau",
    "tetapi", "namun", "memang", "sudahnya",
    "nya", "lah", "sih", "deh", "dong", "kan",
    "pun", "aja", "jadi", "juga", "lagi", "masih", "mau",
    "baik", "iya", "ya", "yah", "oh", "ah", "eh", "hm",
    "saja", "cuma", "hanya", "setiap", "semua", "semuanya",
    "beberapa", "sedikit", "cukup", "amat",
}


# ── Kamus Afiksasi Sederhana (tanpa PySastrawi) ───────────

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """
    Pipeline preprocessing teks ulasan:
      1. Lowercase
      2. Hapus URL, mention, hashtag
      3. Hapus karakter non-alfanumerik
      4. Hapus angka
      5. Tokenisasi
      6. Hapus stopwords
      7. Stemming sederhana
      8. Filter token pendek (< 3 karakter)
    """
    text = str(text).lower()
    # Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # Hapus mention & hashtag
    text = re.sub(r"[@#]\w+", " ", text)
    # Hapus karakter non-alfabet (termasuk emoji)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Normalisasi spasi
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS_ID]
    text = " ".join(tokens)
    text = stemmer.stem(text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) >= 3]

    return " ".join(tokens)


# ─────────────────────────────────────────────
#  GENERATE DATASET SINTETIS (Fallback)
# ─────────────────────────────────────────────
def generate_synthetic_dataset(n: int = 2000) -> pd.DataFrame:
    """
    Membuat dataset sintetis ulasan Shopee apabila dataset Kaggle
    belum tersedia.  Digunakan sebagai fallback saat demo.
    """
    positif = [
        "produk bagus sekali sesuai deskripsi pengiriman cepat",
        "barang original kualitas mantap seller terpercaya recommended",
        "packaging rapi aman sampai tidak ada kerusakan bintang lima",
        "harga murah kualitas premium puas banget belanja disini",
        "cepat sampai kondisi mulus sesuai gambar makasih seller",
        "produk berkualitas tinggi sesuai ekspektasi pengiriman kilat",
        "barang asli original sudah dicek terimakasih sudah amanah",
        "toko terpercaya fast respon seller baik hati banget",
        "barang sampai utuh kualitas bagus harga terjangkau mantap",
        "sangat puas dengan pembelian ini akan beli lagi nanti",
        "pengiriman super cepat barang sesuai foto tidak mengecewakan",
        "top seller kualitas terjamin barang mewah harga bersahabat",
        "cocok dengan yang saya cari kualitas bagus recommended deh",
        "produk premium harga worth it packing aman dan rapi",
        "seller responsif barang dikirim cepat kualitas tidak diragukan",
    ]
    negatif = [
        "barang tidak sesuai gambar sangat mengecewakan tidak mau beli lagi",
        "kualitas jelek cepat rusak harga mahal tidak sebanding sama sekali",
        "pengiriman sangat lambat sudah seminggu lebih belum sampai",
        "produk palsu tidak original seller bohong sangat kecewa",
        "barang datang rusak packaging buruk tidak ada bubble wrap",
        "seller tidak responsif chat tidak dibalas sangat mengecewakan",
        "kualitas bahan tipis murahan tidak sesuai ekspektasi sama sekali",
        "salah kirim ukuran beda minta retur seller tidak mau",
        "harga terlalu mahal untuk kualitas segini sangat tidak worth it",
        "barang tidak ada isinya kosong sudah bayar tapi tidak dapat apa",
        "foto produk menipu aslinya jauh berbeda sangat kecewa sekali",
        "pengiriman lambat seller cuek tidak ada kabar sama sekali buruk",
        "rusak sudah dipakai sekali kualitas rendah tidak tahan lama",
        "tidak sesuai deskripsi warna beda ukuran beda menipu pembeli",
        "penipu barang tidak dikirim uang sudah masuk sangat merugikan",
    ]
    netral = [
        "barang sudah sampai kondisi biasa saja sesuai harga",
        "kualitas standar tidak terlalu bagus tidak terlalu jelek lumayan",
        "produk oke untuk harga segini tidak ada yang istimewa",
        "pengiriman normal sesuai estimasi barang sesuai deskripsi",
        "cukup baik untuk pemakaian sehari hari tidak ada masalah",
        "standar saja tidak lebih tidak kurang sesuai ekspektasi harga",
        "produk biasa saja packing oke pengiriman normal tidak ada masalah",
        "kualitas sesuai harga tidak mengecewakan tapi tidak istimewa juga",
        "oke lah untuk dipakai sehari hari tidak ada keluhan berarti",
        "lumayan untuk harga segini tidak buruk tidak terlalu bagus juga",
    ]

    np.random.seed(42)
    records = []

    n_pos  = n * 50 // 100
    n_neg  = n * 30 // 100
    n_neut = n - n_pos - n_neg

    for _ in range(n_pos):
        base = np.random.choice(positif)
        records.append({"review": base, "sentiment": "positif"})
    for _ in range(n_neg):
        base = np.random.choice(negatif)
        records.append({"review": base, "sentiment": "negatif"})
    for _ in range(n_neut):
        base = np.random.choice(netral)
        records.append({"review": base, "sentiment": "netral"})

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
#  LOAD DATASET
# ─────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """
    Prioritas pembacaan dataset:
      1. CSV asli dari Kaggle  (dataset/shopee_reviews.csv)
      2. Dataset sintetis fallback
    """
    kaggle_path = os.path.join(BASE_DIR, "dataset", "datasetshopee", "data.csv")
    if os.path.exists(kaggle_path):
        print(f"📂  Memuat dataset Kaggle: {kaggle_path}")
        df = pd.read_csv(kaggle_path, encoding="utf-8")

        # Normalisasi nama kolom
        col_map = {}
        for col in df.columns:
            low = col.lower()
            if any(k in low for k in ["review", "ulasan", "comment", "text", "content"]):
                col_map[col] = "review"
            elif any(k in low for k in ["label", "sentimen", "sentiment", "class"]):
                col_map[col] = "sentiment"
            elif any(k in low for k in ["rating", "star", "bintang"]):
                col_map[col] = "rating"
        df = df.rename(columns=col_map)

        # Buat kolom sentimen dari rating jika belum ada
        if "sentiment" not in df.columns and "rating" in df.columns:
            df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(3)
            df["sentiment"] = df["rating"].apply(
                lambda r: "positif" if r >= 4 else ("negatif" if r <= 2 else "netral")
            )

        df = df.dropna(subset=["review", "sentiment"])
        df["review"]    = df["review"].astype(str)
        df["sentiment"] = df["sentiment"].str.lower().str.strip()
        print(f"✅  Dataset dimuat: {len(df)} baris")
        return df

    print("⚠️   Dataset Kaggle tidak ditemukan → menggunakan dataset sintetis")
    return generate_synthetic_dataset(2000)


# ─────────────────────────────────────────────
#  TRAINING MODEL
# ─────────────────────────────────────────────
def train_model():
    """Latih Multinomial Naive Bayes, simpan artefak, return metrics."""
    print("🏋️  Memulai training model ...")

    df = load_dataset()

    # Preprocessing
    print("🔧  Preprocessing teks ...")
    df["clean_text"] = df["review"].apply(preprocess_text)
    df = df[df["clean_text"].str.len() > 5]

    X = df["clean_text"].values
    y = df["sentiment"].values

    # Encode label
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_vec = vectorizer.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # Training Naive Bayes
    model = MultinomialNB(alpha=0.5, class_prior=None)
    model.fit(X_train, y_train, sample_weight=None)

# Hitung class weight manual untuk balancing
    from sklearn.utils.class_weight import compute_sample_weight
    weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model = MultinomialNB(alpha=0.5)
    model.fit(X_train, y_train, sample_weight=weights)

    # Evaluasi
    
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    labels = le.classes_.tolist()

    # Cross-validation
    cv_scores = cross_val_score(
        MultinomialNB(alpha=0.5), X_vec, y_enc, cv=5, scoring="accuracy"
    )

    metrics = {
        "accuracy":   round(float(acc) * 100, 2),
        "precision":  round(float(prec) * 100, 2),
        "recall":     round(float(rec) * 100, 2),
        "f1_score":   round(float(f1) * 100, 2),
        "cv_mean":    round(float(cv_scores.mean()) * 100, 2),
        "cv_std":     round(float(cv_scores.std()) * 100, 2),
        "confusion_matrix": cm,
        "labels":     labels,
        "total_data": len(df),
        "train_size": int(X_train.shape[0]),
        "test_size":  int(X_test.shape[0]),
        "vocab_size": len(vectorizer.vocabulary_),
        "distribution": {
            lbl: int((y == lbl).sum()) for lbl in labels
        },
    }

    # Simpan artefak model
    joblib.dump(model,       os.path.join(MODEL_DIR, "naive_bayes.pkl"))
    joblib.dump(vectorizer,  os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(le,          os.path.join(MODEL_DIR, "label_encoder.pkl"))
    joblib.dump(df,          os.path.join(MODEL_DIR, "dataframe.pkl"))

    with open(METRIC_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"✅  Training selesai! Akurasi: {metrics['accuracy']}%")
    return model, vectorizer, le, metrics


def load_or_train_model():
    """Load model tersimpan atau train ulang jika belum ada."""
    paths = [
        os.path.join(MODEL_DIR, "naive_bayes.pkl"),
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"),
        os.path.join(MODEL_DIR, "label_encoder.pkl"),
    ]
    if all(os.path.exists(p) for p in paths):
        print("📦  Memuat model tersimpan ...")
        model     = joblib.load(paths[0])
        vectorizer = joblib.load(paths[1])
        le        = joblib.load(paths[2])
        return model, vectorizer, le

    model, vectorizer, le, _ = train_model()
    return model, vectorizer, le


def get_model_metrics() -> dict:
    if os.path.exists(METRIC_FILE):
        with open(METRIC_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    _, _, _, metrics = train_model()
    return metrics


# ─────────────────────────────────────────────
#  PREDIKSI
# ─────────────────────────────────────────────
def predict_sentiment(model, vectorizer, le, text: str) -> dict:
    """Prediksi sentimen satu teks, kembalikan detail lengkap."""
    cleaned  = preprocess_text(text)
    X_vec    = vectorizer.transform([cleaned])
    pred_enc = model.predict(X_vec)[0]
    proba    = model.predict_proba(X_vec)[0]

    label    = le.inverse_transform([pred_enc])[0]
    classes  = le.classes_.tolist()
    proba_map = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}

    emoji_map = {"positif": "😊", "negatif": "😠", "netral": "😐"}
    color_map = {"positif": "success", "negatif": "danger", "netral": "warning"}

    return {
        "original_text":  text,
        "cleaned_text":   cleaned,
        "sentiment":      label,
        "emoji":          emoji_map.get(label, "🤔"),
        "color":          color_map.get(label, "info"),
        "confidence":     round(float(proba.max()) * 100, 2),
        "probabilities":  proba_map,
    }


# ─────────────────────────────────────────────
#  WORD CLOUD DATA
# ─────────────────────────────────────────────
def get_wordcloud_data(sentiment: str = "positif") -> list:
    df_path = os.path.join(MODEL_DIR, "dataframe.pkl")
    if not os.path.exists(df_path):
        return []

    df = joblib.load(df_path)
    subset = df[df["sentiment"] == sentiment]["clean_text"].dropna()
    all_words = " ".join(subset).split()
    counter = Counter(all_words)
    # Kembalikan 50 kata teratas dalam format [{text, value}]
    return [
        {"text": w, "value": c}
        for w, c in counter.most_common(50)
        if len(w) >= 3
    ]


# ─────────────────────────────────────────────
#  JALANKAN TRAINING DARI CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀  Menjalankan training mandiri ...")
    m, v, le, met = train_model()
    print("\n📊  HASIL EVALUASI:")
    print(f"    Akurasi   : {met['accuracy']}%")
    print(f"    Precision : {met['precision']}%")
    print(f"    Recall    : {met['recall']}%")
    print(f"    F1-Score  : {met['f1_score']}%")
    print(f"    CV Mean   : {met['cv_mean']}% ± {met['cv_std']}%")
