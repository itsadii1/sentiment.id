# 🛒 Analisis Sentimen Ulasan E-Commerce Shopee
## Menggunakan Metode Naive Bayes Classifier

---

## 📁 Struktur Proyek

```
shopee-sentiment/
│
├── app.py                    # Server Flask (titik masuk aplikasi)
├── ml_model.py               # Modul Machine Learning (Naive Bayes)
├── requirements.txt          # Daftar dependensi Python
│
├── templates/
│   └── index.html            # Tampilan utama (UI)
│
├── static/
│   ├── css/style.css         # Stylesheet
│   └── js/main.js            # JavaScript frontend
│
├── dataset/
│   └── shopee_reviews.csv    # ← Letakkan dataset Kaggle di sini
│
├── models/                   # Otomatis dibuat saat training
│   ├── naive_bayes.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   ├── dataframe.pkl
│   └── metrics.json
│
└── notebooks/
    └── analisis_sentimen.ipynb  # Jupyter Notebook EDA & Training
```

---

## ⚙️ Langkah-Langkah Setup (VS Code)

### STEP 1 — Install Python
1. Download Python 3.11+ dari https://python.org
2. Centang **"Add Python to PATH"** saat instalasi
3. Verifikasi: buka terminal → ketik `python --version`

---

### STEP 2 — Buka Proyek di VS Code
1. Buka VS Code
2. **File → Open Folder** → pilih folder `shopee-sentiment`
3. Install ekstensi:
   - **Python** (Microsoft)
   - **Pylance**
   - **Jupyter** (untuk membuka notebook)

---

### STEP 3 — Buat Virtual Environment

Buka terminal di VS Code (**Ctrl + `**) dan jalankan:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Jika berhasil, terminal akan menampilkan `(venv)` di depan prompt.

---

### STEP 4 — Install Dependensi

```bash
pip install -r requirements.txt
```

Tunggu hingga semua paket terinstall (±2–5 menit tergantung koneksi).

---

### STEP 5 — Download Dataset dari Kaggle

**Opsi A — Langsung dari Kaggle:**
1. Buka https://www.kaggle.com
2. Cari salah satu dari:
   - `shopee product reviews`
   - `shopee indonesia ulasan`
   - `e-commerce review bahasa indonesia`
3. Klik **Download** → ekstrak ZIP
4. Rename/pindahkan file CSV ke `dataset/shopee_reviews.csv`

**Format CSV yang diharapkan:**
```
review,rating
"Barang bagus dan cepat sampai",5
"Kualitas jelek tidak sesuai",1
"Biasa saja sesuai harga",3
```

Atau dengan kolom `sentiment` langsung:
```
review,sentiment
"Barang bagus",positif
"Kualitas jelek",negatif
```

**Opsi B — Tanpa Dataset Kaggle:**
Jika tidak punya dataset, sistem otomatis menggunakan dataset sintetis 2.000 data untuk demo.

---

### STEP 6 — Jalankan Aplikasi

```bash
python app.py
```

Output yang muncul:
```
🔄  Memuat model Naive Bayes ...
🏋️  Memulai training model ...
✅  Training selesai! Akurasi: 85.50%
✅  Model siap digunakan!

============================================================
  🛒  Shopee Sentiment Analyzer - Naive Bayes
  🌐  Buka browser: http://127.0.0.1:5000
============================================================
```

Buka browser → http://127.0.0.1:5000

---

### STEP 7 — Menggunakan Aplikasi

1. **Ketik ulasan** di kotak teks, atau klik tombol contoh (😊/😠/😐)
2. Klik **Analisis Sentimen** (atau Ctrl+Enter)
3. Hasil tampil:
   - Label sentimen (POSITIF / NEGATIF / NETRAL)
   - Confidence score (%)
   - Distribusi probabilitas 3 kelas
   - Teks setelah preprocessing

---

### STEP 8 — Eksplorasi dengan Jupyter Notebook (Opsional)

```bash
pip install jupyter matplotlib seaborn
jupyter notebook notebooks/analisis_sentimen.ipynb
```

Notebook berisi:
- EDA (Exploratory Data Analysis)
- Visualisasi distribusi data
- Training step-by-step dengan penjelasan
- Evaluasi lengkap (confusion matrix, classification report)
- Cross-validation 5-fold
- Prediksi manual

---

## 🧠 Cara Kerja Algoritma

### 1. Preprocessing
```
Input: "Barang BAGUS banget!! Packing rapi 👍"
→ Lowercase: "barang bagus banget!! packing rapi 👍"
→ Hapus simbol: "barang bagus banget packing rapi"
→ Tokenisasi: ["barang", "bagus", "banget", "packing", "rapi"]
→ Hapus stopwords: ["barang", "bagus", "packing", "rapi"]
→ Stemming: ["barang", "bagus", "packing", "rapi"]
Output: "barang bagus packing rapi"
```

### 2. TF-IDF Vectorization
- **TF (Term Frequency):** Frekuensi kata dalam dokumen
- **IDF (Inverse Document Frequency):** Bobot kepentingan kata di seluruh corpus
- **Ngram (1,2):** Unigram + Bigram (contoh: "tidak bagus" sebagai satu fitur)

### 3. Naive Bayes Classifier
```
P(Positif | ulasan) = P(ulasan | Positif) × P(Positif) / P(ulasan)
```
Kelas dengan probabilitas posterior tertinggi = hasil prediksi.

### 4. Evaluasi
| Metrik | Keterangan |
|--------|-----------|
| **Akurasi** | % prediksi benar dari total data |
| **Precision** | % prediksi positif yang benar-benar positif |
| **Recall** | % data positif yang berhasil diprediksi |
| **F1-Score** | Rata-rata harmonik Precision & Recall |

---

## 🐛 Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `ModuleNotFoundError` | Pastikan virtual env aktif: `venv\Scripts\activate` |
| Port 5000 sudah dipakai | Edit `app.run(port=5001)` di `app.py` |
| Dataset tidak terbaca | Cek nama file: harus `shopee_reviews.csv` di folder `dataset/` |
| Model error | Hapus folder `models/`, restart `app.py` |
| Akurasi rendah | Dataset sintetis lebih kecil. Pakai dataset Kaggle asli |

---

## 📚 Referensi

- Scikit-learn Naive Bayes: https://scikit-learn.org/stable/modules/naive_bayes.html
- Flask Documentation: https://flask.palletsprojects.com
- Dataset Kaggle Shopee: https://www.kaggle.com/search?q=shopee+reviews
- TF-IDF: https://scikit-learn.org/stable/modules/feature_extraction.html

---

*Dibuat untuk keperluan Tugas Akhir / Skripsi Machine Learning*
