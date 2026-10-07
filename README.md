# 🫀 Clinical Risk Prediction — Heart Disease

End-to-end machine learning project untuk memprediksi risiko penyakit jantung
dari data klinis dasar, lengkap dengan explainable AI (SHAP), REST API, dan
dashboard interaktif.

> ⚠️ **Disclaimer**: Project ini dibuat untuk tujuan edukasi/portofolio.
> Hasil prediksi BUKAN pengganti diagnosis medis profesional.

---

## 📌 Problem Statement

Penyakit jantung adalah penyebab kematian nomor satu secara global. Deteksi
dini terhadap pasien berisiko tinggi dapat membantu tenaga medis
memprioritaskan pemeriksaan lanjutan. Project ini membangun sistem prediksi
risiko penyakit jantung dari 13 parameter klinis dasar (usia, tekanan darah,
kolesterol, hasil EKG, dll), menggunakan dataset **UCI Heart Disease
(Cleveland)**.

---

## 🎯 Hasil Utama

| Model | Accuracy | Recall (Berisiko) | Precision (Berisiko) |
|---|---|---|---|
| **Logistic Regression** ✅ | **0.90** | 0.88 | **0.88** |
| Random Forest | 0.88 | 0.88 | 0.84 |
| XGBoost | 0.83 | 0.88 | 0.75 |

**Model terpilih: Logistic Regression.**

Insight menarik: meskipun model yang lebih kompleks (Random Forest, XGBoost)
sering dianggap "lebih canggih", hasil eksperimen menunjukkan model linear
sederhana justru lebih optimal untuk dataset berukuran kecil (297 baris
setelah cleaning) — menegaskan pentingnya memilih model sesuai karakteristik
data, bukan sekadar mengejar kompleksitas.

Evaluasi difokuskan pada **recall** kelas berisiko (bukan hanya akurasi),
karena di konteks medis, *false negative* (pasien berisiko diprediksi sehat)
jauh lebih berbahaya daripada *false positive*.

---

## 🧠 Explainability (SHAP)

Setiap prediksi disertai breakdown SHAP value per-pasien, menjelaskan fitur
apa yang paling mendorong hasil prediksi ke arah "berisiko" atau "sehat".
Fitur dengan pengaruh terbesar secara umum: `thal`, `ca`, `oldpeak`, `cp`,
`exang`, `thalach`.

---

## 🏗️ Arsitektur

```
┌─────────────────┐      HTTP (JSON)      ┌──────────────────┐
│   Streamlit      │ ───────────────────▶ │     FastAPI        │
│   Dashboard       │ ◀─────────────────── │  (Model + SHAP)    │
└─────────────────┘                        └──────────────────┘
```

API dan Dashboard dijalankan sebagai service terpisah (separation of
concerns) — masing-masing bisa di-scale atau diperbarui secara independen.
Keduanya dibungkus dalam container Docker terpisah, dikoordinasikan dengan
Docker Compose.

---

## 📁 Struktur Project

```
clinical-risk-prediction/
├── data/
│   ├── raw/               # dataset mentah
│   └── processed/         # hasil preprocessing
├── notebooks/
│   └── 01_eda.py          # eksplorasi data
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py         # SHAP
├── models/                 # model, scaler, X_train tersimpan (.pkl)
├── main.py                 # FastAPI service
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
└── README.md
```

---

## 🚀 Cara Menjalankan

### Opsi 1 — Manual

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Terminal 1 — API
uvicorn main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run app.py
```

Buka `http://localhost:8501` untuk dashboard, atau `http://localhost:8000/docs`
untuk dokumentasi API interaktif.

### Opsi 2 — Docker Compose

```bash
docker compose up --build
```

Dashboard otomatis tersedia di `http://localhost:8501`, API di
`http://localhost:8000`.

---

## 🛠️ Tech Stack

- **Data & Modeling**: pandas, scikit-learn, XGBoost
- **Explainability**: SHAP
- **API**: FastAPI
- **Dashboard**: Streamlit, Plotly
- **Deployment**: Docker, Docker Compose

---

## 📈 Pengembangan Lanjutan

- [ ] Threshold tuning untuk optimasi recall lebih lanjut
- [ ] Cross-validation & hyperparameter tuning
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model monitoring untuk data drift
