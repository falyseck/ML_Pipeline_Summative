# 🔬 Thyroid Cancer Image Classifier

Binary classification of thyroid ultrasound images — **Benign vs Malignant** — using EfficientNetB0 transfer learning, deployed as a FastAPI + Streamlit pipeline.

---

## 🎥 Demo

- **Video Demo:** [https://youtu.be/vAoCSVoRtFY]
- **Live UI:** [Thyroid Classifier · Streamlit](https://thyroid-classifier.streamlit.app/)
- **Live API:** [Thyroid Cancer Classifier API](https://thyroid-classifier-api1.onrender.com/)
- **API Docs (Swagger UI):** [https://your-api.onrender.com/docs](https://thyroid-classifier-api1.onrender.com/docs) 

---

## 📋 Project Description

This project implements an end-to-end Machine Learning pipeline for thyroid cancer classification from ultrasound images. It covers:

- **Data Acquisition** — Thyroid ultrasound dataset from Kaggle (~2879 images, 2 classes)
- **Preprocessing** — EfficientNet-native `preprocess_input`, augmentation, 70/15/15 train/val/test split
- **Model Development** — EfficientNetB0 transfer learning with a documented 3-version journey:
  - v1: Baseline with incorrect preprocessing → ~50% AUC (random guessing)
  - v2: Fixed preprocessing → 80% AUC but overfitting (18% train/val gap)
  - v3: Regularised with AdamW, stronger Dropout, reduced unfrozen layers → stable generalisation
- **Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix, optimal threshold tuning
- **API** — FastAPI with predict, upload, retrain, and health endpoints
- **UI** — Streamlit dashboard with dark medical aesthetic
- **Deployment** — FastAPI on Render, Streamlit UI on Streamlit Community Cloud
- **Load Testing** — Locust flood simulation across 5–40 concurrent users

### Model Performance (v3 — Final)

| Metric | Score |
|---|---|
| AUC-ROC | ~0.80 |
| Accuracy | ~68% (val) |
| Recall | ~72% |
| F1 Score | ~68% |

> Val accuracy is intentionally modest — v3 prioritises generalisation over memorisation. The train/val gap was reduced from 13% (v2) to ~7% (v3) through regularisation.

---

## 🗂 Directory Structure

```
thyroid-classifier/
├── .streamlit/
│   └── config.toml
├── api/
│   ├── main.py
│   └── requirements.txt
├── data/
│   ├── raw/                    # original images (benign/, malignant/, …)
│   ├── train/                  # train split (benign/, malignant/, …)
│   ├── test/                   # test split (if present)
│   ├── retrain/                # created at runtime for retrain uploads
│   ├── uploads/                # created at runtime for zip uploads
│   └── *.png                   # visualization exports (e.g. viz_*.png)
├── loadtest/
│   ├── locustfile.py
│   ├── run_locust_matrix.ps1
│   ├── README.md
│   └── results/                # Locust CSV output 
├── models/
│   ├── thyroid_efficientnet.h5
│   ├── thyroid_efficientnet.keras
│   ├── thyroid_efficientnet.tf/
│   ├── model_meta.pkl
│   ├── training_log.csv
│   ├── training_log_v3.csv
│   └── …                       # other checkpoints / logs as saved
├── notebooks/
│   └── thyroid_cancer_classification_v2.ipynb
├── src/
│   ├── model.py
│   ├── prediction.py
│   └── preprocessing.py
├── app.py                      # Streamlit dashboard
├── Dockerfile
├── render.yaml
└── requirements.txt            # Streamlit UI deps (root)
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.11+

### 1. Clone the repository
```bash
git clone https://github.com/falyseck/thyroid-classifier.git
cd thyroid-classifier
```

### 2. Install dependencies
```bash
# API
pip install -r api/requirements.txt

# UI
pip install -r requirements.txt
```

### 3. Prepare the dataset
Download from Kaggle and place images here:
```
data/raw/benign/      ← benign .jpg images
data/raw/malignant/   ← malignant .jpg images
```
source: https://www.kaggle.com/datasets/diveshzz/thyroid-cancer-classification-ultrasound-dataset
### 4. Run the notebook
Open and run `notebook/thyroid_cancer_classification.ipynb` end-to-end. This will:
- Split the dataset (70/15/15)
- Train EfficientNetB0 (v2 + v3 with overfitting fix)
- Save `models/thyroid_efficientnet.h5` and `models/model_meta.pkl`

### 5. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
Test at: http://localhost:8000/docs

### 6. Start the Streamlit UI
```bash
streamlit run app.py
```
Opens at: http://localhost:8501 — set API URL to `http://localhost:8000` and click **Connect**.

---

## ☁️ Deployment

### API — Render
1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo
3. Runtime: **Dockerfile**
4. Deploy and copy the service URL

### UI — Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) → New app
2. Connect your GitHub repo
3. Main file path: `app.py`
4. Deploy — done

> ⚠️ Render free tier services spin down after 15 minutes of inactivity. The first request after sleep may take ~30 seconds.

---

## 🔁 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health ping |
| GET | `/health` | Detailed system status |
| GET | `/model-info` | Model version, metrics, threshold |
| GET | `/metrics` | Evaluation metrics |
| POST | `/predict` | Predict a single ultrasound image |
| POST | `/upload-data` | Upload ZIP of new labelled images |
| POST | `/retrain` | Trigger background retraining |
| GET | `/retrain-status` | Check retraining progress |
| GET | `/stats` | API usage statistics |

---

## 🌊 Load Testing with Locust

### Install & Run
```bash
pip install locust
locust -f locustfile.py --host=https://your-api.onrender.com
```
Open http://localhost:8089, configure users and spawn rate, then start.

### Flood Simulation Results

Tests were run against the deployed Render API simulating concurrent prediction requests (`POST /predict`) mixed with health checks and metrics polling.

| Users | Total Requests | Failures | Median Latency | p95 Latency | p99 Latency | RPS |
|---|---|---|---|---|---|---|
| 5 | 25 | 0 | 0ms | 49,000ms | 54,000ms | 0.32 |
| 10 | 68 | 0 | 0ms | 11,000ms | 14,000ms | 0.78 |
| 20 | 65 | 0 | 0ms | 16,000ms | 19,000ms | 0.75 |
| 40 | 59 | 0 | 0ms | 44,000ms | 46,000ms | 0.66 |

### Interpretation

- **Zero failures across all load levels** — the API handles all requests without crashing at any concurrency level tested
- **High p95/p99 latency** — values of 11,000–54,000ms reflect the Render free tier cold start behaviour (service sleeps after 15 minutes of inactivity). On a warm instance or paid tier, latency drops significantly
- **Low RPS (0.32–0.78)** — expected for a TensorFlow inference endpoint on a single free-tier container; each prediction runs a full EfficientNetB0 forward pass (~224×224 image through ~7M parameters)
- **10 users yielded the best p95** (11,000ms) — the service was likely already warm at that point in the test sequence
- **Scaling recommendation** — horizontal scaling to 2–3 containers on Render's paid tier would distribute prediction load and reduce per-request latency proportionally; vertical scaling (larger instance) would reduce cold start time

---

## 👤 Author

**El Hadji Faly Seck**  
African Leadership University — BSE  
Machine Learning Pipeline — Summative Assignment
