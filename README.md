# ☕ Amazon Fine Food Reviews — Sentiment Intelligence & Multi-Model Arena

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.1-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn%20%26%20XGBoost-orange.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/UI-Apple%20Liquid%20Glass-crimson.svg)](#)
[![Tests](https://img.shields.io/badge/Tests-5%2F5%20Passed-brightgreen.svg)](#)

> End-to-end Natural Language Processing (NLP) sentiment intelligence application built on the official **Stanford SNAP Amazon Fine Food Reviews dataset (568,454 reviews)**. Evaluates and compares **6 machine learning algorithms**, implements a calibrated **Combined Ensemble Average** soft-voting system, provides bulk batch analysis with quality defect alert thresholds, and delivers a modern **Apple Liquid Glassmorphism web application** split across 3 interactive pages.

---

## 📊 Live Measured Model Tournament Results

Measured on the held-out test set ($N = 8,000$ reviews, stratified $80/20$ split, random seed 42) from the authentic cleaned dataset:

| # | Algorithm | Featurization | Accuracy | F1-Score | ROC-AUC | Recall | Precision | Latency | Primary Strength |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **Logistic Regression** | TF-IDF (10k n-grams) | **92.79%** | **95.81%** | **95.63%** | 97.70% | 93.98% | 0.004 ms | **Individual Champion**. Optimal margin with L2 penalty ($C=5.0$). |
| **2** | **Linear SVM** | TF-IDF (10k n-grams) | **92.70%** | **95.77%** | **95.09%** | 98.00% | 93.64% | 0.003 ms | **Maximum Margin Hyperplane**. Smooth calibrated probabilities via Modified Huber loss. |
| **3** | **Multinomial Naive Bayes** | CountVectorizer (BoW) | **90.92%** | **94.59%** | **92.64%** | 94.01% | 95.17% | 0.002 ms | **Ultra-Fast NLP**. Conditional probability with Laplace smoothing ($\alpha=0.10$). |
| **4** | **XGBoost Classifier** | TF-IDF Sparse | **90.30%** | **94.50%** | **92.85%** | 98.87% | 90.50% | 0.025 ms | **Gradient Boosted Ensembling**. 100 sequential shallow decision trees. |
| **5** | **Random Forest** | TF-IDF | **84.51%** | **91.59%** | **90.44%** | **100.0%** | 84.48% | 0.119 ms | **De-correlated Bagging**. Variance reduction across 100 random feature trees. |
| **6** | **K-Nearest Neighbors (KNN)** | TF-IDF Cosine | **86.56%** | **92.51%** | **77.88%** | 98.46% | 87.24% | 0.350 ms | **Spatial Memory Search**. Nearest $K=7$ neighbor search in cosine space. |
| 👑 | **Combined Ensemble Average** | **All 6 Combined** | **92.36%** | **95.62%** | **95.44%** | **98.99%** | **92.48%** | **0.503 ms** | **Consensus Soft-Voting**. $P_{\text{ens}} = \frac{1}{6}\sum_{i=1}^6 P_i$. Eliminates single-model variance with **94.37% test agreement**. |

*Note: The Combined Ensemble achieves **98.99% recall on positive feedback**, outperforming individual models such as Logistic Regression (97.70%) and SVM (98.00%) in catching true positive sentiment while maintaining robust generalization.*

---

## 🌐 The 3-Page Web Application

1. **Page 1: Live Sentiment Studio (`/`)**
   - Real-time review text analysis with instant sample presets.
   - Dynamic selection of individual models or the **Combined Ensemble Average**.
   - Dual-gauge confidence percentage indicator and probability spectrum.
   - 6-model consensus agreement breakdown with agreement percentage.
   - Explainability tokens: Top positive and negative feature weights with directional attribution.
   - Clear button and responsive input validation.

2. **Page 2: Batch and Product Health Analytics (`/batch`)**
   - Bulk review ingestion via pasted reviews (line-by-line) or **direct CSV file upload** (`Reviews.csv` schema auto-detected).
   - Configurable **Quality Defect Alert Threshold** (default: 25% unhappy).
   - Sample size detection: Highlights preliminary findings when $N < 10$.
   - Interactive sentiment distribution progress chart.
   - Filterable & searchable review table with live keyword filtering and sentiment toggle.
   - **Downloadable CSV Export** of sentiment predictions and confidence scores.
   - 1-click **Inspect in Studio** deep link integration.
   - Distinguishes **Demo QA Groups** from authentic SNAP catalog products.

3. **Page 3: Model Tournament & Visual Flow (`/benchmarks`)**
   - Side-by-side cards for all 6 models + the crown Ensemble card.
   - Real measured metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Latency, and Training Time.
   - **Live Confusion Matrix breakdown** (True Negatives, False Positives, False Negatives, True Positives).
   - Visual ROC-AUC Leaderboard comparison bar chart.
   - Dataset distribution and data hygiene audit card.
   - Toggle between **Freshly Measured Test Set** and **Historical Notebook Research (34 experiments)**.
   - 4-stage visual data pipeline: Raw Dataset $\rightarrow$ Preprocessing $\rightarrow$ Feature Extraction $\rightarrow$ Ensemble Verdict.

---

## 🚀 Windows PowerShell Quick Start Guide

No Docker required. Run directly in Windows PowerShell:

### 1. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Preprocess Dataset (Stanford SNAP)
```powershell
# Ingests data/Reviews.csv, runs deduplication, cleans HTML/negations, and saves data/processed_reviews.csv
python data_pipeline.py
```

### 4. Train Models & Generate Benchmarks
```powershell
# Trains all 6 algorithms, measures test metrics, and writes models/evaluation_results.json
python model_trainer.py --sample-size 40000
```
*(To train across the entire 364k binary dataset, run with `--full`)*

### 5. Run Automated Tests
```powershell
pytest test_app.py -v
```

### 6. Start the Web Server
```powershell
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📁 Repository Structure

```
Amazon-Fine-Food-Reviews/
├── app.py                             # Flask web application & REST API endpoints
├── data_pipeline.py                   # Data ingestion, deduplication & preprocessing pipeline
├── model_trainer.py                   # 6-Algorithm training, ensemble scoring & evaluation engine
├── benchmark_data.py                  # Historical reference metrics from research notebooks
├── test_app.py                        # Automated pytest suite (5/5 tests passing)
│
├── PROJECT_DOCUMENTATION.md           # Exhaustive technical documentation & formulations
├── PROJECT_FLOW_AND_ARCHITECTURE.md   # Architectural pipeline flow & Mermaid diagrams
├── README.md                          # Project overview and quick start guide
├── requirements.txt                   # Production dependencies
│
├── data/
│   ├── Reviews.csv                    # Official raw SNAP dataset (568,454 records, ~300 MB)
│   ├── processed_reviews.csv          # Cleaned & deduplicated dataset (393,931 records)
│   ├── dataset_stats.json             # Sanitization audit & sentiment split statistics
│   └── amazon_food_reviews.csv        # Compact seed dataset for fast testing
│
├── models/
│   ├── evaluation_results.json        # Freshly measured test set metrics & confusion matrices
│   ├── logistic_regression.joblib     # Serialized Logistic Regression model
│   ├── naive_bayes.joblib             # Serialized Multinomial Naive Bayes model
│   ├── svm.joblib                     # Serialized Linear SVM model
│   ├── xgboost.joblib                 # Serialized XGBoost Classifier
│   ├── random_forest.joblib           # Serialized Random Forest Classifier
│   ├── knn.joblib                     # Serialized K-Nearest Neighbors Classifier
│   ├── tfidf_vectorizer.joblib        # Fitted 10,000-feature TF-IDF vectorizer
│   ├── bow_vectorizer.joblib          # Fitted CountVectorizer
│   └── trained_models_6.joblib        # Bundled model archive
│
├── templates/
│   ├── index.html                     # Page 1: Live Sentiment Studio
│   ├── batch.html                     # Page 2: Batch & Product Health Analytics
│   └── benchmarks.html                # Page 3: Model Tournament & Visual Flow
│
├── static/
│   ├── style.css                      # Apple Liquid Glass design system styles
│   ├── app.js                         # Live Sentiment Studio controller
│   └── batch.js                       # Batch analytics, CSV upload/export controller
│
└── 1-8 .ipynb                         # Original 8 research Jupyter notebooks
```

---

## 📡 REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/predict` | `POST` | Live sentiment prediction for single review. Supports individual model or `ensemble_avg`. Returns prediction, confidence, 6-model consensus, agreement %, and word attributions. |
| `/api/predict-batch` | `POST` | Bulk review evaluation. Returns total analyzed, positive/negative counts & percentages, defect alert boolean, satisfaction verdict, and individual review scores. |
| `/api/evaluation` | `GET` | Returns freshly measured test set evaluation results from `models/evaluation_results.json` including confusion matrices, latency, and sample splits. |
| `/api/health` | `GET` | System health check reporting loaded models, vectorizer feature counts, and operational readiness. |
| `/api/stats` | `GET` | Returns dataset ingestion and cleaning statistics from `data/dataset_stats.json`. |
| `/api/top-words` | `GET` | Returns top positive and negative vocabulary tokens based on trained model coefficients. |
| `/api/benchmark` | `GET` | Historical leaderboard data across 34 experiments for reference comparison. |
| `/api/reviews` | `GET` | Paginated, filterable reviews from the authentic dataset catalog. |
| `/api/samples` | `GET` | Curated sample reviews for 1-click test runs. |
