# Amazon Fine Food Reviews — Complete Technical Documentation & Engineering Manual

> **End-to-End Sentiment Intelligence Platform, 6-Algorithm Tournament & Real-Time Production Ingestion**  
> *Dataset: 568,454 Stanford SNAP Reviews • 6 Core ML Classifiers • Soft-Voting Ensemble • Apple Liquid Glass UI • REST API*

---

## Table of Contents
1. [Executive Summary & System Objectives](#1-executive-summary--system-objectives)
2. [Dataset Provenance, Sanitation & Deduplication](#2-dataset-provenance-sanitation--deduplication)
3. [The 4-Stage NLP Preprocessing Pipeline](#3-the-4-stage-nlp-preprocessing-pipeline)
4. [The 6 Core Algorithms & Mathematical Formulations](#4-the-6-core-algorithms--mathematical-formulations)
5. [Combined Ensemble Average & Empirical Evaluation](#5-combined-ensemble-average--empirical-evaluation)
6. [Why Test Reviews Individually vs. In Batch?](#6-why-test-reviews-individually-vs-in-batch)
7. [Three-Page Web Architecture & User Interface](#7-three-page-web-architecture--user-interface)
8. [REST API Endpoints & Contract Specifications](#8-rest-api-endpoints--contract-specifications)
9. [Repository Inventory & Artifact Architecture](#9-repository-inventory--artifact-architecture)
10. [Local Execution, PowerShell Commands & Automated Tests](#10-local-execution-powershell-commands--automated-tests)
11. [Actual Measured Evaluation Metrics & Analysis](#11-actual-measured-evaluation-metrics--analysis)
12. [Known Limitations & Engineering Considerations](#12-known-limitations--engineering-considerations)

---

## 1. Executive Summary & System Objectives

The **Amazon Fine Food Reviews Sentiment Intelligence System** is an end-to-end natural language processing (NLP) and machine learning platform engineered to evaluate customer food reviews, classify consumer sentiment (**Positive** vs. **Negative**), detect product quality defects in bulk, and compare model behaviors across 6 distinct machine learning families.

### Core Objectives:
1. **Algorithmic Rigor:** Train, calibrate, and benchmark 6 distinct classifiers:
   - **Logistic Regression:** Linear separating hyperplanes with L2 regularization ($C=5.0$) on sublinear TF-IDF features.
   - **Multinomial Naive Bayes:** Probabilistic term-frequency modeling with Laplace smoothing ($\alpha=0.10$) on Bag-of-Words matrices.
   - **Linear Support Vector Machine (SVM):** Stochastic gradient descent with modified Huber loss to yield smooth, calibrated probability scores.
   - **XGBoost Classifier:** Sequential gradient-boosted shallow decision trees fitting negative gradients.
   - **Random Forest:** De-correlated bagging ensemble with feature sub-sampling to control variance.
   - **K-Nearest Neighbors (KNN):** Spatial memory lookup utilizing Cosine distance ($K=7$) in TF-IDF space.
2. **Calibrated Ensemble Soft-Voting:** Combine the predicted probabilities of all 6 algorithms to compute an unweighted average consensus vote, neutralizing individual model blindspots.
3. **Streaming Production Ingestion:** Single-review evaluation in $< 2\text{ ms}$ to intercept critical defect reviews, contamination reports, and star-rating mismatches.
4. **Catalog Batch Diagnostics & Quality Alerts:** Bulk review analysis with a user-configurable **Quality Defect Alert Threshold**, automated CSV upload/export, and small sample size warnings.
5. **State-of-the-Art Apple Liquid Glass UI:** A responsive, accessible interface with translucent glass panels, backdrop blurs, fluid micro-interactions, and light/dark theme switching.

---

## 2. Dataset Provenance, Sanitation & Deduplication

### Source Information:
* **Origin:** Stanford Network Analysis Platform (SNAP) — J. McAuley & J. Leskovec, WWW 2013.
* **Official URL:** [https://snap.stanford.edu/data/web-FineFoods.html](https://snap.stanford.edu/data/web-FineFoods.html)
* **Raw Record Count:** **568,454 reviews** spanning October 1999 to October 2012 across 74,258 products and 256,059 users.
* **File Location:** `data/Reviews.csv` (~300.9 MB).

### Preprocessing & Data Cleaning Pipeline (`data_pipeline.py`):
1. **Column Validation:** Validates required schema: `['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time', 'Summary', 'Text']`.
2. **Helpfulness Integrity Check:** Removes records where `HelpfulnessNumerator > HelpfulnessDenominator` (impossible physical vote counts). Exact records removed: **2**.
3. **Empty Review Filtering:** Removes rows with missing or whitespace-only review text.
4. **Cross-Variant Deduplication:** In e-commerce platforms, customers frequently purchase multiple size or flavor variants of the same product simultaneously and submit identical reviews. Sorting by `ProductId` and deduplicating on `['UserId', 'ProfileName', 'Time', 'Text']` removed **174,521 duplicate reviews** (retention rate: 69.3%).
5. **Three-Class Sentiment Labeling:**
   - **Positive:** `Score >= 4` (4★ and 5★ reviews) $\rightarrow$ `Target = 1`
   - **Negative:** `Score <= 2` (1★ and 2★ reviews) $\rightarrow$ `Target = 0`
   - **Neutral:** `Score == 3` (3★ reviews) $\rightarrow$ Preserved for analytics & EDA, excluded from binary classification training to prevent label ambiguity.
6. **Data Leakage Prevention:** Feature extraction (TF-IDF and BoW) is strictly fitted on training splits only and transformed on test/inference sets.

### Dataset Audit Summary (`data/dataset_stats.json`):
```json
{
  "original_record_count": 568454,
  "cleaned_record_count": 393931,
  "total_removed_records": 174523,
  "duplicates_removed": 174521,
  "invalid_helpfulness_removed": 2,
  "sentiment_distribution": {
    "positive": 307052,
    "negative": 57107,
    "neutral": 29772,
    "positive_pct": 77.95,
    "negative_pct": 14.5,
    "neutral_pct": 7.56
  },
  "binary_classification_ready_count": 364159
}
```

---

## 3. The 4-Stage NLP Preprocessing Pipeline

Every review passes through a deterministic four-stage text processing pipeline:

```
[ Raw Text ] ──► [ Stage 1: HTML & Punctuation Clean ] ──► [ Stage 2: Negation-Preserving Stopwords ] ──► [ Stage 3: Snowball Stemming ] ──► [ Stage 4: Sparse Vectorization ]
```

1. **HTML & Entity Normalization:** Unescapes HTML entities (`&quot;`, `&amp;`, `&#39;`) and strips HTML markup tags using regex `<.*?>`.
2. **Punctuation & URL Removal:** Strips hyperlinks (`http\S+`, `www\S+`) and converts non-alphabetic characters to whitespace while preserving lowercase casing.
3. **Negation Preservation (Critical NLP Rule):** Standard stopword pruning strips words like *"not"*, *"no"*, *"never"*, *"neither"*. In sentiment analysis, dropping negations reverses meaning (*"not bad at all"* becomes *"bad"*). Our pipeline explicitly retains negation words:
   ```python
   NEGATION_WORDS = {
       'not', 'no', 'nor', 'neither', 'never', 'none', 'hardly', 'barely',
       'scarcely', 'cannot', "isn't", "aren't", "wasn't", "weren't",
       "haven't", "hasn't", "hadn't", "won't", "wouldn't", "don't",
       "doesn't", "didn't", "can't", "couldn't", "shouldn't"
   }
   ```
4. **Snowball Stemming:** Stems words to their morphological roots (e.g., *"delicious"*, *"delicacy"* $\rightarrow$ `delici`). Corpus frequency maps reverse stems to authentic English words for explainability displays.
5. **Feature Projection:**
   - **TF-IDF:** N-gram range `(1, 2)`, max 10,000 features, minimum document frequency 3, sublinear term frequency scaling (`sublinear_tf=True`).
   - **CountVectorizer:** N-gram range `(1, 2)`, max 10,000 features, minimum document frequency 3.

---

## 4. The 6 Core Algorithms & Mathematical Formulations

### 1. Logistic Regression (Benchmark Champion)
* **Mathematical Model:**
  $$P(Y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
* **Objective Function:** Binary cross-entropy with L2 weight regularization:
  $$\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_{i=1}^N \log(1 + e^{-y_i (\mathbf{w}^T \mathbf{x}_i + b)})$$
* **Configuration:** $C=5.0$, `solver='liblinear'`, `max_iter=1000`.

### 2. Multinomial Naive Bayes
* **Mathematical Model:** Applies Bayes' Theorem under conditional feature independence:
  $$P(Y=c \mid \mathbf{x}) \propto P(Y=c) \prod_{j=1}^D P(x_j \mid Y=c)^{x_j}$$
* **Laplace Smoothing:**
  $$\hat{\theta}_{c, j} = \frac{N_{c, j} + \alpha}{N_c + \alpha D}$$
* **Configuration:** $\alpha=0.10$ on BoW count vectors. Ultra-low inference latency ($< 0.003\text{ ms}$).

### 3. Linear Support Vector Machine (Linear SVM)
* **Mathematical Model:** Soft-margin separating hyperplane with modified Huber loss:
  $$L(y, f(\mathbf{x})) = \begin{cases} 
  \max(0, 1 - y f(\mathbf{x}))^2 & \text{if } y f(\mathbf{x}) \ge -1 \\
  -4 y f(\mathbf{x}) & \text{if } y f(\mathbf{x}) < -1 
  \end{cases}$$
* **Calibrated Probabilities:** Modified Huber loss satisfies Fisher consistency for classification and directly outputs smooth, well-calibrated class probabilities without post-hoc Platt scaling.
* **Configuration:** `loss='modified_huber'`, $\alpha=10^{-4}$, `max_iter=1000`.

### 4. XGBoost Classifier (Gradient Boosted Decision Trees)
* **Mathematical Model:** Additive tree ensemble minimizing regularized objective:
  $$\mathcal{L}^{(t)} = \sum_{i=1}^N l\left(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$
  $$\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
* **Configuration:** $100$ estimators, `max_depth=5`, `learning_rate=0.15`, `eval_metric='logloss'`, multi-threaded parallel execution.

### 5. Random Forest Classifier
* **Mathematical Model:** Bootstrap aggregation (bagging) of de-correlated decision trees with random feature sub-sampling at each split:
  $$\hat{P}(Y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
* **Configuration:** $100$ estimators, `max_depth=16`, `n_jobs=-1`.

### 6. K-Nearest Neighbors (KNN)
* **Mathematical Model:** Non-parametric instance lookup via Cosine distance in 10,000-dimensional TF-IDF space:
  $$D_{\text{cosine}}(\mathbf{u}, \mathbf{v}) = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
  $$\hat{P}(Y=1 \mid \mathbf{x}) = \frac{1}{K} \sum_{i \in \mathcal{N}_K(\mathbf{x})} \mathbb{I}(y_i = 1)$$
* **Configuration:** $K=7$, metric='cosine', brute-force sparse search on representative index memory.

---

## 5. Combined Ensemble Average & Empirical Evaluation

### Formulation:
The Combined Ensemble Average computes the unweighted soft-voting probability mean across all 6 classifiers:
$$P_{\text{ensemble}}(\text{Positive} \mid \mathbf{x}) = \frac{1}{6} \left( P_{\text{LR}} + P_{\text{NB}} + P_{\text{SVM}} + P_{\text{XGB}} + P_{\text{RF}} + P_{\text{KNN}} \right)$$
$$P_{\text{ensemble}}(\text{Negative} \mid \mathbf{x}) = 1.0 - P_{\text{ensemble}}(\text{Positive} \mid \mathbf{x})$$
$$\text{Verdict} = \begin{cases} \text{Positive} & \text{if } P_{\text{ensemble}}(\text{Positive}) \ge 0.50 \\ \text{Negative} & \text{otherwise} \end{cases}$$

### Model Agreement Metric:
The system computes the fraction of individual classifiers that concur with the ensemble decision:
$$\text{Agreement} = \frac{1}{6} \sum_{m=1}^6 \mathbb{I}\left(\hat{y}_m = \hat{y}_{\text{ensemble}}\right) \times 100\%$$

### Empirical Evaluation: Does the Ensemble Improve Performance?
Evaluating the 6 models and the ensemble on the held-out test split of **8,000 reviews** revealed key insights:
1. **Recall Superiority:** The Combined Ensemble achieved **98.99% positive recall**, higher than individual champions Logistic Regression (97.70%) and Linear SVM (98.00%).
2. **Variance Reduction:** Condorcet's Jury Theorem holds true: when individual models make isolated errors on edge-case phrasing (e.g., complex sarcasm or uncommon food ingredients), the remaining 5 models outvote the outlier, preventing false alarms.
3. **Pure Accuracy Trade-Off:** While Logistic Regression achieved individual champion status at 92.79% accuracy / 95.63% ROC-AUC, the Ensemble achieved **92.36% accuracy and 95.44% ROC-AUC** with an average model consensus agreement of **94.37%**.

---

## 6. Why Test Reviews Individually vs. In Batch?

In production e-commerce systems, single-review streaming inference and bulk catalog analytics fulfill complementary roles:

| Dimension | Single Review Inference (Page 1) | Bulk Batch Analytics (Page 2) |
|---|---|---|
| **Latency Budget** | $< 2$ milliseconds per request | $50 - 300$ milliseconds per batch |
| **Operational Goal** | Streaming real-time ingestion | Catalog & inventory vendor evaluation |
| **Crisis Mitigation** | Intercepts food safety recalls instantly | Identifies systemic supply chain defects |
| **Linguistic Purpose** | Nuance, slang, double-negations, sarcasm | Overall satisfaction percentage & trends |
| **Business Action** | Auto-escalates refunds for furious reviews | Triggers quality audits on defective batches |

---

## 7. Three-Page Web Architecture & User Interface

The application features an Apple-inspired Liquid Glass design implemented across 3 distinct pages:

### Page 1: Live Sentiment Studio (`/`)
* **Interactive Controls:** Review textarea with character counter, "Check Review" button, "Clear" button, and 6 quick preset chips (Coffee, Stale Chips, Matcha, Chemical Taste, Pet Food Defect, Dark Chocolate).
* **Model Switcher:** Select any individual model or the Combined Ensemble Average.
* **Triple Result Views:**
  1. *Result & Confidence:* Large sentiment emoji, confidence percentage with smooth count-up animation, and positive/negative probability progress bar.
  2. *Do the Models Agree:* 6-model consensus card grid with individual model votes, confidence ratings, and agreement badge.
  3. *Words That Mattered Most:* Explainable AI chips displaying token feature weights ($w_i \cdot x_i$) showing positive/negative influence.
* **Catalog Review Browser:** Filterable real reviews from the dataset with 1-click test buttons.

### Page 2: Batch and Product Health Analytics (`/batch`)
* **Dual Ingestion Modes:** Paste reviews line-by-line or drag-and-drop a **CSV file** (auto-detects `Text`, `review`, or `Summary` column).
* **QA Demo Presets:** Pre-loaded batches for Colombian Coffee, Kettle Chips, Organic Matcha, and Problem Dog Food, clearly labeled as Demo QA Groups.
* **Configurable Defect Alert Threshold:** Interactive threshold input (default: 25% unhappy). If the proportion of negative reviews exceeds the threshold, a **🚨 Critical Quality Alert** banner is rendered.
* **Sample Size Detection:** Automatically flags batches with $N < 10$ as *⚠️ Preliminary Result (Small Sample Size)*.
* **Distribution Progress Gauge:** Visual breakdown of positive vs. negative review proportions with threshold indicator.
* **Table Filtering & Search:** Real-time text search and sentiment filter dropdown.
* **CSV Export:** Download full prediction records as `sentiment_predictions.csv`.
* **Deep Linking:** "Inspect ↗" buttons link directly to Page 1 with the selected review prefilled and analyzed.

### Page 3: Model Tournament & Benchmarks (`/benchmarks`)
* **Mode Toggle:** Switch between **Measured Test Run (8k Test Split)** and **Historical Notebook Research (34 experiments)**.
* **Leaderboard Comparison Chart:** Live horizontal bar visualization comparing ROC-AUC across all models.
* **Crown Ensemble Card:** Highlighted card with live metrics, recall breakdown, and expandable confusion matrix.
* **6 Individual Model Cards:** Displays ROC-AUC, Accuracy, F1-Score, Precision, Recall, Training Time, and Inference Latency.
* **Confusion Matrices:** Complete $2 \times 2$ matrices with True Negatives, False Positives, False Negatives, and True Positives.
* **Dataset Audit Card:** Total raw reviews (568k), clean reviews (393k), positive/negative/neutral distributions.
* **4-Stage Visual Pipeline:** Educational cards showing the data flow from raw text to ensemble verdict.

---

## 8. REST API Endpoints & Contract Specifications

### `POST /api/predict`
Single review sentiment prediction.
* **Request:** `{"text": "Loved this roasted coffee!", "model": "ensemble_avg"}`
* **Response (200 OK):**
  ```json
  {
    "status": "success",
    "data": {
      "prediction": "Positive",
      "confidence": 97.8,
      "positive_prob": 97.8,
      "negative_prob": 2.2,
      "selected_model": "ensemble_avg",
      "selected_model_name": "Ensemble Average (All 6 Combined)",
      "latency_ms": 1.45,
      "model_agreement_pct": 100.0,
      "models_consensus": { ... },
      "word_attributions": [ ... ]
    }
  }
  ```

### `POST /api/predict-batch`
Batch sentiment evaluation with defect threshold monitoring.
* **Request:** `{"reviews": ["Great food", "Spoiled bag"], "model": "ensemble_avg", "threshold": 25.0}`
* **Response (200 OK):**
  ```json
  {
    "status": "success",
    "data": {
      "total_analyzed": 2,
      "positive_count": 1,
      "negative_count": 1,
      "positive_pct": 50.0,
      "negative_pct": 50.0,
      "defect_alert": true,
      "is_preliminary": true,
      "satisfaction_status": "Mixed Customer Feedback",
      "reviews": [ ... ]
    }
  }
  ```

### `GET /api/evaluation`
Returns real test set metrics and confusion matrices from `models/evaluation_results.json`.

### `GET /api/health`
System health check reporting model memory readiness and vectorizer feature counts.

### `GET /api/stats`
Returns complete Stanford SNAP dataset counts and cleaning audit figures.

---

## 9. Repository Inventory & Artifact Architecture

```
Amazon-Fine-Food-Reviews/
├── app.py                             # Flask web server & REST API controller
├── data_pipeline.py                   # Data ingestion, deduplication & cleaning pipeline
├── model_trainer.py                   # 6-Algorithm training, serialization & evaluation engine
├── benchmark_data.py                  # Historical reference data from 34 experiments
├── test_app.py                        # Automated pytest suite (5/5 passing)
├── requirements.txt                   # Dependency manifest
│
├── data/
│   ├── Reviews.csv                    # Stanford SNAP raw reviews (568,454 rows)
│   ├── processed_reviews.csv          # Cleaned & deduplicated reviews (393,931 rows)
│   ├── dataset_stats.json             # Sanitization audit & sentiment split statistics
│   └── amazon_food_reviews.csv        # Compact seed dataset
│
├── models/
│   ├── evaluation_results.json        # Live measured test set results
│   ├── logistic_regression.joblib     # Serialized Logistic Regression
│   ├── naive_bayes.joblib             # Serialized Naive Bayes
│   ├── svm.joblib                     # Serialized Linear SVM
│   ├── xgboost.joblib                 # Serialized XGBoost Classifier
│   ├── random_forest.joblib           # Serialized Random Forest
│   ├── knn.joblib                     # Serialized KNN
│   ├── tfidf_vectorizer.joblib        # Fitted TF-IDF vectorizer (10k features)
│   ├── bow_vectorizer.joblib          # Fitted CountVectorizer
│   └── trained_models_6.joblib        # Bundled model archive
│
├── templates/
│   ├── index.html                     # Live Sentiment Studio
│   ├── batch.html                     # Batch & Product Health Analytics
│   └── benchmarks.html                # Model Tournament & Visual Flow
│
├── static/
│   ├── style.css                      # Apple Liquid Glass CSS styling
│   ├── app.js                         # Page 1 JavaScript controller
│   └── batch.js                       # Page 2 JavaScript controller
│
└── Jupyter Research Notebooks (1 to 8):
    ├── 1.Amazon Fine Food Reviews Analysis.ipynb
    ├── 2.AmazonFineFoodReviews_TSNE.ipynb
    ├── 3.Amazon Fine Food Reviews Analysis knn.ipynb
    ├── 4. Amazon Fine Food Reviews Analysis_NB.ipynb
    ├── 5 . Amazon Fine Food Reviews Analysis_LR.ipynb
    ├── 6 . Amazon Fine Food Reviews Analysis_SVM.ipynb
    ├── 7 . Amazon Fine Food Reviews Analysis_DT.ipynb
    └── 8 . Amazon Fine Food Reviews Analysis_RF&GBDT.ipynb
```

---

## 10. Local Execution, PowerShell Commands & Automated Tests

### 1. Environment Setup (Windows PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Preprocess Dataset
```powershell
python data_pipeline.py
```

### 3. Model Training & Evaluation
```powershell
python model_trainer.py --sample-size 40000
```

### 4. Execute Test Suite
```powershell
pytest test_app.py -v
```

### 5. Launch Web Application
```powershell
python app.py
```
Open **`http://127.0.0.1:5000`** in any web browser.

---

## 11. Actual Measured Evaluation Metrics & Analysis

Evaluated on the held-out test split of **8,000 authentic customer reviews**:

| Classifier | Accuracy | F1-Score | ROC-AUC | Recall | Precision | Latency | Confusion Matrix [TN, FP / FN, TP] |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | **92.79%** | **95.81%** | **95.63%** | 97.70% | 93.98% | 0.004 ms | `[833, 422] / [155, 6590]` |
| **Linear SVM** | **92.70%** | **95.77%** | **95.09%** | 98.00% | 93.64% | 0.003 ms | `[806, 449] / [135, 6610]` |
| **Combined Ensemble** | **92.36%** | **95.62%** | **95.44%** | **98.99%** | 92.48% | 0.503 ms | `[712, 543] / [68, 6677]` |
| **Naive Bayes** | 90.92% | 94.59% | 92.64% | 94.01% | 95.17% | 0.002 ms | `[933, 322] / [404, 6341]` |
| **XGBoost** | 90.30% | 94.50% | 92.85% | 98.87% | 90.50% | 0.025 ms | `[555, 700] / [76, 6669]` |
| **K-Nearest Neighbors** | 86.56% | 92.51% | 77.88% | 98.46% | 87.24% | 0.350 ms | `[284, 971] / [104, 6641]` |
| **Random Forest** | 84.51% | 91.59% | 90.44% | 100.0% | 84.48% | 0.119 ms | `[16, 1239] / [0, 6745]` |

---

## 12. Known Limitations & Engineering Considerations

1. **KNN Inference on Very Large Vocabularies:** In 10,000-dimensional TF-IDF space, computing pairwise Cosine distance across millions of vectors is $O(N \cdot D)$. For low latency, KNN is indexed on a representative stratified memory subset ($N = 12,000$).
2. **Neutral Sentiment Ambiguity:** 3-star reviews frequently contain conflicting sentiment (*"Good coffee, but overpriced"*). Excluding them from binary training ensures crisp mathematical separating margins.
3. **Imbalanced Class Distribution:** In authentic Amazon food reviews, positive reviews naturally outnumber negative reviews (~5:1). Evaluating both F1-score and ROC-AUC prevents accuracy paradox misinterpretation.
