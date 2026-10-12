"""
Model Trainer and Inference Engine for Amazon Fine Food Reviews.
Implements the 6 Core Algorithms:
1. Logistic Regression (TF-IDF features)
2. Multinomial Naive Bayes (CountVectorizer features)
3. Linear Support Vector Machine (TF-IDF features with smooth calibrated probabilities)
4. XGBoost Classifier (TF-IDF sparse features)
5. Random Forest Classifier (TF-IDF features)
6. K-Nearest Neighbors (TF-IDF features with cosine distance)

Plus:
- Combined Ensemble Average:
    P_ensemble = (P_LR + P_NB + P_SVM + P_XGB + P_RF + P_KNN) / 6
- Comprehensive evaluation metrics on held-out test set:
    Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Latency
- Explainability via token feature weights and log-odds
- Resumable/incremental model saving to models/
- Serialization to models/evaluation_results.json
"""

import os
import re
import html
import json
import time
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from nltk.stem import SnowballStemmer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_CSV = os.path.join(DATA_DIR, 'processed_reviews.csv')
EVALUATION_JSON = os.path.join(MODELS_DIR, 'evaluation_results.json')
os.makedirs(MODELS_DIR, exist_ok=True)

stemmer = SnowballStemmer('english')

NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'none', 'hardly', 'barely',
    'scarcely', 'cannot', "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "won't", "wouldn't", "don't",
    "doesn't", "didn't", "can't", "couldn't", "shouldn't"
}
CUSTOM_STOP_WORDS = set(ENGLISH_STOP_WORDS) - NEGATION_WORDS

def clean_html(text):
    """Strip HTML tags and unescape entities"""
    text = html.unescape(str(text))
    return re.sub(r'<.*?>', ' ', text)

def clean_punctuation(text):
    """Remove URLs and punctuation characters"""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    return text

def preprocess_text(text, return_tokens=False):
    """
    NLP preprocessing pipeline:
    1. Unescape & strip HTML
    2. Lowercase & strip punctuation/URLs
    3. Filter stopwords (strictly preserving negations)
    4. Snowball Stemming
    """
    raw_step = str(text) if text is not None else ""
    html_cleaned = clean_html(raw_step)
    punc_cleaned = clean_punctuation(html_cleaned).lower()

    tokens = []
    stemmed_tokens = []

    for word in punc_cleaned.split():
        if len(word) > 2 and word not in CUSTOM_STOP_WORDS:
            tokens.append(word)
            stemmed_word = stemmer.stem(word)
            stemmed_tokens.append(stemmed_word)

    final_cleaned_str = " ".join(stemmed_tokens)

    if return_tokens:
        return {
            "raw": raw_step,
            "html_cleaned": html_cleaned.strip(),
            "punc_cleaned": punc_cleaned.strip(),
            "tokens": tokens,
            "stemmed_tokens": stemmed_tokens,
            "cleaned_text": final_cleaned_str
        }
    return final_cleaned_str

def train_and_save_models(data_path=None, sample_size=50000, test_size=0.2, random_state=42, skip_trained=False):
    """
    Trains all 6 algorithms on the real Amazon Fine Food Reviews dataset.
    Evaluates individual models and the combined ensemble on the held-out test set.
    Saves serialized model artifacts and evaluation_results.json.
    """
    t_global_start = time.time()
    if data_path is None:
        if os.path.exists(PROCESSED_CSV):
            data_path = PROCESSED_CSV
        elif os.path.exists(os.path.join(DATA_DIR, 'Reviews.csv')):
            # Run pipeline first
            from data_pipeline import run_preprocessing_pipeline
            print("Processed dataset missing. Executing data pipeline on Reviews.csv...")
            run_preprocessing_pipeline()
            data_path = PROCESSED_CSV
        else:
            data_path = os.path.join(DATA_DIR, 'amazon_food_reviews.csv')

    print(f"\n=======================================================")
    print(f"  AMAZON FINE FOOD REVIEWS - 6-MODEL TRAINING PIPELINE")
    print(f"=======================================================")
    print(f"Dataset path: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} total records.")

    # Exclude neutral reviews (Score = 3) for binary sentiment classification
    if 'Sentiment' in df.columns:
        df = df[df['Sentiment'] != 'neutral'].copy()
        y = (df['Sentiment'] == 'positive').astype(int).values
    else:
        df = df[df['Score'] != 3].copy()
        y = (df['Score'] > 3).astype(int).values

    # Never use Score as input feature; use Text (or Summary + Text)
    texts = df['Text'].fillna('').astype(str).values

    total_avail = len(texts)
    print(f"Binary classification reviews available: {total_avail:,} (Pos: {(y==1).sum():,}, Neg: {(y==0).sum():,})")

    # Stratified subsampling if sample_size is set and smaller than total
    if sample_size and sample_size < total_avail:
        print(f"Using stratified sample of {sample_size:,} reviews for efficient training...")
        train_idx, _ = train_test_split(
            np.arange(total_avail), train_size=sample_size,
            stratify=y, random_state=random_state
        )
        texts = texts[train_idx]
        y = y[train_idx]

    # Preprocessing
    print("\n[Step 1/4] Preprocessing text (cleaning HTML, punctuation, negations, stemming)...")
    cleaned_texts = [preprocess_text(t) for t in texts]

    # Stratified Train/Test Split (ensuring 0 data leakage)
    print(f"[Step 2/4] Splitting data into Train ({int((1-test_size)*100)}%) and Test ({int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_texts, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set: {len(X_train):,} reviews (Pos: {(y_train==1).sum():,}, Neg: {(y_train==0).sum():,})")
    print(f"Testing set:  {len(X_test):,} reviews (Pos: {(y_test==1).sum():,}, Neg: {(y_test==0).sum():,})")

    # Fit Vectorizers strictly on training data
    print("\n[Step 3/4] Fitting feature extractors on training data...")
    tfidf_file = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
    bow_file = os.path.join(MODELS_DIR, 'bow_vectorizer.joblib')

    if skip_trained and os.path.exists(tfidf_file) and os.path.exists(bow_file):
        print("Loading cached vectorizers from disk...")
        tfidf = joblib.load(tfidf_file)
        bow = joblib.load(bow_file)
    else:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=3, sublinear_tf=True)
        bow = CountVectorizer(ngram_range=(1, 2), max_features=10000, min_df=3)

        print("Fitting TF-IDF Vectorizer...")
        X_tr_tfidf = tfidf.fit_transform(X_train)
        print(f"TF-IDF shape: {X_tr_tfidf.shape}")

        print("Fitting Count (BoW) Vectorizer...")
        X_tr_bow = bow.fit_transform(X_train)
        print(f"Count (BoW) shape: {X_tr_bow.shape}")

        joblib.dump(tfidf, tfidf_file)
        joblib.dump(bow, bow_file)
        print("Saved vectorizers to models/ directory.")

    # Transform features
    X_tr_tfidf = tfidf.transform(X_train)
    X_ts_tfidf = tfidf.transform(X_test)
    X_tr_bow = bow.transform(X_train)
    X_ts_bow = bow.transform(X_test)

    # Dictionary to hold models and predictions for ensemble evaluation
    models = {}
    metrics = {}
    test_probs = {}

    def evaluate_model(name, y_true, probs, train_time, latency_ms):
        preds = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_true, preds).tolist()
        acc = round(accuracy_score(y_true, preds) * 100, 2)
        prec = round(precision_score(y_true, preds, zero_division=0) * 100, 2)
        rec = round(recall_score(y_true, preds, zero_division=0) * 100, 2)
        f1 = round(f1_score(y_true, preds, zero_division=0) * 100, 2)
        auc = round(roc_auc_score(y_true, probs) * 100, 2)
        return {
            "name": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "training_time_sec": round(train_time, 2),
            "inference_latency_ms": round(latency_ms, 3)
        }

    def measure_latency(model, X_sample, is_bow=False):
        n_eval = min(500, X_sample.shape[0])
        sub = X_sample[:n_eval]
        t0 = time.perf_counter()
        _ = model.predict_proba(sub)
        t_diff = (time.perf_counter() - t0) * 1000 / n_eval
        return t_diff

    # ---------------- 1. Logistic Regression ----------------
    print("\n--- Model 1/6: Logistic Regression (TF-IDF) ---")
    m1_file = os.path.join(MODELS_DIR, 'logistic_regression.joblib')
    if skip_trained and os.path.exists(m1_file):
        print("Loading cached Logistic Regression...")
        lr = joblib.load(m1_file)
        train_time = 0.0
    else:
        t0 = time.time()
        lr = LogisticRegression(C=5.0, max_iter=1000, random_state=random_state, solver='liblinear')
        lr.fit(X_tr_tfidf, y_train)
        train_time = time.time() - t0
        joblib.dump(lr, m1_file)
    lat = measure_latency(lr, X_ts_tfidf)
    lr_probs = lr.predict_proba(X_ts_tfidf)[:, 1]
    test_probs['logistic_regression'] = lr_probs
    models['logistic_regression'] = lr
    m_info = evaluate_model("Logistic Regression", y_test, lr_probs, train_time, lat)
    m_info['badge'] = "🥇 Champion"
    m_info['vectorizer'] = "TF-IDF"
    m_info['description'] = "Optimal margin on sparse text vectors with L2 regularization."
    metrics['logistic_regression'] = m_info
    print(f"LR Results -> Acc: {m_info['accuracy']}%, F1: {m_info['f1_score']}%, AUC: {m_info['roc_auc']}%")

    # ---------------- 2. Multinomial Naive Bayes ----------------
    print("\n--- Model 2/6: Multinomial Naive Bayes (CountVectorizer) ---")
    m2_file = os.path.join(MODELS_DIR, 'naive_bayes.joblib')
    if skip_trained and os.path.exists(m2_file):
        print("Loading cached Naive Bayes...")
        nb = joblib.load(m2_file)
        train_time = 0.0
    else:
        t0 = time.time()
        nb = MultinomialNB(alpha=0.1)
        nb.fit(X_tr_bow, y_train)
        train_time = time.time() - t0
        joblib.dump(nb, m2_file)
    lat = measure_latency(nb, X_ts_bow, is_bow=True)
    nb_probs = nb.predict_proba(X_ts_bow)[:, 1]
    test_probs['naive_bayes'] = nb_probs
    models['naive_bayes'] = nb
    m_info = evaluate_model("Naive Bayes", y_test, nb_probs, train_time, lat)
    m_info['badge'] = "⚡ Fast NLP"
    m_info['vectorizer'] = "BoW"
    m_info['description'] = "Fast conditional probability modeling with Laplace smoothing."
    metrics['naive_bayes'] = m_info
    print(f"NB Results -> Acc: {m_info['accuracy']}%, F1: {m_info['f1_score']}%, AUC: {m_info['roc_auc']}%")

    # ---------------- 3. Support Vector Machine ----------------
    print("\n--- Model 3/6: Linear SVM (Modified Huber calibrated probabilities) ---")
    m3_file = os.path.join(MODELS_DIR, 'svm.joblib')
    if skip_trained and os.path.exists(m3_file):
        print("Loading cached Linear SVM...")
        svm = joblib.load(m3_file)
        train_time = 0.0
    else:
        t0 = time.time()
        svm = SGDClassifier(loss='modified_huber', alpha=1e-4, max_iter=1000, random_state=random_state)
        svm.fit(X_tr_tfidf, y_train)
        train_time = time.time() - t0
        joblib.dump(svm, m3_file)
    lat = measure_latency(svm, X_ts_tfidf)
    svm_probs = svm.predict_proba(X_ts_tfidf)[:, 1]
    test_probs['svm'] = svm_probs
    models['svm'] = svm
    m_info = evaluate_model("Support Vector Machine (SVM)", y_test, svm_probs, train_time, lat)
    m_info['badge'] = "🛡️ Max Margin"
    m_info['vectorizer'] = "TF-IDF"
    m_info['description'] = "Maximum margin hyperplane with smooth calibrated probabilities."
    metrics['svm'] = m_info
    print(f"SVM Results -> Acc: {m_info['accuracy']}%, F1: {m_info['f1_score']}%, AUC: {m_info['roc_auc']}%")

    # ---------------- 4. XGBoost Classifier ----------------
    print("\n--- Model 4/6: XGBoost Classifier (TF-IDF) ---")
    m4_file = os.path.join(MODELS_DIR, 'xgboost.joblib')
    if skip_trained and os.path.exists(m4_file):
        print("Loading cached XGBoost...")
        xgb = joblib.load(m4_file)
        train_time = 0.0
    else:
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.15,
            eval_metric='logloss', random_state=random_state, n_jobs=-1
        )
        xgb.fit(X_tr_tfidf, y_train)
        train_time = time.time() - t0
        joblib.dump(xgb, m4_file)
    lat = measure_latency(xgb, X_ts_tfidf)
    xgb_probs = xgb.predict_proba(X_ts_tfidf)[:, 1]
    test_probs['xgboost'] = xgb_probs
    models['xgboost'] = xgb
    m_info = evaluate_model("XGBoost", y_test, xgb_probs, train_time, lat)
    m_info['badge'] = "🚀 Gradient Boosted"
    m_info['vectorizer'] = "TF-IDF"
    m_info['description'] = "Ensemble of shallow decision trees sequentially minimizing residual errors."
    metrics['xgboost'] = m_info
    print(f"XGB Results -> Acc: {m_info['accuracy']}%, F1: {m_info['f1_score']}%, AUC: {m_info['roc_auc']}%")

    # ---------------- 5. Random Forest Classifier ----------------
    print("\n--- Model 5/6: Random Forest Classifier (TF-IDF) ---")
    m5_file = os.path.join(MODELS_DIR, 'random_forest.joblib')
    if skip_trained and os.path.exists(m5_file):
        print("Loading cached Random Forest...")
        rf = joblib.load(m5_file)
        train_time = 0.0
    else:
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=16, random_state=random_state, n_jobs=-1
        )
        rf.fit(X_tr_tfidf, y_train)
        train_time = time.time() - t0
        joblib.dump(rf, m5_file)
    lat = measure_latency(rf, X_ts_tfidf)
    rf_probs = rf.predict_proba(X_ts_tfidf)[:, 1]
    test_probs['random_forest'] = rf_probs
    models['random_forest'] = rf
    m_info = evaluate_model("Random Forest", y_test, rf_probs, train_time, lat)
    m_info['badge'] = "🌲 Bagging Trees"
    m_info['vectorizer'] = "TF-IDF"
    m_info['description'] = "Bootstrap bagging ensemble of de-correlated decision trees."
    metrics['random_forest'] = m_info
    print(f"RF Results -> Acc: {m_info['accuracy']}%, F1: {m_info['f1_score']}%, AUC: {m_info['roc_auc']}%")

    # ---------------- 6. K-Nearest Neighbors (KNN) ----------------
    print("\n--- Model 6/6: K-Nearest Neighbors (TF-IDF Cosine Distance) ---")
    m6_file = os.path.join(MODELS_DIR, 'knn.joblib')
    if skip_trained and os.path.exists(m6_file):
        print("Loading cached KNN...")
        knn = joblib.load(m6_file)
        train_time = 0.0
    else:
        t0 = time.time()
        # For memory efficiency and responsive inference, fit KNN on representative subset of train set
        knn_n_train = min(12000, X_tr_tfidf.shape[0])
        knn = KNeighborsClassifier(n_neighbors=7, metric='cosine', algorithm='brute')
        knn.fit(X_tr_tfidf[:knn_n_train], y_train[:knn_n_train])
        train_time = time.time() - t0
        joblib.dump(knn, m6_file)
    lat = measure_latency(knn, X_ts_tfidf[:300])
    knn_probs = knn.predict_proba(X_ts_tfidf)[:, 1]
    test_probs['knn'] = knn_probs
    models['knn'] = knn
    m_info = evaluate_model("K-Nearest Neighbors (KNN)", y_test, knn_probs, train_time, lat)
    m_info['badge'] = "📍 Spatial Distance"
    m_info['vectorizer'] = "TF-IDF Cosine"
    m_info['description'] = "Instance-based nearest neighbor search using cosine metric in TF-IDF space."
    metrics['knn'] = m_info
    print(f"KNN Results -> Acc: {m_info['accuracy']}%, F1: {m_info['f1_score']}%, AUC: {m_info['roc_auc']}%")

    # ---------------- COMBINED ENSEMBLE AVERAGE ----------------
    print("\n--- [Step 4/4] Evaluating Combined Ensemble Average (All 6 Models) ---")
    all_prob_arrays = [
        test_probs['logistic_regression'],
        test_probs['naive_bayes'],
        test_probs['svm'],
        test_probs['xgboost'],
        test_probs['random_forest'],
        test_probs['knn']
    ]
    ensemble_probs = np.mean(all_prob_arrays, axis=0)
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    # Compute agreement on test set
    individual_preds = [np.where(p >= 0.5, 1, 0) for p in all_prob_arrays]
    # For each sample, how many models agreed with the ensemble prediction?
    agreements = [
        sum(individual_preds[m][i] == ensemble_preds[i] for m in range(6)) / 6.0
        for i in range(len(ensemble_preds))
    ]
    avg_test_agreement = round(float(np.mean(agreements)) * 100, 2)

    ensemble_cm = confusion_matrix(y_test, ensemble_preds).tolist()
    ensemble_acc = round(accuracy_score(y_test, ensemble_preds) * 100, 2)
    ensemble_prec = round(precision_score(y_test, ensemble_preds, zero_division=0) * 100, 2)
    ensemble_rec = round(recall_score(y_test, ensemble_preds, zero_division=0) * 100, 2)
    ensemble_f1 = round(f1_score(y_test, ensemble_preds, zero_division=0) * 100, 2)
    ensemble_auc = round(roc_auc_score(y_test, ensemble_probs) * 100, 2)
    ensemble_latency = round(sum(m['inference_latency_ms'] for m in metrics.values()), 3)

    metrics['ensemble_avg'] = {
        "name": "Combined Ensemble Average",
        "badge": "👑 Consensus Vote",
        "vectorizer": "Multi-Modal",
        "accuracy": ensemble_acc,
        "precision": ensemble_prec,
        "recall": ensemble_rec,
        "f1_score": ensemble_f1,
        "roc_auc": ensemble_auc,
        "confusion_matrix": ensemble_cm,
        "training_time_sec": round(sum(m['training_time_sec'] for m in metrics.values()), 2),
        "inference_latency_ms": ensemble_latency,
        "average_agreement_pct": avg_test_agreement,
        "description": "Probability average of all 6 distinct classifiers: P_ens = 1/6 ∑ P_i.",
        "improvement_notes": (
            f"Ensemble achieves {ensemble_acc}% test accuracy and {ensemble_auc}% ROC-AUC. "
            f"Average model agreement on held-out test data is {avg_test_agreement}%."
        )
    }

    print(f"\n=======================================================")
    print(f"  EVALUATION SUMMARY ON HELD-OUT TEST DATA ({len(y_test):,} reviews)")
    print(f"=======================================================")
    print(f"{'Model':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Latency':<10}")
    print("-" * 78)
    for k, v in metrics.items():
        print(f"{v['name']:<30} | {v['accuracy']:>8}% | {v['f1_score']:>8}% | {v['roc_auc']:>8}% | {v['inference_latency_ms']:>7.2f}ms")
    print("=" * 78)

    # Save legacy joblib bundles for immediate backwards compatibility
    joblib.dump(models, os.path.join(MODELS_DIR, 'trained_models_6.joblib'))
    joblib.dump(metrics, os.path.join(MODELS_DIR, 'model_metrics_6.joblib'))

    # Save rich evaluation results JSON
    eval_results = {
        "dataset_version": "Amazon Fine Food Reviews (Stanford SNAP)",
        "total_binary_reviews": total_avail,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "test_split_ratio": test_size,
        "random_seed": random_state,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_training_wall_clock_sec": round(time.time() - t_global_start, 2),
        "models": metrics
    }
    with open(EVALUATION_JSON, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nEvaluation results written to {EVALUATION_JSON}")

    return models, tfidf, bow, metrics

# Global memory caches for low-latency Flask inference
_CACHED_MODELS = None
_CACHED_TFIDF = None
_CACHED_BOW = None
_CACHED_METRICS = None

def get_models():
    """
    Load and cache trained models in memory.
    Guarantees models are initialized once and reused across all requests.
    """
    global _CACHED_MODELS, _CACHED_TFIDF, _CACHED_BOW, _CACHED_METRICS
    if _CACHED_MODELS is None:
        models_file = os.path.join(MODELS_DIR, 'trained_models_6.joblib')
        tfidf_file = os.path.join(MODELS_DIR, 'tfidf_vectorizer.joblib')
        bow_file = os.path.join(MODELS_DIR, 'bow_vectorizer.joblib')
        metrics_file = os.path.join(MODELS_DIR, 'model_metrics_6.joblib')

        if not (os.path.exists(models_file) and os.path.exists(tfidf_file) and os.path.exists(bow_file)):
            print("Model artifacts not found on disk. Initiating training run...")
            train_and_save_models(sample_size=40000)

        _CACHED_MODELS = joblib.load(models_file)
        _CACHED_TFIDF = joblib.load(tfidf_file)
        _CACHED_BOW = joblib.load(bow_file)
        if os.path.exists(metrics_file):
            _CACHED_METRICS = joblib.load(metrics_file)
        else:
            _CACHED_METRICS = {}
    return _CACHED_MODELS, _CACHED_TFIDF, _CACHED_BOW, _CACHED_METRICS

def predict_review(text, selected_model='ensemble_avg'):
    """
    Runs sentiment inference across all 6 models + Combined Ensemble Average.
    Returns:
    - active prediction (Positive or Negative)
    - confidence percentage
    - 6-model individual predictions and probabilities
    - ensemble agreement percentage
    - word attributions / explainability
    - latency in milliseconds
    """
    start_time = time.perf_counter()
    models, tfidf, bow, metrics = get_models()

    trace = preprocess_text(text, return_tokens=True)
    cleaned_str = trace['cleaned_text']

    if not cleaned_str.strip():
        return {
            "prediction": "Neutral",
            "confidence": 50.0,
            "positive_prob": 50.0,
            "negative_prob": 50.0,
            "selected_model": selected_model,
            "selected_model_name": "Ensemble (Average of All)",
            "latency_ms": 0.5,
            "trace": trace,
            "models_consensus": {},
            "model_agreement_pct": 100.0,
            "word_attributions": []
        }

    vec_tfidf = tfidf.transform([cleaned_str])
    vec_bow = bow.transform([cleaned_str])

    consensus = {}
    pos_probs_list = []
    model_keys = ['logistic_regression', 'naive_bayes', 'svm', 'xgboost', 'random_forest', 'knn']

    missing_models = []
    for m_key in model_keys:
        if m_key not in models:
            missing_models.append(m_key)
            continue
        m_obj = models[m_key]
        if m_key == 'naive_bayes':
            probs = m_obj.predict_proba(vec_bow)[0]
        else:
            probs = m_obj.predict_proba(vec_tfidf)[0]

        pos_p = float(probs[1])
        pos_probs_list.append(pos_p)

        meta = metrics.get(m_key, {})
        m_name = meta.get('name', m_key.replace('_', ' ').title())
        m_badge = meta.get('badge', '')

        consensus[m_key] = {
            "name": m_name,
            "badge": m_badge,
            "prediction": "Positive" if pos_p >= 0.5 else "Negative",
            "pos_prob": round(pos_p * 100, 1),
            "neg_prob": round((1 - pos_p) * 100, 1),
            "confidence": round(max(pos_p, 1 - pos_p) * 100, 1)
        }

    if missing_models:
        print(f"Warning: Missing models in consensus calculation: {missing_models}")

    # Ensemble Average calculation
    if pos_probs_list:
        avg_pos_prob = float(np.mean(pos_probs_list))
    else:
        avg_pos_prob = 0.5

    avg_neg_prob = 1.0 - avg_pos_prob
    ensemble_verdict = "Positive" if avg_pos_prob >= 0.5 else "Negative"
    ensemble_conf = round(max(avg_pos_prob, avg_neg_prob) * 100, 1)

    # Agreement percentage: % of models whose binary prediction agrees with ensemble verdict
    agreed_count = sum(1 for m_key, res in consensus.items() if res['prediction'] == ensemble_verdict)
    total_models_count = len(consensus)
    agreement_pct = round((agreed_count / total_models_count * 100), 1) if total_models_count > 0 else 100.0

    consensus['ensemble_avg'] = {
        "name": "Ensemble Average (All 6 Combined)",
        "badge": "👑 Consensus Vote",
        "prediction": ensemble_verdict,
        "pos_prob": round(avg_pos_prob * 100, 1),
        "neg_prob": round(avg_neg_prob * 100, 1),
        "confidence": ensemble_conf,
        "agreement_pct": agreement_pct
    }

    if selected_model in consensus:
        active_res = consensus[selected_model]
        active_name = active_res['name']
    else:
        active_res = consensus['ensemble_avg']
        active_name = active_res['name']
        selected_model = 'ensemble_avg'

    # Explainability: compute word contributions
    # Uses Logistic Regression coefficient weights & token presence
    word_attributions = []
    if 'logistic_regression' in models:
        lr = models['logistic_regression']
        feature_names = tfidf.get_feature_names_out()
        cx = vec_tfidf.tocoo()
        for col, val in zip(cx.col, cx.data):
            feat_name = feature_names[col]
            weight = lr.coef_[0][col]
            contribution = weight * val
            word_attributions.append({
                "word": feat_name,
                "contribution": round(float(contribution), 4),
                "sentiment": "Positive" if contribution > 0 else "Negative",
                "magnitude": round(abs(float(contribution)), 4),
                "explanation": f"Weight {weight:+.2f} on TF-IDF {val:.2f}"
            })
        word_attributions.sort(key=lambda x: x['magnitude'], reverse=True)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "prediction": active_res['prediction'],
        "confidence": active_res['confidence'],
        "positive_prob": active_res['pos_prob'],
        "negative_prob": active_res['neg_prob'],
        "selected_model": selected_model,
        "selected_model_name": active_name,
        "latency_ms": elapsed_ms,
        "trace": trace,
        "models_consensus": consensus,
        "model_agreement_pct": agreement_pct,
        "word_attributions": word_attributions[:12]
    }

def predict_batch_reviews(reviews_list, selected_model='ensemble_avg'):
    """
    Performs batch sentiment prediction across an array of review texts.
    Returns aggregated metrics and individual review records.
    """
    results = []
    pos_count = 0
    neg_count = 0
    total_time = 0.0

    for idx, text in enumerate(reviews_list):
        text_str = str(text).strip()
        if not text_str:
            continue
        res = predict_review(text_str, selected_model=selected_model)
        if res['prediction'] == 'Positive':
            pos_count += 1
        elif res['prediction'] == 'Negative':
            neg_count += 1
        total_time += res['latency_ms']
        results.append({
            "index": idx + 1,
            "text": text_str[:160] + ('...' if len(text_str) > 160 else ''),
            "full_text": text_str,
            "prediction": res['prediction'],
            "confidence": res['confidence'],
            "positive_prob": res['positive_prob'],
            "negative_prob": res['negative_prob'],
            "agreement_pct": res.get('model_agreement_pct', 100.0)
        })

    total = len(results)
    return {
        "total_analyzed": total,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "positive_pct": round((pos_count / total * 100), 1) if total > 0 else 0,
        "negative_pct": round((neg_count / total * 100), 1) if total > 0 else 0,
        "avg_latency_ms": round(total_time / total, 2) if total > 0 else 0,
        "reviews": results
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate the 6 Amazon Review Classifiers")
    parser.add_argument('--data', type=str, default=None, help="Path to input processed dataset")
    parser.add_argument('--sample-size', type=int, default=50000, help="Stratified sample size (default 50,000 for fast training)")
    parser.add_argument('--full', action='store_true', help="Train on all 364k binary reviews")
    parser.add_argument('--skip-trained', action='store_true', help="Skip already trained models on disk")
    args = parser.parse_args()

    s_size = None if args.full else args.sample_size
    train_and_save_models(data_path=args.data, sample_size=s_size, skip_trained=args.skip_trained)

    # Sanity inference test
    sample_pos = "Five stars! The whole beans are fresh with a lovely oily sheen, rich cocoa notes."
    sample_neg = "Completely stale and expired. Bag arrived punctured and smelled like chemicals."
    print("\n--- Test Inferences ---")
    r1 = predict_review(sample_pos)
    print(f"Positive review -> Verdict: {r1['prediction']} ({r1['confidence']}%), Agreement: {r1['model_agreement_pct']}%")
    r2 = predict_review(sample_neg)
    print(f"Negative review -> Verdict: {r2['prediction']} ({r2['confidence']}%), Agreement: {r2['model_agreement_pct']}%")
