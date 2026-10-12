"""
Flask Web Application for Amazon Fine Food Reviews Sentiment Intelligence.
Serves:
- Page 1: Live Sentiment Studio (/)
- Page 2: Batch & Product Health Analytics (/batch)
- Page 3: 6-Algorithm Benchmark Tournament & Visual Architecture (/benchmarks)
REST API Endpoints:
- /api/predict: Live inference across 6 algorithms + Combined Ensemble Average
- /api/predict-batch: Bulk review evaluation with satisfaction health & defect alerts
- /api/evaluation: Live measured test set metrics from models/evaluation_results.json
- /api/benchmark: Historical notebook benchmark comparison
- /api/health: System and model readiness health check
- /api/stats: Complete dataset ingestion and preprocessing statistics
- /api/top-words: Explainability token feature weights
- /api/eda: Exploratory data analysis distributions
- /api/reviews: Paginated reviews explorer
- /api/samples: Curated preset testing reviews
"""

import os
import re
import json
import time
import pandas as pd
from flask import Flask, render_template, jsonify, request

from benchmark_data import PROJECT_STATS, BENCHMARK_RESULTS, NOTEBOOK_CHAPTERS, SAMPLE_REVIEWS
from model_trainer import predict_review, predict_batch_reviews, get_models

app = Flask(__name__, static_folder='static', template_folder='templates')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATS_FILE = os.path.join(DATA_DIR, 'dataset_stats.json')
EVAL_RESULTS_FILE = os.path.join(MODELS_DIR, 'evaluation_results.json')

# Preload reviews dataset for fast browsing and EDA
DATA_CSV = os.path.join(DATA_DIR, 'amazon_food_reviews.csv')
PROCESSED_CSV = os.path.join(DATA_DIR, 'processed_reviews.csv')

if os.path.exists(DATA_CSV):
    REVIEWS_DF = pd.read_csv(DATA_CSV)
elif os.path.exists(PROCESSED_CSV):
    # Load first 5000 rows for responsive EDA / browsing
    REVIEWS_DF = pd.read_csv(PROCESSED_CSV, nrows=5000)
else:
    REVIEWS_DF = pd.DataFrame()

# Load manifest for notebook image assets
MANIFEST_PATH = os.path.join(BASE_DIR, 'static', 'notebook_assets', 'manifest.json')
if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        ASSETS_MANIFEST = json.load(f)
else:
    ASSETS_MANIFEST = []

# Pre-load models on startup once
try:
    get_models()
    print("All 6 models and vectorizers verified and ready in memory.")
except Exception as e:
    print(f"Warning: Initial model load notification: {e}")

# Stem to authentic dataset word frequency map for un-stemming
_STEM_UNSTEM_MAP = None

def get_stem_unstem_map():
    global _STEM_UNSTEM_MAP
    if _STEM_UNSTEM_MAP is None:
        _STEM_UNSTEM_MAP = {}
        if not REVIEWS_DF.empty:
            from nltk.stem import SnowballStemmer
            stemmer = SnowballStemmer('english')
            counts = {}
            summary_col = REVIEWS_DF['Summary'].fillna('') if 'Summary' in REVIEWS_DF.columns else ''
            text_col = REVIEWS_DF['Text'].fillna('') if 'Text' in REVIEWS_DF.columns else ''
            texts = (summary_col + ' ' + text_col).values
            for t in texts:
                for w in re.findall(r'[a-zA-Z]+', str(t).lower()):
                    if len(w) > 2:
                        s = stemmer.stem(w)
                        if s not in counts:
                            counts[s] = {}
                        counts[s][w] = counts[s].get(w, 0) + 1
            for s, w_dict in counts.items():
                _STEM_UNSTEM_MAP[s] = max(w_dict, key=w_dict.get)
    return _STEM_UNSTEM_MAP

def unstem_token(stem):
    mapping = get_stem_unstem_map()
    return mapping.get(stem, stem)

# ================= PAGE ROUTES =================

@app.route('/')
def index():
    """Page 1: Live Sentiment Studio & Real-time Detector"""
    return render_template('index.html', active_page='studio')

@app.route('/batch')
def batch_page():
    """Page 2: Batch & Product Health Analytics"""
    return render_template('batch.html', active_page='batch')

@app.route('/benchmarks')
def benchmarks_page():
    """Page 3: Model Tournament & Visual Flow"""
    return render_template('benchmarks.html', active_page='benchmarks')

# ================= REST API ROUTES =================

@app.route('/api/health')
def api_health():
    """System, memory, and model availability health check endpoint"""
    try:
        models, tfidf, bow, metrics = get_models()
        model_keys = list(models.keys())
        has_all_6 = len(model_keys) >= 6
        return jsonify({
            "status": "healthy" if has_all_6 else "degraded",
            "ensemble_operational": True,
            "models_loaded_count": len(model_keys),
            "available_models": model_keys,
            "tfidf_features": tfidf.get_feature_names_out().shape[0] if tfidf else 0,
            "bow_features": bow.get_feature_names_out().shape[0] if bow else 0,
            "reviews_catalog_loaded": len(REVIEWS_DF)
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "message": "Error loading model components"
        }), 500

@app.route('/api/stats')
def api_stats():
    """Return high-level project and dataset statistics"""
    stats = dict(PROJECT_STATS)
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                disk_stats = json.load(f)
                stats.update(disk_stats)
        except Exception:
            pass
    return jsonify({
        "status": "success",
        "data": stats
    })

@app.route('/api/evaluation')
def api_evaluation():
    """
    Returns freshly measured metrics from models/evaluation_results.json.
    Includes test accuracy, precision, recall, F1, ROC-AUC, confusion matrices,
    training time, latency, and sample sizes on the held-out test split.
    """
    if os.path.exists(EVAL_RESULTS_FILE):
        try:
            with open(EVAL_RESULTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({
                "status": "success",
                "source": "measured_evaluation_run",
                "data": data
            })
        except Exception:
            pass
    # Fallback to in-memory metrics
    try:
        _, _, _, metrics = get_models()
        return jsonify({
            "status": "success",
            "source": "cached_memory",
            "data": {"models": metrics}
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Evaluation results not found"
        }), 404

@app.route('/api/benchmark')
def api_benchmark():
    """Return historical 8-notebook benchmark results for reference"""
    vectorizers = {}
    for item in BENCHMARK_RESULTS:
        vec = item['vectorizer']
        if vec not in vectorizers:
            vectorizers[vec] = []
        vectorizers[vec].append(item['test_auc'])

    vec_summary = {
        v: {
            "avg_auc": round(sum(scores) / len(scores), 2),
            "max_auc": max(scores),
            "count": len(scores)
        }
        for v, scores in vectorizers.items()
    }

    return jsonify({
        "status": "success",
        "total_experiments": len(BENCHMARK_RESULTS),
        "results": BENCHMARK_RESULTS,
        "vectorizer_summary": vec_summary
    })

@app.route('/api/top-words')
def api_top_words():
    """
    Reads trained Logistic Regression coefficients and TF-IDF vocabulary.
    Returns top positive and negative impact words influencing sentiment.
    """
    try:
        models, tfidf, bow, metrics = get_models()
        lr = models.get('logistic_regression')
        if lr is None or not hasattr(lr, 'coef_'):
            return jsonify({
                "status": "error",
                "message": "Logistic regression model is not loaded"
            }), 500

        features = tfidf.get_feature_names_out()
        coefs = lr.coef_[0]

        unigram_indices = [i for i, f in enumerate(features) if ' ' not in f]
        if not unigram_indices:
            unigram_indices = list(range(len(features)))

        top_pos_idx = sorted(unigram_indices, key=lambda i: coefs[i], reverse=True)[:8]
        top_neg_idx = sorted(unigram_indices, key=lambda i: coefs[i])[:8]

        happy_words = []
        for idx in top_pos_idx:
            stem = str(features[idx])
            display_word = unstem_token(stem)
            happy_words.append({
                "word": display_word,
                "stem": stem,
                "weight": round(float(coefs[idx]), 4)
            })

        unhappy_words = []
        for idx in top_neg_idx:
            stem = str(features[idx])
            display_word = unstem_token(stem)
            unhappy_words.append({
                "word": display_word,
                "stem": stem,
                "weight": round(float(coefs[idx]), 4)
            })

        return jsonify({
            "status": "success",
            "happy_words": happy_words,
            "unhappy_words": unhappy_words,
            "data": {
                "happy_words": happy_words,
                "unhappy_words": unhappy_words
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Could not compute top words"
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Live inference prediction endpoint with 6-model comparison and ensemble"""
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    model = data.get('model', 'ensemble_avg').strip()

    if not text:
        return jsonify({
            "status": "error",
            "message": "Please enter review text to analyze."
        }), 400

    # Cap text length for safety
    if len(text) > 15000:
        text = text[:15000]

    try:
        result = predict_review(text, selected_model=model)
        return jsonify({
            "status": "success",
            "data": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Prediction failed. Ensure model artifacts are generated."
        }), 500

@app.route('/api/predict-batch', methods=['POST'])
def api_predict_batch():
    """
    Batch inference endpoint for bulk review evaluation.
    Supports negative threshold alerting and small sample detection.
    """
    data = request.get_json(silent=True) or {}
    reviews_list = data.get('reviews', [])
    model = data.get('model', 'ensemble_avg').strip()
    threshold = float(data.get('threshold', 25.0))  # negative percentage threshold
    product_name = data.get('product_name', 'Custom Batch').strip()

    if not reviews_list:
        return jsonify({
            "status": "error",
            "message": "Please provide a list of reviews to evaluate."
        }), 400

    # Cap max reviews per batch for responsiveness
    if len(reviews_list) > 1000:
        reviews_list = reviews_list[:1000]

    try:
        batch_out = predict_batch_reviews(reviews_list, selected_model=model)
        total = batch_out['total_analyzed']
        neg_pct = batch_out['negative_pct']
        pos_pct = batch_out['positive_pct']

        # Determine health status
        is_small_sample = total < 10
        is_defect_alert = (neg_pct >= threshold) and (total > 0)

        if pos_pct >= 85:
            satisfaction_status = "Exceptional Satisfaction"
            satisfaction_emoji = "⭐"
        elif pos_pct >= 70:
            satisfaction_status = "Mostly Happy & Healthy"
            satisfaction_emoji = "👍"
        elif pos_pct >= 50:
            satisfaction_status = "Mixed Customer Feedback"
            satisfaction_emoji = "⚖️"
        else:
            satisfaction_status = "Critical Negative Feedback"
            satisfaction_emoji = "🚨"

        batch_out.update({
            "product_name": product_name,
            "threshold_used_pct": threshold,
            "defect_alert": is_defect_alert,
            "is_preliminary": is_small_sample,
            "satisfaction_status": satisfaction_status,
            "satisfaction_emoji": satisfaction_emoji,
            "selected_model": model
        })

        return jsonify({
            "status": "success",
            "data": batch_out
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Batch evaluation encountered an error."
        }), 500

@app.route('/api/samples')
def api_samples():
    """Return curated test reviews for instant 1-click testing"""
    return jsonify({
        "status": "success",
        "samples": SAMPLE_REVIEWS
    })

@app.route('/api/eda')
def api_eda():
    """Compute and return Exploratory Data Analysis metrics"""
    if REVIEWS_DF.empty:
        return jsonify({"status": "error", "message": "Dataset not found"}), 404

    score_counts = REVIEWS_DF['Score'].value_counts().to_dict()
    score_dist = [{"score": int(k), "count": int(v)} for k, v in sorted(score_counts.items())]

    sentiment_counts = REVIEWS_DF['Sentiment'].value_counts().to_dict() if 'Sentiment' in REVIEWS_DF.columns else {}

    df_temp = REVIEWS_DF.copy()
    df_temp['char_length'] = df_temp['Text'].fillna('').apply(len)
    df_temp['word_count'] = df_temp['Text'].fillna('').apply(lambda x: len(x.split()))

    length_by_sentiment = {}
    for sent in ['positive', 'negative', 'neutral']:
        sub = df_temp[df_temp['Sentiment'] == sent] if 'Sentiment' in df_temp.columns else pd.DataFrame()
        if not sub.empty:
            length_by_sentiment[sent] = {
                "avg_chars": round(float(sub['char_length'].mean()), 1),
                "avg_words": round(float(sub['word_count'].mean()), 1),
                "median_words": int(sub['word_count'].median()),
                "count": len(sub)
            }

    top_products = []
    if 'ProductName' in REVIEWS_DF.columns:
        prod_vc = REVIEWS_DF['ProductName'].value_counts().head(8).to_dict()
        for prod, count in prod_vc.items():
            sub = REVIEWS_DF[REVIEWS_DF['ProductName'] == prod]
            avg_score = round(float(sub['Score'].mean()), 2)
            top_products.append({
                "product_name": prod,
                "review_count": int(count),
                "avg_rating": avg_score
            })

    return jsonify({
        "status": "success",
        "total_reviews": len(REVIEWS_DF),
        "score_distribution": score_dist,
        "sentiment_distribution": sentiment_counts,
        "length_by_sentiment": length_by_sentiment,
        "top_products": top_products
    })

@app.route('/api/reviews')
def api_reviews():
    """Paginated, filterable reviews from authentic dataset"""
    if REVIEWS_DF.empty:
        return jsonify({"status": "error", "message": "No reviews available"}), 404

    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))
    query = request.args.get('q', '').strip().lower()
    sentiment = request.args.get('sentiment', '').strip().lower()
    score = request.args.get('score', '')

    filtered_df = REVIEWS_DF.copy()

    if sentiment and sentiment != 'all' and 'Sentiment' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Sentiment'].str.lower() == sentiment]

    if score and score != 'all':
        try:
            score_int = int(score)
            filtered_df = filtered_df[filtered_df['Score'] == score_int]
        except ValueError:
            pass

    if query:
        text_match = filtered_df['Text'].fillna('').str.lower().str.contains(query, regex=False)
        summary_match = filtered_df['Summary'].fillna('').str.lower().str.contains(query, regex=False) if 'Summary' in filtered_df.columns else False
        filtered_df = filtered_df[text_match | summary_match]

    total_count = len(filtered_df)
    total_pages = max(1, (total_count + per_page - 1) // per_page)

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    page_records = filtered_df.iloc[start_idx:end_idx].to_dict(orient='records')

    cleaned_records = []
    for r in page_records:
        cleaned_records.append({
            "id": int(r.get('Id', 0)),
            "product_id": str(r.get('ProductId', '')),
            "product_name": str(r.get('ProductName', r.get('ProductId', 'Amazon Product'))),
            "score": int(r.get('Score', 0)),
            "sentiment": str(r.get('Sentiment', 'positive' if r.get('Score', 0) > 3 else 'negative')),
            "summary": str(r.get('Summary', '')),
            "text": str(r.get('Text', ''))
        })

    return jsonify({
        "status": "success",
        "total": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "reviews": cleaned_records
    })

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({"status": "error", "message": "API endpoint not found"}), 404
    return render_template('index.html', active_page='studio'), 404

@app.errorhandler(500)
def server_error(e):
    if request.path.startswith('/api/'):
        return jsonify({"status": "error", "message": "Internal server error"}), 500
    return "Internal Server Error", 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Amazon Food Reviews Analysis Web Server on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
