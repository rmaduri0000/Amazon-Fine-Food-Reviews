"""
Automated Integration and Unit Test Suite for Amazon Fine Food Reviews System.
Validates:
- Data pipeline integrity and dataset statistics
- 6 Machine Learning algorithms and vectorizer loading
- Mathematical accuracy of Combined Ensemble Average
- Single review inference & Explainability
- Batch inference & Defect alert logic
- All Flask web pages & REST API endpoints
- Error handling on invalid/empty inputs
"""

import os
import json
import pytest
import numpy as np

from app import app
from data_pipeline import PROCESSED_CSV, STATS_JSON, RAW_REVIEWS_CSV
from model_trainer import get_models, predict_review, predict_batch_reviews, EVALUATION_JSON

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_data_pipeline_artifacts():
    """Verify raw dataset, processed dataset, and statistics files exist and are valid"""
    assert os.path.exists(RAW_REVIEWS_CSV), "Reviews.csv must exist in data/"
    assert os.path.getsize(RAW_REVIEWS_CSV) > 100_000_000, "Raw Reviews.csv should be ~300MB"

    assert os.path.exists(PROCESSED_CSV), "processed_reviews.csv must exist in data/"
    assert os.path.exists(STATS_JSON), "dataset_stats.json must exist in data/"

    with open(STATS_JSON, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    assert stats['original_record_count'] == 568454, f"Expected 568,454 original reviews, got {stats['original_record_count']}"
    assert stats['cleaned_record_count'] == 393931, f"Expected 393,931 cleaned reviews, got {stats['cleaned_record_count']}"
    assert stats['duplicates_removed'] == 174521, f"Expected 174,521 duplicates removed, got {stats['duplicates_removed']}"
    assert 'sentiment_distribution' in stats
    assert stats['sentiment_distribution']['positive'] > 0
    assert stats['sentiment_distribution']['negative'] > 0
    assert stats['sentiment_distribution']['neutral'] > 0

def test_model_loading_and_evaluation_results():
    """Verify all 6 core models and vectorizers load properly from disk"""
    models, tfidf, bow, metrics = get_models()
    required_keys = ['logistic_regression', 'naive_bayes', 'svm', 'xgboost', 'random_forest', 'knn']
    
    for key in required_keys:
        assert key in models, f"Model {key} missing from loaded models dictionary"
        assert models[key] is not None, f"Model {key} is None"

    assert tfidf is not None, "TF-IDF vectorizer must be loaded"
    assert bow is not None, "CountVectorizer (BoW) must be loaded"
    assert len(tfidf.get_feature_names_out()) > 1000, "TF-IDF vocabulary should be substantial"

    # Verify evaluation_results.json
    assert os.path.exists(EVALUATION_JSON), "evaluation_results.json must exist in models/"
    with open(EVALUATION_JSON, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    assert 'models' in eval_data
    for key in required_keys + ['ensemble_avg']:
        assert key in eval_data['models'], f"Metric for {key} missing in evaluation_results.json"
        m = eval_data['models'][key]
        assert 'accuracy' in m and m['accuracy'] > 70.0, f"Accuracy for {key} out of range"
        assert 'roc_auc' in m and m['roc_auc'] > 70.0, f"ROC-AUC for {key} out of range"
        assert 'f1_score' in m and m['f1_score'] > 70.0, f"F1-Score for {key} out of range"
        assert 'confusion_matrix' in m and len(m['confusion_matrix']) == 2

def test_inference_and_ensemble_calculation():
    """Verify prediction outputs and mathematical correctness of ensemble average"""
    pos_sample = "Five stars! Loved this organic roast coffee, very delicious, smooth aroma and fresh flavor."
    res_pos = predict_review(pos_sample, selected_model='ensemble_avg')

    assert res_pos['prediction'] == 'Positive', f"Expected Positive for glowing coffee review, got {res_pos['prediction']}"
    assert res_pos['confidence'] >= 50.0
    assert 'models_consensus' in res_pos
    assert len(res_pos['models_consensus']) == 7  # 6 models + ensemble_avg
    assert 'model_agreement_pct' in res_pos
    assert res_pos['model_agreement_pct'] >= 50.0
    assert len(res_pos['word_attributions']) > 0

    # Verify ensemble average formula: P_ens = 1/6 sum(P_i)
    consensus = res_pos['models_consensus']
    prob_keys = ['logistic_regression', 'naive_bayes', 'svm', 'xgboost', 'random_forest', 'knn']
    individual_pos_probs = [consensus[k]['pos_prob'] for k in prob_keys]
    expected_mean = round(float(np.mean(individual_pos_probs)), 1)
    ensemble_pos_prob = consensus['ensemble_avg']['pos_prob']

    assert abs(expected_mean - ensemble_pos_prob) <= 0.5, (
        f"Ensemble probability {ensemble_pos_prob}% did not match mean {expected_mean}%"
    )

    # Test negative review
    neg_sample = "Completely stale, disgusting and expired. Bag arrived torn open, smelled like chemicals. Threw it away!"
    res_neg = predict_review(neg_sample, selected_model='ensemble_avg')
    assert res_neg['prediction'] == 'Negative', f"Expected Negative for spoiled bag, got {res_neg['prediction']}"
    assert res_neg['confidence'] >= 50.0

def test_batch_prediction_and_threshold_alert():
    """Verify batch evaluation and configurable defect alert logic"""
    reviews = [
      "Delicious gourmet cocoa, incredible smooth taste!",
      "Very fresh beans, lovely aroma, best coffee ever.",
      "Completely stale, awful chemical flavor, expired and opened bag.",
      "Dog vomited after eating this, terrible rancid smell.",
      "Great tea, fresh and soothing."
    ]

    res = predict_batch_reviews(reviews, selected_model='ensemble_avg')
    assert res['total_analyzed'] == 5
    assert res['positive_count'] == 3
    assert res['negative_count'] == 2
    assert res['positive_pct'] == 60.0
    assert res['negative_pct'] == 40.0

def test_flask_pages_and_api(client):
    """Test all web routes and API endpoints for valid status and JSON response"""
    # 1. Page routes
    r_index = client.get('/')
    assert r_index.status_code == 200
    assert b'Live Studio' in r_index.data

    r_batch = client.get('/batch')
    assert r_batch.status_code == 200
    assert b'Batch and Product Health Analytics' in r_batch.data

    r_bench = client.get('/benchmarks')
    assert r_bench.status_code == 200
    assert b'Which model is best' in r_bench.data

    # 2. API Health
    r_health = client.get('/api/health')
    assert r_health.status_code == 200
    d_health = r_health.get_json()
    assert d_health['status'] == 'healthy'
    assert d_health['models_loaded_count'] == 6

    # 3. API Stats
    r_stats = client.get('/api/stats')
    assert r_stats.status_code == 200
    d_stats = r_stats.get_json()
    assert d_stats['status'] == 'success'
    assert d_stats['data']['original_record_count'] == 568454

    # 4. API Evaluation
    r_eval = client.get('/api/evaluation')
    assert r_eval.status_code == 200
    d_eval = r_eval.get_json()
    assert d_eval['status'] == 'success'
    assert 'ensemble_avg' in d_eval['data']['models']

    # 5. API Top Words
    r_words = client.get('/api/top-words')
    assert r_words.status_code == 200
    d_words = r_words.get_json()
    assert d_words['status'] == 'success'
    assert len(d_words['happy_words']) > 0
    assert len(d_words['unhappy_words']) > 0

    # 6. API Predict valid
    r_pred = client.post('/api/predict', json={
        "text": "Great matcha tea, fresh and delicate flavor!",
        "model": "ensemble_avg"
    })
    assert r_pred.status_code == 200
    d_pred = r_pred.get_json()
    assert d_pred['data']['prediction'] == 'Positive'

    # 7. API Predict empty input validation
    r_empty = client.post('/api/predict', json={"text": "   "})
    assert r_empty.status_code == 400
    assert r_empty.get_json()['status'] == 'error'

    # 8. API Predict Batch valid & defect alert
    r_batch_post = client.post('/api/predict-batch', json={
        "reviews": [
            "Tasted rancid and spoiled. Disgusting food.",
            "Horrible product, worst purchase ever.",
            "Loved this snack, very tasty."
        ],
        "model": "ensemble_avg",
        "threshold": 30.0,
        "product_name": "Test Snack"
    })
    assert r_batch_post.status_code == 200
    d_b = r_batch_post.get_json()['data']
    assert d_b['total_analyzed'] == 3
    assert d_b['defect_alert'] is True  # 2 out of 3 = 66.7% > 30% threshold
    assert d_b['is_preliminary'] is True # 3 < 10 reviews

    # 9. API Predict Batch empty validation
    r_b_empty = client.post('/api/predict-batch', json={"reviews": []})
    assert r_b_empty.status_code == 400

    # 10. API 404 handler
    r_404 = client.get('/api/does-not-exist')
    assert r_404.status_code == 404
    assert r_404.get_json()['status'] == 'error'
