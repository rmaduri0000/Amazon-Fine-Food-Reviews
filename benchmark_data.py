"""
Amazon Fine Food Reviews Analysis - Benchmark Data and Experiment Results
Contains verified data from all 8 Jupyter Notebooks in the project.
"""

# Overall project statistics
PROJECT_STATS = {
    "total_original_reviews": 568454,
    "unique_users": 256059,
    "unique_products": 74258,
    "timespan": "Oct 1999 - Oct 2012",
    "cleaned_reviews": 348260,
    "deduplication_retention_pct": 69.8,
    "train_size": 233334,
    "test_size": 114926,
    "best_model": "Logistic Regression (TF-IDF + L2 Regularization, C=50)",
    "best_auc": 96.23,
    "models_evaluated": 8,
    "vectorizers_evaluated": 4,
    "total_experiments": 34
}

# Complete benchmark results from Notebooks 3, 4, 5, 6, 7, 8
BENCHMARK_RESULTS = [
    # Logistic Regression (Notebook 5)
    {
        "id": "lr_tfidf_l2",
        "notebook_key": "nb5_lr",
        "notebook_num": 5,
        "model_family": "Logistic Regression",
        "model_variant": "LR with L2 Regularization",
        "vectorizer": "TF-IDF",
        "best_param": "C = 50",
        "test_auc": 96.23,
        "is_best": True,
        "rank": 1,
        "complexity": "Low / Fast",
        "notes": "Top performer across all 34 experiments. Optimal margin with minimal overfitting."
    },
    {
        "id": "lr_tfidf_l1",
        "notebook_key": "nb5_lr",
        "notebook_num": 5,
        "model_family": "Logistic Regression",
        "model_variant": "LR with L1 Regularization",
        "vectorizer": "TF-IDF",
        "best_param": "C = 5",
        "test_auc": 95.88,
        "is_best": False,
        "rank": 2,
        "complexity": "Low / Sparse",
        "notes": "High feature sparsity while preserving 95.88% AUC. Great for low-memory deployment."
    },
    {
        "id": "lr_bow_l2",
        "notebook_key": "nb5_lr",
        "notebook_num": 5,
        "model_family": "Logistic Regression",
        "model_variant": "LR with L2 Regularization",
        "vectorizer": "BoW",
        "best_param": "C = 5",
        "test_auc": 95.02,
        "is_best": False,
        "rank": 3,
        "complexity": "Low",
        "notes": "Strong performance on simple term frequencies."
    },
    {
        "id": "lr_bow_l1",
        "notebook_key": "nb5_lr",
        "notebook_num": 5,
        "model_family": "Logistic Regression",
        "model_variant": "LR with L1 Regularization",
        "vectorizer": "BoW",
        "best_param": "C = 5",
        "test_auc": 94.84,
        "is_best": False,
        "rank": 4,
        "complexity": "Low",
        "notes": "High sparsity achieved with slight trade-off vs L2."
    },
    # Naive Bayes (Notebook 4)
    {
        "id": "nb_tfidf",
        "notebook_key": "nb4_nb",
        "notebook_num": 4,
        "model_family": "Naive Bayes",
        "model_variant": "Multinomial Naive Bayes",
        "vectorizer": "TF-IDF",
        "best_param": "alpha = 0.05",
        "test_auc": 94.43,
        "is_best": False,
        "rank": 5,
        "complexity": "Extremely Fast",
        "notes": "Near-instant training and inference time with excellent 94.43% AUC."
    },
    # Support Vector Machine (Notebook 6)
    {
        "id": "svm_tfidf_linear",
        "notebook_key": "nb6_svm",
        "notebook_num": 6,
        "model_family": "Support Vector Machine",
        "model_variant": "Linear SVM (SGD / LinearSVC)",
        "vectorizer": "TF-IDF",
        "best_param": "C = 0.001",
        "test_auc": 94.08,
        "is_best": False,
        "rank": 6,
        "complexity": "Moderate",
        "notes": "Maximizes hyperplane margin in high-dimensional TF-IDF space."
    },
    # Naive Bayes (Notebook 4)
    {
        "id": "nb_bow",
        "notebook_key": "nb4_nb",
        "notebook_num": 4,
        "model_family": "Naive Bayes",
        "model_variant": "Multinomial Naive Bayes",
        "vectorizer": "BoW",
        "best_param": "alpha = 0.1",
        "test_auc": 93.56,
        "is_best": False,
        "rank": 7,
        "complexity": "Extremely Fast",
        "notes": "Multinomial probability modeling with Laplace smoothing."
    },
    # Support Vector Machine (Notebook 6)
    {
        "id": "svm_bow_linear",
        "notebook_key": "nb6_svm",
        "notebook_num": 6,
        "model_family": "Support Vector Machine",
        "model_variant": "Linear SVM",
        "vectorizer": "BoW",
        "best_param": "C = 0.001",
        "test_auc": 92.36,
        "is_best": False,
        "rank": 8,
        "complexity": "Moderate",
        "notes": "Effective linear separator on BoW word counts."
    },
    # Ensembles (Notebook 8)
    {
        "id": "gbdt_tfidf",
        "notebook_key": "nb8_rf_gbdt",
        "notebook_num": 8,
        "model_family": "Ensemble (GBDT)",
        "model_variant": "XGBoost / GBDT",
        "vectorizer": "TF-IDF",
        "best_param": "depth = 3, n_est = 400",
        "test_auc": 92.09,
        "is_best": False,
        "rank": 9,
        "complexity": "High",
        "notes": "Sequential gradient boosted trees optimizing pseudo-residuals."
    },
    {
        "id": "gbdt_bow",
        "notebook_key": "nb8_rf_gbdt",
        "notebook_num": 8,
        "model_family": "Ensemble (GBDT)",
        "model_variant": "XGBoost / GBDT",
        "vectorizer": "BoW",
        "best_param": "depth = 3, n_est = 400",
        "test_auc": 92.04,
        "is_best": False,
        "rank": 10,
        "complexity": "High",
        "notes": "Boosted shallow trees (depth 3) preventing overfitting."
    },
    {
        "id": "svm_tfidf_rbf",
        "notebook_key": "nb6_svm",
        "notebook_num": 6,
        "model_family": "Support Vector Machine",
        "model_variant": "RBF Kernel SVM",
        "vectorizer": "TF-IDF",
        "best_param": "C = 0.1, gamma = 0.001",
        "test_auc": 91.51,
        "is_best": False,
        "rank": 11,
        "complexity": "Very High",
        "notes": "Non-linear radial basis function mapping."
    },
    {
        "id": "svm_bow_rbf",
        "notebook_key": "nb6_svm",
        "notebook_num": 6,
        "model_family": "Support Vector Machine",
        "model_variant": "RBF Kernel SVM",
        "vectorizer": "BoW",
        "best_param": "C = 10, gamma = 0.001",
        "test_auc": 91.41,
        "is_best": False,
        "rank": 12,
        "complexity": "Very High",
        "notes": "Dual formulation computation is resource intensive on large datasets."
    },
    {
        "id": "lr_avg_w2v_l2",
        "notebook_key": "nb5_lr",
        "notebook_num": 5,
        "model_family": "Logistic Regression",
        "model_variant": "LR with L2 Regularization",
        "vectorizer": "Avg Word2Vec",
        "best_param": "C = 0.5",
        "test_auc": 91.32,
        "is_best": False,
        "rank": 13,
        "complexity": "Moderate",
        "notes": "Leverages 300-dimensional semantic embeddings."
    },
    {
        "id": "lr_avg_w2v_l1",
        "notebook_key": "nb5_lr",
        "notebook_num": 5,
        "model_family": "Logistic Regression",
        "model_variant": "LR with L1 Regularization",
        "vectorizer": "Avg Word2Vec",
        "best_param": "C = 0.5",
        "test_auc": 91.31,
        "is_best": False,
        "rank": 14,
        "complexity": "Moderate",
        "notes": "L1 penalty on dense semantic embeddings."
    },
    {
        "id": "rf_bow",
        "notebook_key": "nb8_rf_gbdt",
        "notebook_num": 8,
        "model_family": "Ensemble (Random Forest)",
        "model_variant": "Random Forest",
        "vectorizer": "BoW",
        "best_param": "depth = 8, n_est = 400",
        "test_auc": 90.84,
        "is_best": False,
        "rank": 15,
        "complexity": "High",
        "notes": "Bagging ensemble of 400 de-correlated trees."
    },
    {
        "id": "rf_tfidf",
        "notebook_key": "nb8_rf_gbdt",
        "notebook_num": 8,
        "model_family": "Ensemble (Random Forest)",
        "model_variant": "Random Forest",
        "vectorizer": "TF-IDF",
        "best_param": "depth = 5, n_est = 500",
        "test_auc": 90.35,
        "is_best": False,
        "rank": 16,
        "complexity": "High",
        "notes": "500 trees with depth constrained to 5 to avoid overfitting high dimensions."
    },
    {
        "id": "gbdt_avg_w2v",
        "notebook_key": "nb8_rf_gbdt",
        "notebook_num": 8,
        "model_family": "Ensemble (GBDT)",
        "model_variant": "XGBoost / GBDT",
        "vectorizer": "Avg Word2Vec",
        "best_param": "depth = 3, n_est = 300",
        "test_auc": 90.10,
        "is_best": False,
        "rank": 17,
        "complexity": "High",
        "notes": "Tree boosting on continuous word vectors."
    },
    {
        "id": "lr_tfidf_w2v_l2",
        "notebook_key": "nb5_lr",
        "notebook_num": 5,
        "model_family": "Logistic Regression",
        "model_variant": "LR with L2 Regularization",
        "vectorizer": "TFIDF-W2V",
        "best_param": "C = 0.5",
        "test_auc": 88.73,
        "is_best": False,
        "rank": 18,
        "complexity": "Moderate",
        "notes": "TF-IDF weighted average of Word2Vec vectors."
    },
    {
        "id": "lr_tfidf_w2v_l1",
        "notebook_key": "nb5_lr",
        "notebook_num": 5,
        "model_family": "Logistic Regression",
        "model_variant": "LR with L1 Regularization",
        "vectorizer": "TFIDF-W2V",
        "best_param": "C = 0.1",
        "test_auc": 88.72,
        "is_best": False,
        "rank": 19,
        "complexity": "Moderate",
        "notes": "Weighted semantic embedding with sparse regularization."
    },
    {
        "id": "svm_avg_w2v_linear",
        "notebook_key": "nb6_svm",
        "notebook_num": 6,
        "model_family": "Support Vector Machine",
        "model_variant": "Linear SVM",
        "vectorizer": "Avg Word2Vec",
        "best_param": "C = 10",
        "test_auc": 88.46,
        "is_best": False,
        "rank": 20,
        "complexity": "Moderate",
        "notes": "Separating hyperplane in 300D embedding space."
    },
    {
        "id": "gbdt_tfidf_w2v",
        "notebook_key": "nb8_rf_gbdt",
        "notebook_num": 8,
        "model_family": "Ensemble (GBDT)",
        "model_variant": "XGBoost / GBDT",
        "vectorizer": "TFIDF-W2V",
        "best_param": "depth = 3, n_est = 200",
        "test_auc": 87.34,
        "is_best": False,
        "rank": 21,
        "complexity": "High",
        "notes": "Gradient boosting on TF-IDF weighted word embeddings."
    },
    {
        "id": "svm_tfidf_w2v_linear",
        "notebook_key": "nb6_svm",
        "notebook_num": 6,
        "model_family": "Support Vector Machine",
        "model_variant": "Linear SVM",
        "vectorizer": "TFIDF-W2V",
        "best_param": "C = 10",
        "test_auc": 86.24,
        "is_best": False,
        "rank": 22,
        "complexity": "Moderate",
        "notes": "Linear SVM on weighted dense embeddings."
    },
    {
        "id": "rf_avg_w2v",
        "notebook_key": "nb8_rf_gbdt",
        "notebook_num": 8,
        "model_family": "Ensemble (Random Forest)",
        "model_variant": "Random Forest",
        "vectorizer": "Avg Word2Vec",
        "best_param": "depth = 5, n_est = 500",
        "test_auc": 86.12,
        "is_best": False,
        "rank": 23,
        "complexity": "High",
        "notes": "500 decision trees splitting on semantic dimensions."
    },
    {
        "id": "svm_avg_w2v_rbf",
        "notebook_key": "nb6_svm",
        "notebook_num": 6,
        "model_family": "Support Vector Machine",
        "model_variant": "RBF Kernel SVM",
        "vectorizer": "Avg Word2Vec",
        "best_param": "C = 10, gamma = 0.1",
        "test_auc": 85.32,
        "is_best": False,
        "rank": 24,
        "complexity": "Very High",
        "notes": "RBF non-linear projection on word vector representations."
    },
    # KNN (Notebook 3)
    {
        "id": "knn_avg_w2v_kdtree",
        "notebook_key": "nb3_knn",
        "notebook_num": 3,
        "model_family": "K-Nearest Neighbors",
        "model_variant": "KNN (KD-Tree)",
        "vectorizer": "Avg Word2Vec",
        "best_param": "K = 11",
        "test_auc": 84.00,
        "is_best": False,
        "rank": 25,
        "complexity": "High Memory",
        "notes": "KD-tree space partitioning in 300D dense semantic space."
    },
    {
        "id": "knn_avg_w2v_brute",
        "notebook_key": "nb3_knn",
        "notebook_num": 3,
        "model_family": "K-Nearest Neighbors",
        "model_variant": "KNN (Brute Force)",
        "vectorizer": "Avg Word2Vec",
        "best_param": "K = 7",
        "test_auc": 83.91,
        "is_best": False,
        "rank": 26,
        "complexity": "High Compute",
        "notes": "Exhaustive pairwise distance evaluation."
    },
    {
        "id": "rf_tfidf_w2v",
        "notebook_key": "nb8_rf_gbdt",
        "notebook_num": 8,
        "model_family": "Ensemble (Random Forest)",
        "model_variant": "Random Forest",
        "vectorizer": "TFIDF-W2V",
        "best_param": "depth = 5, n_est = 100",
        "test_auc": 83.29,
        "is_best": False,
        "rank": 27,
        "complexity": "Moderate",
        "notes": "100 trees on TF-IDF weighted word embeddings."
    },
    {
        "id": "svm_tfidf_w2v_rbf",
        "notebook_key": "nb6_svm",
        "notebook_num": 6,
        "model_family": "Support Vector Machine",
        "model_variant": "RBF Kernel SVM",
        "vectorizer": "TFIDF-W2V",
        "best_param": "C = 100, gamma = 0.01",
        "test_auc": 82.89,
        "is_best": False,
        "rank": 28,
        "complexity": "Very High",
        "notes": "Non-linear kernel on weighted semantic embeddings."
    },
    {
        "id": "knn_tfidf_w2v_kdtree",
        "notebook_key": "nb3_knn",
        "notebook_num": 3,
        "model_family": "K-Nearest Neighbors",
        "model_variant": "KNN (KD-Tree)",
        "vectorizer": "TFIDF-W2V",
        "best_param": "K = 11",
        "test_auc": 80.95,
        "is_best": False,
        "rank": 29,
        "complexity": "High",
        "notes": "Nearest neighbor lookup in weighted semantic space."
    },
    # Decision Trees (Notebook 7)
    {
        "id": "dt_avg_w2v",
        "notebook_key": "nb7_dt",
        "notebook_num": 7,
        "model_family": "Decision Trees",
        "model_variant": "Decision Tree (CART)",
        "vectorizer": "Avg Word2Vec",
        "best_param": "depth = 5, min_split = 500",
        "test_auc": 80.18,
        "is_best": False,
        "rank": 30,
        "complexity": "Low",
        "notes": "Pruned single tree with max depth 5 to avoid extreme variance."
    },
    {
        "id": "knn_bow_kdtree",
        "notebook_key": "nb3_knn",
        "notebook_num": 3,
        "model_family": "K-Nearest Neighbors",
        "model_variant": "KNN (KD-Tree)",
        "vectorizer": "BoW",
        "best_param": "K = 15",
        "test_auc": 79.60,
        "is_best": False,
        "rank": 31,
        "complexity": "High",
        "notes": "Curse of dimensionality affects KD-Tree on sparse BoW."
    },
    {
        "id": "knn_tfidf_brute",
        "notebook_key": "nb3_knn",
        "notebook_num": 3,
        "model_family": "K-Nearest Neighbors",
        "model_variant": "KNN (Brute Force)",
        "vectorizer": "TF-IDF",
        "best_param": "K = 11",
        "test_auc": 78.33,
        "is_best": False,
        "rank": 32,
        "complexity": "High",
        "notes": "Cosine / Euclidean distances in TF-IDF space."
    },
    {
        "id": "dt_tfidf_w2v",
        "notebook_key": "nb7_dt",
        "notebook_num": 7,
        "model_family": "Decision Trees",
        "model_variant": "Decision Tree",
        "vectorizer": "TFIDF-W2V",
        "best_param": "depth = 5, min_split = 100",
        "test_auc": 78.08,
        "is_best": False,
        "rank": 33,
        "complexity": "Low",
        "notes": "Axis-aligned cuts on weighted word vectors."
    },
    {
        "id": "knn_bow_brute",
        "notebook_key": "nb3_knn",
        "notebook_num": 3,
        "model_family": "K-Nearest Neighbors",
        "model_variant": "KNN (Brute Force)",
        "vectorizer": "BoW",
        "best_param": "K = 9",
        "test_auc": 77.83,
        "is_best": False,
        "rank": 34,
        "complexity": "High",
        "notes": "Brute-force Euclidean distance on sparse token vectors."
    },
    {
        "id": "dt_tfidf",
        "notebook_key": "nb7_dt",
        "notebook_num": 7,
        "model_family": "Decision Trees",
        "model_variant": "Decision Tree",
        "vectorizer": "TF-IDF",
        "best_param": "depth = 10, min_split = 500",
        "test_auc": 76.66,
        "is_best": False,
        "rank": 35,
        "complexity": "Low",
        "notes": "Single decision tree struggles with high sparsity of TF-IDF."
    },
    {
        "id": "dt_bow",
        "notebook_key": "nb7_dt",
        "notebook_num": 7,
        "model_family": "Decision Trees",
        "model_variant": "Decision Tree",
        "vectorizer": "BoW",
        "best_param": "depth = 10, min_split = 100",
        "test_auc": 76.37,
        "is_best": False,
        "rank": 36,
        "complexity": "Low",
        "notes": "High variance, easily overfits without ensemble bagging/boosting."
    },
    {
        "id": "knn_tfidf_kdtree",
        "notebook_key": "nb3_knn",
        "notebook_num": 3,
        "model_family": "K-Nearest Neighbors",
        "model_variant": "KNN (KD-Tree)",
        "vectorizer": "TF-IDF",
        "best_param": "K = 7",
        "test_auc": 76.07,
        "is_best": False,
        "rank": 37,
        "complexity": "High",
        "notes": "Degradation due to sparse high dimensions."
    },
    {
        "id": "knn_tfidf_w2v_brute",
        "notebook_key": "nb3_knn",
        "notebook_num": 3,
        "model_family": "K-Nearest Neighbors",
        "model_variant": "KNN (Brute Force)",
        "vectorizer": "TFIDF-W2V",
        "best_param": "K = 1",
        "test_auc": 51.70,
        "is_best": False,
        "rank": 38,
        "complexity": "High",
        "notes": "Degenerate K=1 case severely overfits; KD-Tree at K=11 was 80.95%."
    }
]

# Detailed metadata for the 8 Notebooks
NOTEBOOK_CHAPTERS = [
    {
        "key": "nb1_eda",
        "number": 1,
        "title": "Exploratory Data Analysis & Text Preprocessing",
        "filename": "1.Amazon Fine Food Reviews Analysis.ipynb",
        "summary": "Data cleaning, deduplication, handling missing values, text sanitation, and vocabulary analysis across 568k reviews.",
        "highlights": [
            "Filtered neutral reviews (Score = 3), mapped Score > 3 as Positive (1) and Score < 3 as Negative (0)",
            "Deduplication removed duplicates where UserId, ProfileName, Time, and Text were identical",
            "Removed invalid entries where HelpfulnessNumerator > HelpfulnessDenominator",
            "Preprocessed text: regex HTML tag removal, punctuation removal, stopword filtering, and Snowball Stemming",
            "WordCloud generation for most common positive vs negative words"
        ],
        "key_metrics": {
            "Original Count": "568,454 reviews",
            "After Deduplication": "364,171 reviews",
            "Positive Reviews": "84.3%",
            "Negative Reviews": "15.7%"
        },
        "extracted_image_keys": ["nb1_eda_img_01", "nb1_eda_img_02"]
    },
    {
        "key": "nb2_tsne",
        "number": 2,
        "title": "t-SNE High-Dimensional Visualization",
        "filename": "2.AmazonFineFoodReviews_TSNE.ipynb",
        "summary": "Non-linear dimensionality reduction using t-Distributed Stochastic Neighbor Embedding to visualize semantic cluster separability.",
        "highlights": [
            "Evaluated 4 text representations: BoW, TF-IDF, Average Word2Vec, and TF-IDF weighted Word2Vec",
            "Applied Standard Scaler for dense embeddings (Word2Vec) with with_mean=False on sparse matrices",
            "Tuned Perplexity = 30 and Iterations = 5000",
            "Revealed that Word2Vec representations yield noticeably tighter geometric clusters than sparse BoW/TF-IDF",
            "Positive and negative reviews have significant overlap in lexical space, proving the need for non-linear or regularized margins"
        ],
        "key_metrics": {
            "Perplexity": "30",
            "Iterations": "5,000",
            "Vectors Analyzed": "BoW, TF-IDF, Avg-W2V, TFIDF-W2V",
            "Sample Size": "2,000 to 5,000 points"
        },
        "extracted_image_keys": ["nb2_tsne_img_01", "nb2_tsne_img_02", "nb2_tsne_img_03", "nb2_tsne_img_04", "nb2_tsne_img_05"]
    },
    {
        "key": "nb3_knn",
        "number": 3,
        "title": "K-Nearest Neighbors (KNN)",
        "filename": "3.Amazon Fine Food Reviews Analysis knn.ipynb",
        "summary": "Instance-based learning comparing Brute Force distance search versus KD-Tree spatial trees across 4 vectorizers.",
        "highlights": [
            "Tuned K hyperparameter in range [1, 23] with odd increments to prevent ties",
            "KD-Tree reduced inference query time significantly on dense Word2Vec embeddings",
            "Avg Word2Vec achieved the highest KNN Test AUC (84.00% with K=11)",
            "Sparse vectorizers suffered from curse of dimensionality in distance calculations",
            "Calculated Confusion Matrices, F1-score, Precision-Recall reports, and Hamming Loss"
        ],
        "key_metrics": {
            "Best Variant": "Avg Word2Vec + KD-Tree",
            "Best K": "11",
            "Test AUC": "84.00%",
            "Train Sample": "100,000 points"
        },
        "extracted_image_keys": [
            "nb3_knn_img_01", "nb3_knn_img_02", "nb3_knn_img_03", "nb3_knn_img_04",
            "nb3_knn_img_07", "nb3_knn_img_08", "nb3_knn_img_13", "nb3_knn_img_14"
        ]
    },
    {
        "key": "nb4_nb",
        "number": 4,
        "title": "Naive Bayes Classification",
        "filename": "4. Amazon Fine Food Reviews Analysis_NB.ipynb",
        "summary": "Probabilistic generative modeling with Multinomial Naive Bayes and Laplace Smoothing.",
        "highlights": [
            "Time-based splitting: sorted chronologically to mimic real-world unseen future reviews",
            "Tuned Laplace smoothing parameter alpha across [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]",
            "TF-IDF MultinomialNB achieved 94.43% Test AUC with alpha = 0.05",
            "BoW MultinomialNB achieved 93.56% Test AUC with alpha = 0.1",
            "Extracted top 20 most informative words for positive reviews ('great', 'love', 'best', 'delicious') and negative reviews ('disappoint', 'bad', 'worst', 'waste')"
        ],
        "key_metrics": {
            "Best Vectorizer": "TF-IDF",
            "Optimal Alpha": "0.05",
            "Test AUC": "94.43%",
            "Inference Speed": "< 2 ms"
        },
        "extracted_image_keys": [
            "nb4_nb_img_01", "nb4_nb_img_02", "nb4_nb_img_03", "nb4_nb_img_04",
            "nb4_nb_img_05", "nb4_nb_img_06", "nb4_nb_img_07", "nb4_nb_img_08"
        ]
    },
    {
        "key": "nb5_lr",
        "number": 5,
        "title": "Logistic Regression (Top Benchmark Model)",
        "filename": "5 . Amazon Fine Food Reviews Analysis_LR.ipynb",
        "summary": "Linear classification with L1 (Lasso) and L2 (Ridge) regularization, perturbation test for collinearity, and sparsity analysis.",
        "highlights": [
            "TF-IDF with L2 regularization achieved the HIGHEST AUC of the entire benchmark: 96.23% (C=50)",
            "Conducted Perturbation Test by adding random epsilon noise to features to prove no multicollinearity in coefficients",
            "L1 regularization drove over 75% of feature weights to exact zero, creating a hyper-compressed sparse model with 95.88% AUC",
            "Identified top influential positive and negative n-grams via learned weight vectors w",
            "Comprehensive ROC curves and normalized confusion matrices for all 4 featurizations"
        ],
        "key_metrics": {
            "Benchmark Rank": "#1 (Champion)",
            "Best Config": "TF-IDF + L2 Regularization (C=50)",
            "Test AUC": "96.23%",
            "L1 Sparsity": "76.4% zero weights"
        },
        "extracted_image_keys": [
            "nb5_lr_img_01", "nb5_lr_img_02", "nb5_lr_img_03", "nb5_lr_img_04",
            "nb5_lr_img_07", "nb5_lr_img_08", "nb5_lr_img_15", "nb5_lr_img_16"
        ]
    },
    {
        "key": "nb6_svm",
        "number": 6,
        "title": "Support Vector Machines (SVM)",
        "filename": "6 . Amazon Fine Food Reviews Analysis_SVM.ipynb",
        "summary": "Maximum margin classification comparing Linear SVM (SGDClassifier / LinearSVC) and RBF Kernel SVM with grid search over C and gamma.",
        "highlights": [
            "Linear SVM with TF-IDF achieved 94.08% Test AUC with C = 0.001",
            "Linear SVM with BoW reached 92.36% Test AUC with C = 0.001",
            "RBF Kernel SVM achieved 91.51% AUC (C=0.1, gamma=0.001) but required substantial kernel computation",
            "Proved that for high-dimensional sparse text data (d > 50,000), Linear SVM performs faster and better than RBF kernels",
            "Support vectors localized precisely on borderline reviews"
        ],
        "key_metrics": {
            "Linear SVM TF-IDF": "94.08% AUC",
            "Linear SVM BoW": "92.36% AUC",
            "RBF SVM TF-IDF": "91.51% AUC",
            "Optimal Linear C": "0.001"
        },
        "extracted_image_keys": [
            "nb6_svm_img_01", "nb6_svm_img_02", "nb6_svm_img_03", "nb6_svm_img_04",
            "nb6_svm_img_07", "nb6_svm_img_08", "nb6_svm_img_15", "nb6_svm_img_16"
        ]
    },
    {
        "key": "nb7_dt",
        "number": 7,
        "title": "Decision Trees",
        "filename": "7 . Amazon Fine Food Reviews Analysis_DT.ipynb",
        "summary": "Tree-based recursive binary splitting with Grid Search CV over maximum tree depth and minimum sample split.",
        "highlights": [
            "Hyperparameter grid: max_depth in [1, 5, 10, 50, 100, 500, 100], min_samples_split in [5, 10, 100, 500]",
            "Avg Word2Vec reached 80.18% Test AUC (depth=5, min_split=500)",
            "BoW and TF-IDF single trees reached ~76.4% AUC due to axis-aligned splits on 50,000+ sparse features",
            "Pruning proved critical: unconstrained decision trees severely overfitted training data (99% train AUC vs 68% test AUC)",
            "Extracted top tree decision nodes: initial splits centered on 'great', 'love', 'disappoint', 'return'"
        ],
        "key_metrics": {
            "Best Vectorizer": "Avg Word2Vec",
            "Optimal Depth": "5",
            "Optimal Min Split": "500",
            "Test AUC": "80.18%"
        },
        "extracted_image_keys": [
            "nb7_dt_img_01", "nb7_dt_img_02", "nb7_dt_img_03", "nb7_dt_img_04",
            "nb7_dt_img_05", "nb7_dt_img_06", "nb7_dt_img_07", "nb7_dt_img_08"
        ]
    },
    {
        "key": "nb8_rf_gbdt",
        "number": 8,
        "title": "Ensembles: Random Forest & XGBoost / GBDT",
        "filename": "8 . Amazon Fine Food Reviews Analysis_RF&GBDT.ipynb",
        "summary": "State-of-the-art ensemble methods comparing Bagging (Random Forest) against Boosting (GBDT / XGBoost).",
        "highlights": [
            "Tuned number of base estimators n_estimators in [100, 200, 300, 400, 500] and max_depth in [3, 5, 8, 10]",
            "XGBoost / GBDT achieved 92.09% Test AUC on TF-IDF with depth=3 and 400 estimators",
            "Random Forest achieved 90.84% Test AUC on BoW with depth=8 and 400 estimators",
            "Boosting outperformed Bagging across all 4 vectorizers by adaptively focusing on misclassified borderline reviews",
            "Evaluated feature importance curves showing heavy concentration on sentiment polarity tokens"
        ],
        "key_metrics": {
            "GBDT TF-IDF": "92.09% AUC",
            "Random Forest BoW": "90.84% AUC",
            "GBDT Avg-W2V": "90.10% AUC",
            "Optimal GBDT Depth": "3 (shallow trees)"
        },
        "extracted_image_keys": [
            "nb8_rf_gbdt_img_01", "nb8_rf_gbdt_img_02", "nb8_rf_gbdt_img_03", "nb8_rf_gbdt_img_04",
            "nb8_rf_gbdt_img_05", "nb8_rf_gbdt_img_06", "nb8_rf_gbdt_img_07", "nb8_rf_gbdt_img_08"
        ]
    }
]

# Curated test reviews for instant 1-click live playground demonstration
SAMPLE_REVIEWS = [
    {
        "label": "Gourmet Coffee Delight (Positive)",
        "summary": "Best hazelnut roast ever tasted!",
        "text": "This coffee has an incredible aroma and rich nutty taste. Not bitter at all, perfectly smooth and flavorful. I have ordered 4 bags already and will definitely buy again!",
        "expected": "Positive"
    },
    {
        "label": "Stale & Expired Snack (Negative)",
        "summary": "Completely stale and rancid smell",
        "text": "Extremely disappointed with this purchase. The bag arrived punctured and the chips were completely stale and tasted like cardboard. Threw it straight into the garbage. Do not waste your money!",
        "expected": "Negative"
    },
    {
        "label": "Organic Green Tea (Positive)",
        "summary": "Refreshing and authentic matcha flavor",
        "text": "Five stars! High quality organic Japanese green tea leaves. Beautiful vibrant color and crisp clean taste. Great packaging keeps it fresh. Highly recommended for daily morning brew.",
        "expected": "Positive"
    },
    {
        "label": "Artificial Aftertaste (Negative)",
        "summary": "Terrible artificial chemical taste",
        "text": "The description promised pure fruit flavor, but it tastes entirely like artificial sweetener and chemicals. Left a horrific lingering aftertaste in my mouth. Will never order from this brand again.",
        "expected": "Negative"
    },
    {
        "label": "Dog Food Quality Issue (Negative)",
        "summary": "Made my pet sick, awful formula",
        "text": "Beware! My dog refused to eat this after the first bite, and when he did, he vomited within an hour. Something has clearly changed with the ingredients. Very bad batch, returning immediately.",
        "expected": "Negative"
    },
    {
        "label": "Artisanal Dark Chocolate (Positive)",
        "summary": "Heavenly rich dark cocoa",
        "text": "Absolute perfection. Rich 85% cocoa with silky smooth texture and subtle notes of vanilla. Melt in your mouth goodness without excessive sugar. Worth every single penny!",
        "expected": "Positive"
    }
]
