"""
Data Ingestion and Preprocessing Pipeline for Amazon Fine Food Reviews.
Handles:
- Reading raw dataset (Reviews.csv or SNAP finefoods.txt.gz)
- Column validation and schema integrity
- Missing value cleaning and text normalization
- Deduplication based on (UserId, ProfileName, Time, Text)
- Helpfulness sanity filtering (HelpfulnessNumerator <= HelpfulnessDenominator)
- 3-Class Sentiment Labeling:
    Score 1-2: Negative
    Score 3: Neutral
    Score 4-5: Positive
- Binary vs Multiclass preparation (stratified split, no data leakage)
- Preserving neutral reviews for exploratory data analytics
- Saving processed data to data/processed_reviews.csv and dataset statistics to data/dataset_stats.json
"""

import os
import re
import html
import gzip
import json
import time
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.stem import SnowballStemmer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_REVIEWS_CSV = os.path.join(DATA_DIR, 'Reviews.csv')
RAW_SNAP_GZ = os.path.join(DATA_DIR, 'finefoods.txt.gz')
PROCESSED_CSV = os.path.join(DATA_DIR, 'processed_reviews.csv')
STATS_JSON = os.path.join(DATA_DIR, 'dataset_stats.json')

# Negation words preserved during stopword filtering
NEGATION_WORDS = {
    'not', 'no', 'nor', 'neither', 'never', 'none', 'hardly', 'barely',
    'scarcely', 'cannot', "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "won't", "wouldn't", "don't",
    "doesn't", "didn't", "can't", "couldn't", "shouldn't"
}

_stemmer = None

def get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = SnowballStemmer('english')
    return _stemmer

def clean_review_text(raw_text, stem=True):
    """
    Cleans review text:
    1. Unescape HTML entities & strip HTML tags
    2. Remove URLs
    3. Remove punctuation and non-alphanumeric symbols
    4. Normalize whitespace and lowercase
    5. Optionally apply Snowball Stemmer
    """
    if pd.isna(raw_text):
        return ""
    text = str(raw_text)
    # Decode HTML entities (e.g. &quot;, &#39;, &amp;)
    text = html.unescape(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase & tokenize
    words = text.lower().split()
    
    if not stem:
        return " ".join(words)
    
    stemmer = get_stemmer()
    stemmed = [stemmer.stem(w) for w in words if len(w) > 1]
    return " ".join(stemmed)

def parse_snap_txt_gz(snap_gz_path, output_csv_path, max_records=None):
    """
    Parses original Stanford SNAP finefoods.txt.gz into standard tabular CSV.
    Each review has fields:
    product/productId, review/userId, review/profileName, review/helpfulness,
    review/score, review/time, review/summary, review/text
    """
    print(f"Parsing SNAP gzip archive: {snap_gz_path} -> {output_csv_path}")
    records = []
    current = {}
    review_id = 1
    
    with gzip.open(snap_gz_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    current['Id'] = review_id
                    records.append(current)
                    review_id += 1
                    current = {}
                    if max_records and len(records) >= max_records:
                        break
                continue
            
            if ': ' in line:
                key, val = line.split(': ', 1)
                key = key.strip()
                val = val.strip()
                if key == 'product/productId':
                    current['ProductId'] = val
                elif key == 'review/userId':
                    current['UserId'] = val
                elif key == 'review/profileName':
                    current['ProfileName'] = val
                elif key == 'review/helpfulness':
                    parts = val.split('/')
                    if len(parts) == 2:
                        try:
                            current['HelpfulnessNumerator'] = int(parts[0])
                            current['HelpfulnessDenominator'] = int(parts[1])
                        except ValueError:
                            current['HelpfulnessNumerator'] = 0
                            current['HelpfulnessDenominator'] = 0
                elif key == 'review/score':
                    try:
                        current['Score'] = int(float(val))
                    except ValueError:
                        current['Score'] = 3
                elif key == 'review/time':
                    try:
                        current['Time'] = int(val)
                    except ValueError:
                        current['Time'] = 0
                elif key == 'review/summary':
                    current['Summary'] = val
                elif key == 'review/text':
                    current['Text'] = val

        if current:
            current['Id'] = review_id
            records.append(current)

    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"Extracted {len(df):,} reviews to {output_csv_path}")
    return df

def load_raw_dataset(csv_path=None):
    """
    Loads raw Amazon Fine Food Reviews dataset.
    Checks Reviews.csv first, then finefoods.txt.gz if present.
    """
    if csv_path is None:
        csv_path = RAW_REVIEWS_CSV
        
    if not os.path.exists(csv_path):
        if os.path.exists(RAW_SNAP_GZ):
            print(f"Raw CSV not found at {csv_path}, but found SNAP archive {RAW_SNAP_GZ}. Extracting...")
            return parse_snap_txt_gz(RAW_SNAP_GZ, csv_path)
        fallback = os.path.join(DATA_DIR, 'amazon_food_reviews.csv')
        if os.path.exists(fallback):
            print(f"Warning: Primary dataset {csv_path} not found. Falling back to {fallback}")
            return pd.read_csv(fallback)
        raise FileNotFoundError(
            f"Dataset file not found at {csv_path}. Please place Reviews.csv inside data/ directory."
        )

    print(f"Reading raw reviews from {csv_path}...")
    df = pd.read_csv(csv_path)
    return df

def run_preprocessing_pipeline(input_csv=None, output_csv=None, sample_limit=None):
    """
    Executes full data cleaning and sanitation pipeline.
    Validates columns, filters duplicates & invalid records, generates sentiment labels,
    computes detailed statistics, and saves output.
    """
    t_start = time.time()
    os.makedirs(DATA_DIR, exist_ok=True)
    if output_csv is None:
        output_csv = PROCESSED_CSV

    df = load_raw_dataset(input_csv)
    original_count = len(df)
    print(f"Original record count: {original_count:,}")

    # 1. Validate required columns
    required_cols = ['Id', 'ProductId', 'UserId', 'ProfileName', 
                     'HelpfulnessNumerator', 'HelpfulnessDenominator', 
                     'Score', 'Time', 'Summary', 'Text']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # 2. Filter empty review text
    initial_null_text = df['Text'].isna().sum()
    df = df.dropna(subset=['Text']).copy()
    # Strip whitespace and drop blank text
    df['Text'] = df['Text'].astype(str).str.strip()
    df = df[df['Text'].str.len() > 0].copy()
    empty_text_dropped = original_count - len(df)
    print(f"Removed {empty_text_dropped:,} records with missing/empty review text.")

    # 3. Helpfulness sanity check: Numerator must be <= Denominator
    valid_help_mask = df['HelpfulnessNumerator'] <= df['HelpfulnessDenominator']
    invalid_helpfulness_count = (~valid_help_mask).sum()
    df = df[valid_help_mask].copy()
    print(f"Removed {invalid_helpfulness_count:,} records with HelpfulnessNumerator > HelpfulnessDenominator.")

    # 4. Deduplication
    # Many users review the same product under multiple flavor sizes (same User, ProfileName, Time, Text)
    # Sort by ProductId first to ensure deterministic deduplication
    df = df.sort_values('ProductId')
    dedup_subset = ['UserId', 'ProfileName', 'Time', 'Text']
    before_dedup = len(df)
    df = df.drop_duplicates(subset=dedup_subset, keep='first').copy()
    duplicates_removed = before_dedup - len(df)
    print(f"Removed {duplicates_removed:,} duplicate reviews (retention: {len(df)/original_count*100:.1f}%).")

    # 5. Sentiment Labeling from Score
    # Score 1-2 -> negative (0)
    # Score 3   -> neutral (preserved for analytics / EDA)
    # Score 4-5 -> positive (1)
    def assign_sentiment(score):
        if score >= 4:
            return 'positive'
        elif score <= 2:
            return 'negative'
        else:
            return 'neutral'

    df['Sentiment'] = df['Score'].apply(assign_sentiment)
    # Target label: 1 for positive, 0 for negative, -1 for neutral
    df['Target'] = df['Sentiment'].map({'positive': 1, 'negative': 0, 'neutral': -1})

    # Summary fillna
    df['Summary'] = df['Summary'].fillna('').astype(str).str.strip()

    cleaned_count = len(df)
    total_removed = original_count - cleaned_count

    # 6. Detailed Sentiment & Rating Distributions
    sentiment_counts = df['Sentiment'].value_counts().to_dict()
    score_counts = df['Score'].value_counts().sort_index().to_dict()

    stats = {
        "original_record_count": original_count,
        "cleaned_record_count": cleaned_count,
        "total_removed_records": total_removed,
        "duplicates_removed": duplicates_removed,
        "invalid_helpfulness_removed": int(invalid_helpfulness_count),
        "empty_text_removed": int(empty_text_dropped),
        "sentiment_distribution": {
            "positive": int(sentiment_counts.get('positive', 0)),
            "negative": int(sentiment_counts.get('negative', 0)),
            "neutral": int(sentiment_counts.get('neutral', 0)),
            "positive_pct": round(sentiment_counts.get('positive', 0) / cleaned_count * 100, 2),
            "negative_pct": round(sentiment_counts.get('negative', 0) / cleaned_count * 100, 2),
            "neutral_pct": round(sentiment_counts.get('neutral', 0) / cleaned_count * 100, 2)
        },
        "score_distribution": {str(k): int(v) for k, v in score_counts.items()},
        "binary_classification_ready_count": int(sentiment_counts.get('positive', 0) + sentiment_counts.get('negative', 0)),
        "pipeline_execution_time_seconds": round(time.time() - t_start, 2)
    }

    # Save stats JSON
    with open(STATS_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"Dataset statistics saved to {STATS_JSON}")

    # Optional subsample for fast development / testing if requested
    if sample_limit and sample_limit < len(df):
        print(f"Subsampling {sample_limit:,} records for fast output...")
        df_save = df.sample(n=sample_limit, random_state=42)
    else:
        df_save = df

    print(f"Saving cleaned dataset to {output_csv}...")
    df_save.to_csv(output_csv, index=False)
    print(f"Processed dataset successfully saved! Clean records: {len(df_save):,}")

    return df, stats

def get_train_test_split(df=None, test_size=0.2, random_state=42, max_samples=None):
    """
    Splits dataset into stratified training and testing sets for binary classification.
    Strictly excludes neutral reviews (Score = 3) from training to prevent ambiguity.
    Uses ONLY review 'Text' as input features (never Score or target label).
    Prevents data leakage.
    """
    if df is None:
        if os.path.exists(PROCESSED_CSV):
            print(f"Loading processed reviews from {PROCESSED_CSV}...")
            df = pd.read_csv(PROCESSED_CSV)
        else:
            print("Processed dataset not found. Running preprocessing pipeline...")
            df, _ = run_preprocessing_pipeline()

    # Filter out neutral reviews for binary sentiment classification
    binary_df = df[df['Sentiment'] != 'neutral'].copy()
    
    if max_samples and max_samples < len(binary_df):
        print(f"Stratified subsampling {max_samples:,} records for model training...")
        binary_df, _ = train_test_split(
            binary_df, train_size=max_samples, 
            stratify=binary_df['Target'], random_state=random_state
        )

    # Use strictly 'Text' (or Summary + Text if desired, but never Score)
    X = binary_df['Text'].values
    y = binary_df['Target'].astype(int).values

    print(f"Splitting {len(X):,} reviews (Stratified, test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train size: {len(X_train):,} (Pos: {(y_train==1).sum():,}, Neg: {(y_train==0).sum():,})")
    print(f"Test size:  {len(X_test):,} (Pos: {(y_test==1).sum():,}, Neg: {(y_test==0).sum():,})")

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess Amazon Fine Food Reviews Dataset")
    parser.add_argument('--input', type=str, default=None, help="Path to input Reviews.csv")
    parser.add_argument('--output', type=str, default=None, help="Path to output processed_reviews.csv")
    parser.add_argument('--sample', type=int, default=None, help="Optional sample limit")
    args = parser.parse_args()

    run_preprocessing_pipeline(args.input, args.output, args.sample)
