import pickle
import re
import pandas as pd
from scipy.sparse import hstack
from .utils import (
    load_vectorizer,
    load_classification_model,
    load_regression_model,
    clean_text
)
vectorizer = load_vectorizer()
class_model = load_classification_model()
score_model = load_regression_model()
scaler = pickle.load(open("models/feature_scaler.pkl", "rb"))

def extract_features(text):
    df=pd.DataFrame([text], columns=["text"])
    features=pd.DataFrame()
    features["text_length"]=df["text"].str.len()
    features["word_count"]=df["text"].str.split().str.len()
    features["avg_word_length"]=features["text_length"]/features["word_count"]
    features["has_algorithm_keywords"]=df["text"].str.contains(
        r"algorithm|complexity|optimization|dynamic|recursive|greedy|divide|conquer",
        case=False, na=False
    ).astype(int)
    features["has_data_structures"]=df["text"].str.contains(
        r"array|tree|graph|stack|queue|heap|linked|list|hash|map",
        case=False, na=False
    ).astype(int)
    features["has_math_keywords"]=df["text"].str.contains(
        r"matrix|probability|combinatorics|number|prime|fibonacci|factorial",
        case=False, na=False
    ).astype(int)
    features=features.fillna(0)
    return features

def pred_problem_class(text:str):
    text=clean_text(text)
    X_tfidf=vectorizer.transform([text])
    feats=extract_features(text)
    feats_scaled=scaler.transform(feats)
    X=hstack([X_tfidf,feats_scaled])
    pred=class_model.predict(X)[0]
    return pred

def pred_problem_score(text:str):
    text=clean_text(text)
    X_tfidf=vectorizer.transform([text])
    feats=extract_features(text)
    feats_scaled=scaler.transform(feats)
    X=hstack([X_tfidf, feats_scaled])
    score = score_model.predict(X)[0]
    return float(score)

def full_pred(text:str):
    difficulty=pred_problem_class(text)
    score=pred_problem_score(text)
    return {
        "difficulty": difficulty,
        "predicted_score": round(score, 2)
    }
