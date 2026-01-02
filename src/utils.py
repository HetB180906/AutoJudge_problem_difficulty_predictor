import os
import re
import pickle

BASE_DIR=os.path.dirname(os.path.dirname(__file__))  
MODELS_DIR=os.path.join(BASE_DIR, "models")

def load_pickle(file_name):
    path=os.path.join(MODELS_DIR, file_name)
    with open(path,"rb") as f:
        return pickle.load(f)

def load_vectorizer():
    return load_pickle("tfidf_vectorizer.pkl")

def load_classification_model():
    return load_pickle("probclass_model.pkl")

def clean_text(text:str)->str:
    if not isinstance(text,str):
        text=str(text)
    text=text.lower()
    text=re.sub(r"\s+", " ", text)
    return text.strip()
