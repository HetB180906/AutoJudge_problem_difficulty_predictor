import os
import json
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack,csr_matrix

def text_preprocessing(text):
    text=text.lower()
    text=re.sub(r"[^a-zA-Z\s]"," ",text)
    text=re.sub(r"\s+"," ",text).strip()
    words=[w for w in text.split() if len(w) > 2]
    return " ".join(words)

def extract_features(df_input):
    features_df=pd.DataFrame(index=df_input.index)
    features_df["text_length"]=df_input["text"].str.len()
    features_df["word_count"]=df_input["text"].str.split().str.len()
    features_df["avg_word_length"]=(
        features_df["text_length"]/features_df["word_count"]
    )
    features_df["has_algorithm_keywords"]=df_input["text"].str.contains(
        "algorithm|complexity|optimization|dynamic|recursive|greedy|divide|conquer",case=False,na=False,
    ).astype(int)
    features_df["has_data_structures"]=df_input["text"].str.contains(
        "array|tree|graph|stack|queue|heap|linked|list|hash|map",case=False,na=False,
    ).astype(int)
    features_df["has_math_keywords"]=df_input["text"].str.contains(
        "matrix|probability|combinatorics|number|prime|fibonacci|factorial",case=False,na=False,
    ).astype(int)
    return features_df.fillna(0)

def evaluate(name,model,X_test,y_test):
    pred=model.predict(X_test)
    mae=mean_absolute_error(y_test,pred)
    rmse=np.sqrt(mean_squared_error(y_test,pred))
    r2=r2_score(y_test,pred)
    print(f"\n{name.upper()} Regression Performance:")
    print(f"Samples: {len(y_test)}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

def main():
    data_path="data/problems_data.jsonl"
    model_dir="models"
    os.makedirs(model_dir,exist_ok=True)
    data=[]
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    df=pd.DataFrame(data)
    print(f"Total samples: {len(df)}")
    df=df.fillna("")
    df["text"]=(
        df["title"].astype(str)+" "+df["description"].astype(str)+" "+df["input_description"].astype(str)+" "+df["output_description"].astype(str)+" "+df["sample_io"].astype(str)
    )
    df["text"]=df["text"].apply(text_preprocessing)
    add_features=extract_features(df)
    X=df["text"]
    y_class=df["problem_class"]
    y_score=df["problem_score"]
    X_train,X_test,y_class_train,y_class_test=train_test_split(
        X,y_class,test_size=0.2,random_state=42,stratify=y_class,
    )
    X_train_reg,X_test_reg,y_score_train,y_score_test=train_test_split(
        X,y_score,test_size=0.2,random_state=42,
    )
    add_train=add_features.loc[X_train.index]
    add_test=add_features.loc[X_test.index]
    vectorizer=TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.95,
        ngram_range=(1,2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf=vectorizer.fit_transform(X_train)
    X_test_tfidf=vectorizer.transform(X_test)
    scaler=StandardScaler()
    add_train_scaled=scaler.fit_transform(add_train)
    add_test_scaled=scaler.transform(add_test)
    X_train_combined=hstack([X_train_tfidf,add_train_scaled])
    X_test_combined=hstack([X_test_tfidf,add_test_scaled])
    clf=RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train_combined,y_class_train)
    y_pred=clf.predict(X_test_combined)
    print("\nClassification Accuracy:",accuracy_score(y_class_test,y_pred))
    print("\nClassification Report:\n",classification_report(y_class_test,y_pred))
    print("\nConfusion Matrix:\n",confusion_matrix(y_class_test,y_pred))

    X_train_combined=X_train_combined.tocsr()
    mask_easy=(y_class_train == "easy").values
    mask_medium=(y_class_train == "medium").values
    mask_hard=(y_class_train == "hard").values

    X_easy=X_train_combined[mask_easy]
    X_medium=X_train_combined[mask_medium]
    X_hard=X_train_combined[mask_hard]

    y_easy=y_score.loc[X_train.index][mask_easy]
    y_medium=y_score.loc[X_train.index][mask_medium]
    y_hard=y_score.loc[X_train.index][mask_hard]
    print("Performing regression on each class(will take some time....)...............")
    easy_reg=RandomForestRegressor(n_estimators=250,random_state=42)
    medium_reg=RandomForestRegressor(n_estimators=250,random_state=42)
    hard_reg=RandomForestRegressor(n_estimators=250,random_state=42)

    easy_reg.fit(X_easy,y_easy)
    medium_reg.fit(X_medium,y_medium)
    hard_reg.fit(X_hard,y_hard)

    X_test_combined=X_test_combined.tocsr()
    y_test_class=y_class_test.to_numpy()

    easy=y_test_class == "easy"
    medium=y_test_class == "medium"
    hard=y_test_class == "hard"

    evaluate("easy",easy_reg,X_test_combined[easy],y_score_test[easy])
    evaluate("medium",medium_reg,X_test_combined[medium],y_score_test[medium])
    evaluate("hard",hard_reg,X_test_combined[hard],y_score_test[hard])

# save all the models in models
    pickle.dump(vectorizer,open(os.path.join(model_dir,"tfidf_vectorizer.pkl"),"wb"))
    pickle.dump(clf,open(os.path.join(model_dir,"probclass_model.pkl"),"wb"))
    pickle.dump(scaler,open(os.path.join(model_dir,"feature_scaler.pkl"),"wb"))
    pickle.dump(easy_reg,open(os.path.join(model_dir,"easy_reg.pkl"),"wb"))
    pickle.dump(medium_reg,open(os.path.join(model_dir,"medium_reg.pkl"),"wb"))
    pickle.dump(hard_reg,open(os.path.join(model_dir,"hard_reg.pkl"),"wb"))
    print("\nTraining complete. Models saved.")

if __name__ == "__main__":
    main()
