from .utils import (
    load_vectorizer,
    load_classification_model,
    load_regression_model,
    clean_text
)

vectorizer=load_vectorizer()
class_model=load_classification_model()
score_model=load_regression_model()

def pred_problem_class(text: str):
    text=clean_text(text)
    X=vectorizer.transform([text])
    pred=class_model.predict(X)[0]
    return pred

def pred_problem_score(text: str):
    text=clean_text(text)
    X=vectorizer.transform([text])
    score=score_model.predict(X)[0]
    return float(score)

def full_pred(text: str):
    difficulty=pred_problem_class(text)
    score=pred_problem_score(text)
    return {
        "difficulty": difficulty,
        "predicted_score": round(score, 2)
    }
