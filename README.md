AutoJudge -- A model for predicting score and difficulty for Competitive Programming problems.

Predicts: 
Problem Class ---> easy, medium, hard
Problem Score ---> numerical difficulty score

Dataset:
Source: 
Records: 4112
Important fields:
    - title
    - description
    - input_description
    - output_description
    - problem_class (label)
    - problem_score (target variable of regression)

Features Used:
Combined all the problems text and used a TF-IDF vectorizer

Model Used: 
Logistic Regression: for problem_class
Linear Regression: for problem_score

Status
Baseline model implemented

Future Work
add more features (keywords, mathematical symbols)
Improve accuracy
Add web interface
