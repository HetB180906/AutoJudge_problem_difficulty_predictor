Name: Het Amitbhai Bharadia

Branch: B Tech CSE

Enrollment No: 24114040

**Project**
AutoJudge -- A model for predicting score and difficulty for Competitive Programming problems.

Predicts: 

Problem Class ---> easy, medium, hard

Problem Score ---> numerical difficulty score

**Dataset Used**

Source: https://github.com/AREEG94FAHAD/TaskComplexityEval-24

Records: 4112

Important fields:

    - title

    - description

    - input_description

    - output_description

    - problem_class (label)

    - problem_score (target variable of regression)

**Features Extracted**

Combined all the problems text and used a TF-IDF vectorizer

Extracted other features such as average word length, words and phrases related to competitive programming.

The features were scaled using a standard scaler and combined.

**Classification** 
Random Forest Classifier used for classification:

Accuracy: 0.5078

Confusion Matrix and classification report: Mentioned in report

**Regression**
3 separate random forest regresion models used for each class (Easy, Medium and Hard)

Easy Model:

MAE: 3.255

RMSE: 3.838

R2: -2.209


Medium Model:

MAE: 2.167

RMSE: 2.577

R2: -0.393


Hard Model:

MAE: 2.245

RMSE: 2.801

R2: -0.606


**Steps to run locally**
1.
Clone the github repository. 

2.
Install the base requirements, mentioned in requirements.txt.

3.
Open terminal and start a python virtual environment.

4.
Run the train.py file (The train file contains the entire code of everything form data preprocessing to feature extraction to classification and regression).

5.
After running the file, either run main.py to test locally, or run the command “streamlit run app.py” to run on web.

6.
Enter problem description and get the predicted class and score.


**Web Interface**

Web interface implemented using streamlit. Simple and basic UI. Text fields for entering problem statement, input description and output description.

Click Predict button to predict the score and class.

**Demo Video Link** 


**Project Report Link**
