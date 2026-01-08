Student Details<br>
Name: Het Amitbhai Bharadia<br>
Branch: B Tech CSE<br>
Enrollment No: 24114040<br><br>

Project<br>
AutoJudge -- A model for predicting score and difficulty for Competitive Programming problems.
<br>Predicts:
<br>Problem Class ---> easy, medium, hard
<br>Problem Score ---> numerical difficulty score

<br><br>Dataset Used
<br>Source:
https://github.com/AREEG94FAHAD/TaskComplexityEval-24
<br>Records: 4112
<br>Important fields:
<br>    title
<br>    description
<br>    input_description
<br>    output_description
<br>    problem_class (label)
<br>    problem_score (target variable of regression)

<br><br>Features Extracted
<br>Combined all the problems text and used a TF-IDF vectorizer
<br>Extracted other features such as average word length, words and phrases related to competitive programming
<br>The features were scaled using a standard scaler and combined

<br><br>Classification
<br>Random Forest Classifier used for classification:
<br>Accuracy: 0.5078
<br>Confusion Matrix and classification report: Mentioned in report

<br><br>Regression
<br>3 separate random forest regresion models used for each class (Easy, Medium and Hard)
<br><br>Easy Model:
<br>MAE: 3.255
<br>RMSE: 3.838
<br>R2: -2.209

<br>Medium Model:
<br>MAE: 2.167
<br>RMSE: 2.577
<br>R2: -0.393

<br>Hard Model:
<br>MAE: 2.245
<br>RMSE: 2.801
<br>R2: -0.606

<br><br>Steps to run locally
<br>Clone the github repository.
<br>Install the base requirements, mentioned in requirements.txt.
<br>Open terminal and start a python virtual environment.
<br>Run the train.py file (The train file contains the entire code of everything form data preprocessing to feature extraction to classification and regression).
<br>After running the file, either run main.py to test locally, or run the command “streamlit run app.py” to run on web.
<br>Enter problem description and get the predicted class and score.

<br><br>Web Interface
<br>Web interface implemented using streamlit. Simple and basic UI.
Text fields for entering problem statement, input description and output description.
<br>Click Predict button to predict the score and class.

<br><br>Demo Video Link
<br> https://drive.google.com/file/d/1DoJ6Z23CiD48EjATP785zxmXoLuC80nc/view?usp=sharing
<br><br>Project Report Link
<br> https://drive.google.com/file/d/1OcJNNA3mJtVD6rz2szFzadzurgmqyWiy/view?usp=sharing