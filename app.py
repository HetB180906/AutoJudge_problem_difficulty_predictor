import streamlit as st
import pickle 

with open("models/probclass_model.pkl","rb") as f:
    clf_model=pickle.load(f)

with open("models/probscore_model.pkl","rb") as f:
    reg_model=pickle.load(f)

with open("models/tfidf_vectorizer.pkl","rb") as f:
    vec=pickle.load(f)

st.title("AutoJudge - Problem difficulty predictor")
st.write("Paste your problem details below and know your problem difficulty.")
prob_des=st.text_area("Problem description")
input_des=st.text_area("Input description")
output_des=st.text_area("Output description")

if st.button("Predict Difficulty"):
    if prob_des.strip()=="" and input_des.strip()=="" and output_des.strip()=="":
        st.warning("Enter text before predicting!!!")
    else:
        text=prob_des+" "+input_des+" "+output_des
        X=vec.transform([text])
        pred_class=clf_model.predict(X)[0]
        pred_score=reg_model.predict(X)[0]
        st.subheader("Results")
        st.write(f"**Predicted Difficulty:** {pred_class}")
        st.write(f"**Predicted Score:** {round(pred_score,2)}")

