import streamlit as st
from src.model import full_pred

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
        result=full_pred(text)
        st.subheader("Results")
        st.write(f"**Predicted Difficulty:** {result['difficulty']}")
        st.write(f"**Predicted Score:** {result['predicted_score']}")

