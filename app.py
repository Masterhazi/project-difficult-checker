# app.py
import streamlit as st
from engine import ProjectDifficultyEngine

st.title("Project Difficulty Checker (Hybrid)")

title = st.text_input("Project Title")
abstract = st.text_area("Project Abstract (or paste your justification)")

st.write("Select project characteristics (optional — helps structural model):")
feature_names = [
 "Deep Learning","NLP","Medical/Health","Finance/Fraud","Vision",
 "Real-Time","Unsupervised","Time-Series","Low Data","Noisy Data"
]
feature_dict = {}
cols = st.columns(2)
for i, fname in enumerate(feature_names):
    feature_dict[fname] = int(cols[i % 2].checkbox(fname))

if st.button("Check Difficulty"):
    engine = ProjectDifficultyEngine()
    res = engine.predict(title, abstract, feature_dict)
    st.metric("Difficulty", res["Final_Label"])
    st.write("Details:")
    st.json(res)
