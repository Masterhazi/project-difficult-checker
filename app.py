# app.py
import streamlit as st
from engine import ProjectDifficultyEngine

st.set_page_config(
    page_title="Next Era Unitech's Checker",
    page_icon="logo small.png",   # You can use emoji OR image path
    layout="centered"
)

st.image("logo.png", width=350)
st.title("Project Difficulty Checker ")
st.caption("Built with ❤️ using Sentence Transformers & Hybrid ML Architecture")

st.markdown("""
<style>
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


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
    with st.spinner("Analyzing project complexity..."):
        res = engine.predict(title, abstract, feature_dict)

        difficulty = res["Final_Label"]
        confidence = res["Confidence"]

        if difficulty == "Easy":
            st.markdown(f"""
            <div style="
                padding:20px;
                border-radius:12px;
                background-color:#e8f5e9;
                text-align:center;">
                <h2 style="color:#2e7d32;">🟢 {difficulty}</h2>
                <p style="font-size:16px;">Confidence: {confidence*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)

        elif difficulty == "Average":
            st.markdown(f"""
            <div style="
                padding:20px;
                border-radius:12px;
                background-color:#fff8e1;
                text-align:center;">
                <h2 style="color:#f9a825;">🟡 {difficulty}</h2>
                <p style="font-size:16px;">Confidence: {confidence*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)

        elif difficulty == "Difficult":
            st.markdown(f"""
            <div style="
                padding:20px;
                border-radius:12px;
                background-color:#ffebee;
                text-align:center;">
                <h2 style="color:#c62828;">🔴 {difficulty}</h2>
                <p style="font-size:16px;">Confidence: {confidence*100:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔎 Breakdown")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Structural Score", round(res["Weighted_Score"], 2))
            st.metric("MP Classification", res["MP_num"])

        with col2:
            st.metric("Semantic Cluster", res["Cluster"])
            st.metric("Cluster Similarity", round(res["Cluster_similarity"], 3))
