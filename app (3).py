
import streamlit as st
import joblib

@st.cache_resource
def load_model():
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    model = joblib.load("logistic_regression.joblib")  # or naive_bayes.joblib / random_forest.joblib
    label_encoder = joblib.load("label_encoder.joblib")  # optional
    return vectorizer, model, label_encoder

tfidf, model, le = load_model()

st.set_page_config(page_title="Review Category Classifier", page_icon="🧠")
st.title("🧠 Review Category Classifier")
st.write("Enter a cleaned review text and classify it into categories like positive, negative, or neutral.")

user_input = st.text_area("✍ Enter Cleaned Text Below:")

if st.button("🔍 Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        vector = tfidf.transform([user_input])
        prediction = model.predict(vector)
        category = le.inverse_transform(prediction)[0] if le else prediction[0]
        st.success(f"📌 *Predicted Category:* {category}")
