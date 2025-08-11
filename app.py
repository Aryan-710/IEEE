# streamlit_app.py

import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load model
with open("ieee_recommender.pkl", "rb") as f:
    model_data = pickle.load(f)

tfidf = model_data["tfidf"]
tfidf_matrix = model_data["tfidf_matrix"]
papers = model_data["papers"]

# Streamlit UI
st.title("📚 IEEE Technical Resource Recommender")
st.write("Get recommended IEEE papers, conferences, and resources based on your topic.")

# Input from user
query = st.text_input("Enter your research topic:")

if st.button("Recommend"):
    if query:
        query_vec = tfidf.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-5:][::-1]

        st.subheader("Top Recommendations:")
        for idx in top_indices:
            st.markdown(f"**{papers.iloc[idx]['title']}**")
            st.write(papers.iloc[idx]['abstract'])
            if 'url' in papers.columns:
                st.markdown(f"[Read More]({papers.iloc[idx]['url']})")
            st.write("---")
    else:
        st.warning("Please enter a topic.")
