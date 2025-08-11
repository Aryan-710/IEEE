# streamlit_app.py

import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("ieee_recommender.pkl", "rb") as f:
    model_data = pickle.load(f)

tfidf = model_data["tfidf"]
tfidf_matrix = model_data["tfidf_matrix"]
papers = model_data["papers"]

# Domain list
domain_keywords = {
    "Machine Learning": ["machine learning", "deep learning", "ml", "neural network"],
    "Signal Processing": ["signal processing", "image processing", "speech processing"],
    "Renewable Energy": ["renewable", "solar", "wind", "green energy"],
    "Embedded Systems": ["embedded", "microcontroller", "arduino", "raspberry pi"],
    "Internet of Things": ["internet of things", "iot", "sensor network"],
    "Computer Vision": ["computer vision", "image recognition", "object detection"]
}

st.title("IEEE Project Recommender")

# Multiselect for domain choice
selected_domains = st.multiselect("Select Domains", domain_keywords)

if selected_domains:
    all_keywords = []
    for d in selected_domains:
        all_keywords.extend(domain_keywords[d])
    
    mask = (
        papers['abstract'].str.contains('|'.join(all_keywords), case=False, na=False) |
        papers['title'].str.contains('|'.join(all_keywords), case=False, na=False)
    )
    
    filtered = papers[mask]

    if filtered.empty:
        st.warning("No projects found for the selected domains.")
    else:
        st.write("### Top 30 Recommendations")
        st.dataframe(filtered.head(30))

else:
    st.info("Please select at least one domain to see recommendations.")


