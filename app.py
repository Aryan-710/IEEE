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
domains = [
    "Machine Learning",
    "Signal Processing",
    "Renewable Energy",
    "Embedded Systems",
    "Internet of Things",
    "Computer Vision"
]

st.title("IEEE Project Recommender")

# Multiselect for domain choice
selected_domains = st.multiselect("Select Domains", domains)

if selected_domains:
    st.write(f"Searching for projects in: {', '.join(selected_domains)}")

    # Filter papers based on keyword search in abstract or title
    mask = papers['abstract'].str.contains('|'.join(selected_domains), case=False, na=False) | \
           papers['title'].str.contains('|'.join(selected_domains), case=False, na=False)

    filtered = papers[mask]

    if filtered.empty:
        st.warning("No projects found for the selected domains.")
    else:
        st.write("### Top 30 Recommendations")
        st.dataframe(filtered.head(30))

else:
    st.info("Please select at least one domain to see recommendations.")

