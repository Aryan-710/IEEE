# streamlit_app.py

import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("ieee_recommender.pkl", "rb") as f:
    model_data = pickle.load(f)

# You might need to adjust depending on how you saved it
# Example: if you saved a dict with 'model' and 'data'
model = model_data["model"]
data = model_data["data"]

# Domain list
domains = [
    "Machine Learning",
    "Signal Processing",
    "Renewable Energy",
    "Embedded Systems",
    "Internet of Things",
    "Computer Vision"
]

# Streamlit UI
st.title("IEEE Project Recommender")

# Multiselect for domain choice
selected_domains = st.multiselect("Select Domains", domains)

if selected_domains:
    st.write(f"Searching for projects in: {', '.join(selected_domains)}")
    
    # Simple filter — adjust if your data format is different
    filtered = data[data['domain'].isin(selected_domains)]
    
    # Show top 30 suggestions
    top_suggestions = filtered.head(30)  # Or sorted logic if needed
    st.write("### Top 30 Recommendations")
    st.dataframe(top_suggestions)

else:
    st.info("Please select at least one domain to see recommendations.")
