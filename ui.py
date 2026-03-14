import streamlit as st
import pandas as pd
from utils import load_data, load_model_assets, calculate_features

# Page Setup
st.set_page_config(page_title="IPL Match Predictor", page_icon="🏏", layout="centered")
st.title("🏏 IPL Match Predictor")
st.markdown("---")

# 1. Load Data and Assets
df = load_data()
assets = load_model_assets()
model, team_le, city_le, venue_le, toss_le = assets

# 2. User Input Section
col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Team 1", sorted(df['team1'].unique()))
    city = st.selectbox("Select City", sorted(df['city'].dropna().unique()))

with col2:
    teams_for_2 = [t for t in sorted(df['team1'].unique()) if t != team1]
    team2 = st.selectbox("Select Team 2", teams_for_2)
    venue = st.selectbox("Select Venue", sorted(df['venue'].unique()))

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    toss_winner = st.selectbox("Toss Winner", [team1, team2])
with col4:
    toss_decision = st.selectbox("Toss Decision", sorted(df['toss_decision'].unique()))

st.markdown("---")

# 3. Prediction Section
if st.button("Predict Winning Team", use_container_width=True):
    try:
        # Calculate exactly what the model needs
        input_data = calculate_features(df, team1, team2, city, venue, toss_winner, toss_decision, assets)
        
        # Get prediction
        prediction = model.predict(input_data)
        winner_name = team_le.inverse_transform(prediction)[0]

        # Display results
        st.balloons()
        st.success(f"### Predicted Winner: **{winner_name}**")
        
    except Exception as e:
        st.error(f"Something went wrong: {e}")