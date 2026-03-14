import pandas as pd
import pickle
import streamlit as st

# 1. Load the dataset
@st.cache_data
def load_data():
    """Load and clean the IPL matches dataset."""
    df = pd.read_csv("matches 2.csv")
    df['winner'] = df['winner'].fillna('nan') # Handle missing winners
    return df

# 2. Load the trained model and encoders
@st.cache_resource
def load_model_assets():
    """Load the machine learning model and categorical encoders."""
    model = pickle.load(open("ipl_model.pkl", "rb"))
    team_le = pickle.load(open("team_encoder.pkl", "rb"))
    city_le = pickle.load(open("city_encoder.pkl", "rb"))
    venue_le = pickle.load(open("venue_encoder.pkl", "rb"))
    toss_le = pickle.load(open("toss_decision_encoder.pkl", "rb"))
    return model, team_le, city_le, venue_le, toss_le

# 3. Helper functions for feature engineering
def calculate_features(df, team1, team2, city, venue, toss_winner, toss_decision, assets):
    """Calculate all 12 features required by the model."""
    model, team_le, city_le, venue_le, toss_le = assets

    # Encode categorical inputs
    t1_idx = team_le.transform([team1])[0]
    t2_idx = team_le.transform([team2])[0]
    city_idx = city_le.transform([str(city)])[0]
    venue_idx = venue_le.transform([str(venue)])[0]
    toss_w_idx = team_le.transform([toss_winner])[0]
    toss_d_idx = toss_le.transform([toss_decision])[0]

    # Calculate statistics from the data
    # Win rates
    team_wins = df['winner'].value_counts()
    team_matches = df['team1'].value_counts() + df['team2'].value_counts()
    win_rate_dict = (team_wins / team_matches).to_dict()
    
    t1_win_rate = win_rate_dict.get(team1, 0.0)
    t2_win_rate = win_rate_dict.get(team2, 0.0)

    # Head-to-head
    h2h_matches = df[((df['team1'] == team1) & (df['team2'] == team2)) |
                    ((df['team1'] == team2) & (df['team2'] == team1))]
    t1_h2h_wins = sum(h2h_matches['winner'] == team1)
    total_h2h = len(h2h_matches)
    h2h_ratio = t1_h2h_wins / total_h2h if total_h2h > 0 else 0.5

    # Venue performance
    venue_matches = df[df['venue'] == venue]
    t1_v_wins = sum(venue_matches['winner'] == team1)
    t2_v_wins = sum(venue_matches['winner'] == team2)
    t1_v_rate = t1_v_wins / len(venue_matches) if len(venue_matches) > 0 else 0.0
    t2_v_rate = t2_v_wins / len(venue_matches) if len(venue_matches) > 0 else 0.0

    # Prepare input for model
    return pd.DataFrame([{
        'team1': t1_idx, 'team2': t2_idx, 'city': city_idx, 'venue': venue_idx,
        'toss_winner': toss_w_idx, 'toss_decision': toss_d_idx,
        'team1_win_rate': t1_win_rate, 'team2_win_rate': t2_win_rate,
        'head_to_head_ratio': h2h_ratio,
        'team1_venue_win_rate': t1_v_rate, 'team2_venue_win_rate': t2_v_rate,
        'toss_match_win': 0 # Default placeholder
    }])
