<div align="center">
  <img src="https://img.icons8.com/color/120/000000/cricket.png" alt="Cricket IPL Logo" />
  <h1>🏏 IPL Match Predictor</h1>
  <p><i>A professional machine learning application that predicts the outcome of IPL matches based on historical performance, head-to-head statistics, and venue-specific data.</i></p>

  [![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
  [![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
  [![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
</div>

<hr/>

## 🌟 Overview

Welcome to the **IPL Match Predictor**! This project provides a complete end-to-end Machine Learning solution for anticipating the outcomes of Indian Premier League (IPL) matches. It seamlessly combines a robust data processing pipeline, a trained Random Forest classifier, a scalable backend API powered by **FastAPI**, and a highly interactive modern web dashboard built with **Streamlit**.

Whether you're an avid IPL fan, a data science enthusiast, or a fantasy cricket strategist, this tool brings data-driven insights right to your fingertips.

---

## 🎯 Model Performance & Accuracy

Our prediction model employs a powerful **Random Forest Classifier** trained extensively on historical IPL data (2008-2017+). 

> **Current Model Accuracy: ~73%**  
> *Note: Cricket is a highly unpredictable game. Achieving >70% accuracy using historical and venue statistics demonstrates a strong baseline capability of the model in capturing team form and venue bias.*

---

## ⚙️ Feature Engineering

To achieve high accuracy and provide meaningful predictions, we have meticulously built out **12 critical dynamic features** during the data engineering phase. These features are calculated in real-time before making a prediction:

1. **Team Win Rates (`team1_win_rate`, `team2_win_rate`)**: Historical overall win probability for both competing teams.
2. **Head-to-Head Ratio (`head_to_head_ratio`)**: The historical dominance of Team 1 over Team 2 in direct clashes.
3. **Venue-specific Win Rates (`team1_venue_win_rate`, `team2_venue_win_rate`)**: How well each team performs at the currently selected venue.
4. **Toss Impact Predictors (`toss_match_win`)**: An analytical indicator of whether winning the toss provides a distinct advantage based on field/bat decisions at the specific venue.
5. **Smart Data Encodings**: Intelligent categorical mapping of teams, cities, and venues using pre-trained `LabelEncoders`.

---

## 🚀 Key Features

- **Accurate Predictions**: Reliable ML model generating predictions based on rich historical data.
- **Dual Interface**:
  - **📱 Streamlit Web UI**: An interactive, user-friendly dashboard for fans and analysts.
  - **⚡ FastAPI Endpoint**: A blazing-fast, RESTful, production-ready API for programmatic access and integrations.
- **Real-Time Calculation**: The pipeline auto-calculates deep statistics like head-to-head records instantly based on the chosen matchup.

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Machine Learning**: `scikit-learn`, `NumPy`, `Pandas`
- **Web UI**: `Streamlit`
- **Backend API**: `FastAPI`, `Uvicorn`
- **Data Source**: IPL Matches Dataset (`matches 2.csv`)
- **Serialization**: `pickle` (for optimized model and encoder loading)

---

## 📦 Installation & Setup

Get the predictor up and running on your local machine in just a few steps!

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ipl-match_predictor
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install streamlit fastapi uvicorn pandas numpy scikit-learn
   ```

---

## 🎮 Usage

You can interact with the IPL Predictor through the visual dashboard or the API.

### 1. Launch the Streamlit Web UI
Experience the interactive dashboard:
```bash
streamlit run ui.py
```
*Access the dashboard at: `http://localhost:8501`*

### 2. Run the FastAPI Backend
Start the high-performance prediction server:
```bash
uvicorn app.py:app --reload
```
*Access the API documentation & interactive Swagger UI at: `http://localhost:8000/docs`*

---

## 🧠 Model Methodology & Pipeline

The core prediction engine relies on the following logic pipeline:
1. **Data Cleaning & Loading**: Handles missing values and normalizes team/venue names.
2. **Categorical Encoding**: Transforms string-based locations and team names into mathematical vectors.
3. **Feature Generation**: Derives the 12 key features (momentum, venue dominance, h2h ratio).
4. **Random Forest Classification**: Passes the 12 features into the complex decision tree ensemble to output the most probable winner.

---

## 📂 Project Structure

```text
├── app.py                      # FastAPI server implementation
├── ui.py                       # Streamlit frontend code
├── utils.py                    # Core logic for data loading and feature calculation
├── ipl_model.pkl               # Trained Random Forest model
├── *_encoder.pkl               # Categorical encoders for teams, cities, and venues
├── model.ipynb                 # Jupyter notebook showcasing EDA and model training
├── matches 2.csv               # Historical dataset used for training/inference
└── requirements.txt            # Python package dependencies
```

---

## 🤝 Open Source Contributors

We welcome contributions from the community! Whether it's adding new features, improving the dataset, or tweaking the UI, your input is highly valued. 

- If you find this repository helpful, please consider leaving a ⭐!

---

## 📝 License

This project is open-source and available under the terms of the [MIT License](LICENSE).

<div align="center">
  <i>Created with ❤️ for IPL Fans & Data Geeks</i>
</div>
