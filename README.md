# 🏏 IPL Match Predictor

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

A professional machine learning application that predicts the outcome of IPL matches based on historical performance, head-to-head statistics, and venue-specific data.

## 🌟 Overview

This project provides a complete end-to-end ML solution for IPL match prediction. It includes a robust data processing pipeline, a trained Random Forest model, a modern web interface built with **Streamlit**, and a scalable backend API powered by **FastAPI**.

## 🚀 Key Features

- **Accurate Predictions**: Uses a Random Forest model trained on historical IPL data (2008-2017+).
- **Dynamic Feature Engineering**: Calculates 12 critical features in real-time, including:
  - Team Season Win Rates
  - Head-to-Head Ratios
  - Venue-specific Win Rates
  - Toss Impact Analysis
- **Dual Interface**:
  - **Streamlit Web UI**: Interactive dashboard for fans and analysts.
  - **FastAPI Endpoint**: Production-ready API for programmatic access.
- **Smart Data Handling**: Intelligent mapping of teams, cities, and venues using pre-trained encoders.

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Machine Learning**: Scikit-learn, NumPy, Pandas
- **Web UI**: Streamlit
- **Backend API**: FastAPI, Uvicorn
- **Data Source**: IPL Matches Dataset (`matches 2.csv`)
- **Serialization**: Pickle (for model and encoders)

## 📦 Installation

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

## 🎮 Usage

### 1. Launch the Streamlit Web UI
```bash
streamlit run ui.py
```
*Access the dashboard at `http://localhost:8501`*

### 2. Run the FastAPI Backend
```bash
uvicorn app.py:app --reload
```
*Access the API documentation at `http://localhost:8000/docs`*

## 🧠 Model Methodology

The prediction model relies on a **Random Forest Classifier** that evaluates several factors:
1. **Team Encodings**: Numerical representation of competing teams.
2. **Venue & City**: Historical performance at specific locations.
3. **Toss Logic**: Impact of winning/losing the toss and the decision (bat/field).
4. **Historical Stats**: Win rates and head-to-head ratios derived from the historical dataset.

## 📂 Project Structure

- `app.py`: FastAPI server implementation.
- `ui.py`: Streamlit frontend code.
- `utils.py`: Core logic for data loading and feature calculation.
- `ipl_model.pkl`: Trained Random Forest model.
- `*_encoder.pkl`: Categorical encoders for teams, cities, and venues.
- `model.ipynb`: Jupyter notebook showcasing EDA and model training.
- `matches 2.csv`: Historical dataset used for training/inference.

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---
*Created with ❤️ for IPL Fans*
