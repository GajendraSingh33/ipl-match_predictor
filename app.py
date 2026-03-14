from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

model = pickle.load(open("ipl_model.pkl","rb"))

@app.get("/")
def home():
    return {"message":"IPL Predictor"}

@app.post("/predict")
def predict(team1:int, team2:int, venue:int):

    data = np.array([[team1,team2,venue]])

    prediction = model.predict(data)

    return {"winner": int(prediction[0])}