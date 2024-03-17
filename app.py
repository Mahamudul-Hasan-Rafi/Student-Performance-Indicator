from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.pipelines.predict_pipeline import CustomData, Prediction

app=FastAPI()

templates=Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})


@app.post("/prediction")
def predict_score(request: Request, gender: str=Form(...), ethnicity: str=Form(...), parental_level_of_education: str=Form(...), 
                  lunch: str=Form(...), test_preparation_course: str=Form(...), reading_score: int=Form(...), writing_score: int=Form(...)):
    # print(gender)
    # print(ethnicity)
    # print(parental_level_of_education)
    # print(lunch)
    # print(test_preparation_course)
    # print(reading_score)
    # print(writing_score)

    custom_data=CustomData(gender, ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score)
    features=custom_data.get_data_frame()

    print(features)
    
    prediction=Prediction()
    result=prediction.predict_score(features)

    return templates.TemplateResponse("index.html",{"request":request, "results":result})