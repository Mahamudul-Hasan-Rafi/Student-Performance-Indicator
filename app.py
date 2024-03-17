from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

app=FastAPI()

templates=Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    pass