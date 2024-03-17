import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_train(self, train_arr, test_arr):
        try:
            logging.info("Splitting train and test data")

            X_train, X_test, y_train, y_test=(train_arr[:,:-1],test_arr[:,:-1],train_arr[:,-1],test_arr[:,-1])

            models={
                "LinearRegression": LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "Decision Tree": DecisionTreeRegressor(),
                "K Nearest Neighbors": KNeighborsRegressor(),
                "Support Vector": SVR(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Cat Boost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor(),
                "AdaBoost": AdaBoostRegressor(),
            }

            params={
                "LinearRegression":{},
                "Lasso":{
                    "alpha":[0.2,0.3,0.5,0.7,0.8,1.0],
                    "tol":[1e-3,1e-4],
                    "selection":['random','cyclic']
                },
                "Ridge":{
                    "alpha":[0.2,0.3,0.5,0.7,0.8,1.0],
                    "tol":[1e-3,1e-4]
                },
                "Decision Tree":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "K Nearest Neighbors":{
                    "n_neighbors":[3,5,7,9]
                },
                "Support Vector":{
                    "tol":[1e-3,1e-4]
                },
                "Random Forest":{
                    "n_estimators":[8,16,32,64,100,128],
                    "criterion":["squared_error","absolute_error","friedman_mse","poisson"]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Cat Boost":{
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [50, 100, 500, 1000]
                },
                "XGBoost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost":{
                    "n_estimators":[8,16,32,50,64,128,256],
                    "learning_rate":[0.5,0.1,0.01,0.001,0.005]
                }
            }
            
            model_report: dict=evaluate_model(models, params, X_train, y_train, X_test, y_test)

            best_model_score = max(list(model_report.values()), key=lambda x:x['Test Score'])

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score['Test Score']<0.6:
                raise CustomException("No Best model found")
            
            logging.info("Best model found on train and test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test,predicted)
            
            print(best_model_score)

            return {'Model': best_model_name,'R2 Score': r2_square, 'Model Report':model_report[best_model_name]}
        
        except Exception as e:
            raise CustomException(e,sys)

