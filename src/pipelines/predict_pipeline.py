import sys
import pandas as pd
from src.utils import load_object

class Prediction:
    def __init__(self):
        pass

    def predict_score(self, features):
        model_path="artifacts\\model.pkl"
        preprocessor_path="artifacts\\preprocessor.pkl"

        model=load_object(model_path)
        preprocessor=load_object(preprocessor_path)

        preprocessed_features=preprocessor.transform(features)
        
        return model.predict(preprocessed_features)


class CustomData:
    def __init__(self,  gender, ethnicity, parental_level_of_education, 
                  lunch, test_preparation_course, reading_score, writing_score):
        self.gender=gender
        self.ethnicity=ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_frame(self):
        return pd.DataFrame({
            'gender':[self.gender],
            'race/ethnicity':[self.ethnicity],
            'parental level of education':[self.parental_level_of_education],
            'lunch':[self.lunch],
            'test preparation course':[self.test_preparation_course],
            'reading score':[self.reading_score],
            'writing score':[self.writing_score]
        })