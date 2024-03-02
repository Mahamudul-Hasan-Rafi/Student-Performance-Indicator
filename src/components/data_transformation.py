import numpy as np
import pandas as pd
import os
import sys

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns=['reading score', 'writing score']
            categorical_columns=['gender', 'race/ethnicity', 'parental level of education', 'lunch','test preparation course']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("standard_scaler", StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("standard_scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                    ("num_transform",num_pipeline,numerical_columns),
                    ("cat_transform",cat_pipeline,categorical_columns)
                ]
            )

            logging.info("Numerical columns: {}".format(numerical_columns))
            logging.info("Categorical columns: {}".format(categorical_columns))

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        


    def initiate_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Train and Test data read completed")

            target_column=['math score']
            
            input_features_train_df=train_df.drop(columns=target_column, axis=1)
            target_train_df=train_df['math score']

            input_features_test_df=test_df.drop(columns=target_column, axis=1)
            target_test_df=test_df['math score']

            preprocess_obj=self.get_data_transformer_object()

            input_features_train_arr=preprocess_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocess_obj.transform(input_features_test_df)

            logging.info("Applied preprocessing object on training dataframe and test dataframe")

            train_arr=np.c_[
                input_features_train_arr, np.array(target_train_df)
            ]

            test_arr=np.c_[
                input_features_test_arr, np.array(target_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocess_obj
            )

            logging.info("Object has been saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)