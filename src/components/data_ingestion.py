import os
import sys
#sys.path.append('src')
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_csv: str=os.path.join('artifacts', 'raw.csv')


class DataIngest:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion initiated")

        try:

            df=pd.read_csv("notebook\data\StudentsPerformance.csv")
            logging.info("Data read as pandas dataframe")

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_csv, index=False, header=True)

            train_df, test_df=train_test_split(df, test_size=0.2, random_state=42)

            train_df.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Done")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    obj=DataIngest()
    train_path, test_path=obj.initiate_data_ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_transformation(train_path,test_path)



