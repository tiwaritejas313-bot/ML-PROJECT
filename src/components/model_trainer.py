import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
import os
import sys
warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models
@dataclass 
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifact","model.pkl")

ModelTrainerconfig = ModelTrainerConfig  # Alias for backward compatibility

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self,train_array,test_array,preprocessor_path=None):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            logging.info("training and testing data split successfully")
         
            models={
                "Linear Regression":LinearRegression(),
                "Decision Tree":DecisionTreeRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "XGBoost":XGBRegressor(),
                "CatBoost":CatBoostRegressor(verbose=False),
                "RandomForest":RandomForestRegressor(),
                "SVM":SVR(),
            }

            modelreport:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            #best model score from dict
            best_model_score=max(sorted(modelreport.values()))
            #best model name from dict
            best_model_name=list(modelreport.keys())[list(modelreport.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with score >= 0.6", sys)
            
            logging.info(f"Best model name: {best_model_name}")
            logging.info(f"Best model score: {best_model_score}")
            logging.info(f"Model report: {modelreport}")
            
            #save best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model saved successfully")
            
            predicted=best_model.predict(X_test)
            r2=r2_score(y_test,predicted)
            logging.info(f"R2 score: {r2}")
            
            return r2

        except Exception as e:
            raise CustomException(e,sys)

    initiate_model_trainer = initiate_model_training


            

    

