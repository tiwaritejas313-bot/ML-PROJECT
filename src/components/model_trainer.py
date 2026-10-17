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
         
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
    "Decision Tree": {
        "criterion": [
            "squared_error",
            "friedman_mse",
            "absolute_error",
            "poisson"
        ],
        # "splitter": ["best", "random"],
        # "max_features": ["sqrt", "log2"],
    },

    "Random Forest": {
        # "criterion": [
        #     "squared_error",
        #     "friedman_mse",
        #     "absolute_error",
        #     "poisson"
        # ],
        # "max_features": ["sqrt", "log2", None],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },

    "Gradient Boosting": {
        # "loss": [
        #     "squared_error",
        #     "huber",
        #     "absolute_error",
        #     "quantile"
        # ],
        "learning_rate": [0.1, 0.01, 0.05, 0.001],
        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        # "criterion": ["squared_error", "friedman_mse"],
        # "max_features": ["auto", "sqrt", "log2"],
        "n_estimators": [8, 16, 32, 64, 128, 256],
    },

    "Linear Regression": {},

     "K-Neighbor Regressor": {
            'n_neighbors': [5, 7, 9, 11],
            # 'weights': ['uniform', 'distance'],
            # 'algorithm': ['ball_tree', 'kd_tree', 'brute']
        },

        "XGBRegressor": {
            'learning_rate': [.1, .01, .05, .001],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        },

        "CatBoosting Regressor": {
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100]
        },

        "AdaBoost Regressor": {
            'learning_rate': [.1, .01, 0.5, .001],
            # 'loss': ['linear', 'square', 'exponential'],
            'n_estimators': [8, 16, 32, 64, 128, 256]
        }
            }
            modelreport:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
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


            

    

