import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import saveobject, evaluate_model

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    model_trainer_config_obj_path: sys = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info('Splitting train and test data')
            X_train, X_test, y_train, y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1]
            )

            models = {
                    "Linear Regression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "XGBRegressor": XGBRegressor(), 
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),"AdaBoost Regressor": AdaBoostRegressor()
                    }

            model_report :dict = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models)

            ### best model score
            best_model_score = max(sorted(model_report.values()))

            ### best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            saveobject(
                file_path=self.model_trainer_config.model_trainer_config_obj_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            return r2_score(y_test, predicted), best_model_name
            

        except Exception as e:
            raise CustomException(e, sys)
