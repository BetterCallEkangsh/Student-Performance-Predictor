import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import saveobject, evaluate_model,print_evaluated_results

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

            params = {
        "Linear Regression": {
        'fit_intercept': [True, False],
        'positive': [True, False],
    },
    "Lasso": {
        'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'fit_intercept': [True, False],
        'max_iter': [1000, 5000],
    },
    "Ridge": {
        'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
        'fit_intercept': [True, False],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr'],
    },
    "K-Neighbors Regressor": {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    },
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'max_depth': [None, 3, 5, 8, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    "Random Forest Regressor": {
        'n_estimators': [16, 32, 64, 128, 256],
        'max_depth': [None, 5, 10, 15],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5],
    },
    "XGBRegressor": {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [32, 64, 128, 256],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.85, 1.0],
    },
    "CatBoosting Regressor": {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [50, 100, 200],
        'l2_leaf_reg': [1, 3, 5],
    },
    "AdaBoost Regressor": {
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
        'n_estimators': [16, 32, 64, 128, 256],
        'loss': ['linear', 'square', 'exponential'],
    },
}


            

            model_report :dict = evaluate_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, models=models, params=params)

            ### best model score
            best_model_score = max(sorted(model_report.values()))

            ### best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            saveobject(
                file_path=self.model_trainer_config.model_trainer_config_obj_path,
                obj=best_model
            )

            print_evaluated_results(
                X_train,y_train,X_test,y_test, best_model
            )

            predicted = best_model.predict(X_test)
            return r2_score(y_test, predicted), best_model_name
            

        except Exception as e:
            raise CustomException(e, sys)
