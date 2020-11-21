import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import ensemble, metrics
from sklearn.model_selection import train_test_split

import config as c
from pipeline.model import EnsembleModels, Model


class RandomForestRegression(Model):
    default_model_constructor_parameters = {
        'n_estimators': 100,
        'criterion': "mse",
        'max_features': "auto",
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'min_impurity_decrease': 0.0,
        'ccp_alpha': 0.0
    }

    def __init__(self):
        super().__init__("random_forest_regression")

    def create_model(self, parameters: dict):
        """
        Созднание модели
        :param parameters: Гиперпараметры модели
        """
        self.model = ensemble.RandomForestRegressor(n_estimators=parameters['n_estimators'], criterion=parameters['criterion'],
                                                     max_features=parameters['max_features'], min_samples_split=parameters['min_samples_split'],
                                                     min_samples_leaf=parameters['min_samples_leaf'], min_impurity_decrease=parameters['min_impurity_decrease'],
                                                     ccp_alpha = parameters['ccp_alpha'] ,random_state=c.SEED)

    def fit_model(self, x, y, test_size=0.2) -> float:
        """
        Обучение модели
        :return: Обученная модель, Скор
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=c.SEED)
        self.model.fit(x_train, y_train)
        y_prediction = self.model.predict(x_test)
        return metrics.mean_squared_error(y_test, y_prediction)

    def score(self, x, y):
        """
        Определение точности модели. Шаблон
        :return: rmse
        """
        
        y_prediction = self.predict(x)
        return metrics.mean_squared_error(y, y_prediction)
