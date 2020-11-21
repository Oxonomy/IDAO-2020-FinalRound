import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import ensemble, metrics
from sklearn.model_selection import train_test_split

import config as c
from pipeline.model import EnsembleModels, Model


class RandomForest(Model):
    default_model_constructor_parameters = {
        'n_estimators': 100,
        'criterion': "gini",
        'max_features': "auto",
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }

    def __init__(self):
        super().__init__("random_forest")

    def create_model(self, parameters: dict):
        """
        Созднание модели
        :param parameters: Гиперпараметры модели
        """
        self.model = ensemble.RandomForestClassifier(n_estimators=parameters['n_estimators'], criterion=parameters['criterion'],
                                                     max_features=parameters['max_features'], min_samples_split=parameters['min_samples_split'],
                                                     min_samples_leaf=parameters['min_samples_leaf'], random_state=c.SEED)

    def fit_model(self, x, y, test_size=0.2) -> float:
        """
        Обучение модели
        :return: Обученная модель, Скор
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=c.SEED)
        self.model.fit(x_train, y_train)
        y_prediction = self.model.predict(x_test)
        return metrics.accuracy_score(y_test, y_prediction)

    def score(self, x, y):
        """
        Определение точности модели. Шаблон
        :return: точность
        """
        return self.score_accuracy_classification(x, y)
