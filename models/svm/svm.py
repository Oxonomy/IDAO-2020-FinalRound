import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

import config as c
from pipeline.model import EnsembleModels, Model


class SVM(Model):
    default_model_constructor_parameters = {
        'kernel': 'linear',
        'gamma': 1e-1,
        'tol': 1e-3,
        'C': 1.0
    }

    def __init__(self):
        super().__init__("svm")

    def create_model(self, parameters: dict):
        """
        Созднание модели
        :param parameters: Гиперпараметры модели
        """
        self.model = svm.SVC(kernel=parameters['kernel'], tol=parameters['tol'], gamma=parameters['gamma'],
                             C=parameters['C'], random_state=c.SEED)

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
