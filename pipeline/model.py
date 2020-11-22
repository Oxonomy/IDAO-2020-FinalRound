import os
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import metrics
from sklearn.model_selection import RepeatedKFold

import config as c


# noinspection PyMethodMayBeStatic
class EnsembleModels:
    """ Шаблон класса ансамбля моделей """

    def __init__(self, model_name):
        self.model_name = model_name
        self.models = []

    def preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных. Шаблон функции
        :param data: Исходный DataFrame
        :return: Обработанный DataFrame
        """
        return data

    def __create_model(self, parameters: dict) -> object:
        """
        Созднание модели. Шаблон функции
        :param parameters: Гиперпараметры модели
        :return: Модель
        """
        return None

    def __fit_model(self, model, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array) -> (object, float):
        """
        Обучение модели. Шаблон функции
        :return: Обученная модель, Скор
        """
        return model, 0

    def fit_ensemble(self, n_splits, n_repeats, x, y, model_constructor_parameters) -> float:
        """
        Создает и обучает ансамбль моделей, валидация на основе повторяющегося k-fold
        :param n_splits: Количество фолдов при разбиении k-fold
        :param n_repeats: Сколько раз необходимо повторить кросс-валидатцию
        :param x, y: Данные для обучения
        :param model_constructor_parameters: Гиперпараметры моделей
        :return: Скор ансамбля
        """
        self.models = []
        scores = []

        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=c.SEED)
        for train, test in rkf.split(x):
            model = self.__create_model(model_constructor_parameters)
            model, score = self.__fit_model(model, x[train], y[train], x[test], y[test])

            self.models.append(model)
            scores.append(score)

        return float(np.mean(scores))

    def __save_model(self, model, path: str):
        """
        Сохраняет модель. Шаблон функции
        :param model: Моедль
        :param path: Путь сохранения
        """
        pass

    def save_ensemble(self, extra_name="main"):
        """
        Сохраняет ансамбль
        :param extra_name: Название подпапки для хранения модели
        """
        folder_path = os.path.join("models", self.model_name, "saved_models")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        folder_path = os.path.join(folder_path, extra_name)
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        for i, model in enumerate(self.models):
            path = os.path.join(folder_path, "model_" + str(i))
            self.__save_model(model, path)

    def __load_model(self, path: str) -> object:
        """
        Загружает модель. Шаблон функции
        :param path:
        :return: Модель
        """
        pass

    def load_ensemble(self, extra_name="main"):
        """
        Загружает ансамбль
        :param extra_name: Название подпапки для хранения модели
        """
        folder_path = os.path.join("models", self.model_name, "saved_models", extra_name)
        if os.path.exists(folder_path):
            self.models = []
            for path in os.listdir(folder_path):
                self.models.append(self.__load_model(path))

    def __predict(self, model, x) -> np.array:
        return model.predict(x)

    def predict(self, x) -> np.array:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(x))
        prediction = np.concatenate(predictions, axis=1)
        prediction = np.mean(prediction, axis=1).reshape((-1, 1))
        return prediction


# noinspection PyMethodMayBeStatic
class Model:
    """ Шаблон класса модели """

    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Предобработка данных. Шаблон функции
        :param data: Исходный DataFrame
        :return: Обработанный DataFrame
        """
        return data

    def create_model(self, parameters: dict):
        """
        Созднание модели. Шаблон функции
        :param parameters: Гиперпараметры модели
        """
        pass

    def fit_model(self, x, y, test_size=0.2) -> float:
        """
        Обучение модели. Шаблон функции
        :return: Скор
        """
        return 0

    def __save_model(self, model, path: str):
        """
        Сохраняет модель
        :param path: Путь сохранения
        """
        dump(model, path + '.joblib')

    def save_model(self, extra_name="main"):
        """
        Сохраняет ансамбль
        :param extra_name: Название модели
        """
        folder_path = os.path.join("models", self.model_name, "saved_models")
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

        path = os.path.join(folder_path, extra_name)
        self.__save_model(self.model, path)

    def __load_model(self, path: str) -> object:
        """
        Загружает модель
        :param path:
        :return: Модель
        """
        return load(path + '.joblib')

    def load_model(self, extra_name="main"):
        """
        Загружает модель
        :param extra_name: Название модели
        """
        folder_path = os.path.join("models", self.model_name, "saved_models")
        self.model = self.__load_model(os.path.join(folder_path, extra_name))

    def predict(self, x) -> np.array:
        return self.model.predict(x)#.reshape((-1))

    def score_accuracy_classification(self, x, y):
        y_prediction = self.predict(x)
        return metrics.accuracy_score(y, y_prediction)

    def score(self, x, y):
        """
        Определение точности модели. Шаблон
        :return: точность
        """
        return 0
