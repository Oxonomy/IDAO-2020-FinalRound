import os
import keras as k
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split, RepeatedKFold

import config as c
from pipeline.model import EnsembleModels


class KerasDenseClassifier(EnsembleModels):
    default_model_constructor_parameters = {
        'activation': 'sigmoid',
        'output_node_activation': 'sigmoid',
        'learning_rate': 1e-3,
        'num_dense_layers': 3,
        'dense_shape': 30,
        'early_patience': 5,
    }

    def __init__(self):
        super().__init__('keras_dense_classifier')
        self.callbacks = []

    def __create_model(self, parameters: dict) -> object:
        """
        Созднание модели
        :param parameters: Гиперпараметры модели
        :return: Модель
        """
        model = k.Sequential()

        for _ in range(parameters['num_dense_layers']):
            model.add(k.layers.Dense(parameters['dense_shape'], activation=parameters['activation']))
        model.add(k.layers.Dense(1, activation=parameters['output_node_activation']))

        opt = k.optimizers.Adam(learning_rate=parameters['learning_rate'])
        model.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.callbacks = []
        es = k.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=parameters['early_patience'],
            verbose=0
        )
        self.callbacks.append(es)

        return model

    def __fit_model(self, model, x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array) -> (object, float):
        """
        Обучение модели
        :return: Обученная модель, Скор
        """
        model.fit(x=x_train, y=y_train, batch_size=512, epochs=1000,
                  validation_data=(x_test, y_test), callbacks=self.callbacks, verbose=1)

        return model, model.evaluate(x=x_test, y=y_test, batch_size=128, verbose=1)[1]

    def __save_model(self, model, path: str):
        """
        Сохраняет модель
        :param model: Моедль
        :param path: Путь сохранения
        """
        model.save(path)

    def __load_model(self, path: str) -> object:
        """
        Загружает модель
        :param path:
        :return: Модель
        """
        return k.models.load_model(path)

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
