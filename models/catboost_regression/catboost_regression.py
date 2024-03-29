from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

import config as c
from pipeline.model import Model
from utils.metrics import roc_auc_score_at_K
from utils.preprocess import reset_averages


class CatboostRegressor(Model):
    default_model_constructor_parameters = {
        'iterations': 1000,
        'learning_rate': 3e-2,
        'l2_leaf_reg': 3.0,  # any pos value
        'depth': 6,  # int up to 16
        'min_data_in_leaf': 1,  # 1,2,3,4,5
        'rsm': 1,  # 0.01 .. 1.0
        'langevin': False,
        'diffusion_temperature': 10000  # 100 ... 100000
    }

    def __init__(self):
        super().__init__("catboost_regression")

    def create_model(self, parameters: dict):
        """
        Созднание модели
        :param parameters: Гиперпараметры модели
        """
        self.model = CatBoostRegressor(iterations=parameters['iterations'],
                                       learning_rate=parameters['learning_rate'],
                                       l2_leaf_reg=parameters['l2_leaf_reg'],
                                       depth=parameters['depth'],
                                       min_data_in_leaf=parameters['min_data_in_leaf'],
                                       loss_function="RMSE", eval_metric='AUC', rsm = 1,
                                       diffusion_temperature=parameters['diffusion_temperature'],
                                       random_state=c.SEED,
                                       verbose=0)

    def fit_model(self, x, y, test_size=0.2) -> float:
        """
        Обучение модели
        :return: Обученная модель, Скор
        """
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=c.SEED)
        self.model.fit(x_train, y_train)
        return self.score(x_test, y_test)

    def score(self, x, y):
        y_prediction = self.predict(x)
        return -roc_auc_score_at_K(reset_averages(y_prediction), y, rate=0.1)
