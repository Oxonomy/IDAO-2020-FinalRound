'''import os
os.chdir('../..')'''

import random
import keras as k
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from sklearn.datasets import fetch_california_housing, make_blobs

import config as c
from models.svm_regression.svm_regression import SVMR
from utils.cuda import turn_off_gpu

# Отключение gpu
turn_off_gpu()

# Описание гиперпараметров модели
dimensions = [Categorical(categories=['linear', 'rbf', 'sigmoid'], name='kernel'),
              Real(low=1e-6, high=1e2, prior='log-uniform', name='gamma'),
              Real(low=1e-6, high=1e-1, prior='log-uniform', name='tol'),
              Real(low=1e-6, high=1e2, prior='log-uniform', name='C'),
              Real(low=1e-6, high=1.0, prior='log-uniform', name='epsilon')
              ]

# Для примера генерируется случайный датасет
x, y = make_blobs(n_samples=1000, centers=5, n_features=2, cluster_std=2)

# Глобальные переменные
best_score = 999999999.0
fit_iteration = 0


@use_named_args(dimensions=dimensions)
def skopt_fit(**model_constructor_parameters):
    """
    Создает, обучает и тестирует модель с задаными гиперпараметрами
    :param model_constructor_parameters: гиперпараметры
    :return: Скор
    """
    print(model_constructor_parameters)
    global x, y, best_score, fit_iteration
    c.SEED = random.randint(0, 10000)

    # Создание, обучение и тестирование модели
    model = SVMR()
    model.create_model(model_constructor_parameters)
    score = model.fit_model(x, y)

    print("Score: {0:.2}".format(score))
    print("Best score: {0:.2}".format(best_score))
    print("Fitness iteration:", fit_iteration)
    print('Seed', c.SEED)
    print('--||--' * 10, '\n')
    fit_iteration += 1

    # Сохранение лучшей модели
    if score < best_score:
        model.save_model('best_model')
        best_score = score

    # Очистка памяти
    del model
    k.backend.clear_session()

    # Возврат скора, так-как задача минимизации, то чем лучше модель - тем меньше результат
    return score


# Подбор гиперпараметров, описание параметров смотри:
# https://scikit-optimize.github.io/stable/modules/generated/skopt.plots.plot_objective.html
search_result = gp_minimize(func=skopt_fit,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=30,
                            n_jobs=10,
                            x0=list(SVMR.default_model_constructor_parameters.values()))

print('Best rmse: %.3f' % (search_result.fun))
print('Best Parameters: %s' % search_result.x)

# Отрисовка графиков
_ = plot_objective(result=search_result, n_points=50)
_ = plot_objective(result=search_result, sample_source='result', n_points=50)
plt.show()
