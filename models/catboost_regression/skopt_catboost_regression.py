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
from models.catboost_regression.catboost_regression import CatboostRegressor
from utils.cuda import turn_off_gpu

# Отключение gpu
turn_off_gpu()

# Описание гиперпараметров модели
dimensions = [Integer(low=500, high=2000, prior='uniform', name='iterations'),
              Real(low=1e-5, high=5e-1, prior='uniform', name='learning_rate'),
              Real(low=5e-1, high=5.0, prior='uniform', name='l2_leaf_reg'),
              Integer(low=4, high=16, prior='uniform', name='depth'),
              Integer(low=1, high=5, prior='uniform', name='min_data_in_leaf'),
              Real(low=1e-2, high=1.0, prior='uniform', name='rsm'),
              Categorical(categories=[False, True], name='langevin'),
              Integer(low=100, high=100000, prior='uniform', name='diffusion_temperature')]

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
    model = CatboostRegressor()
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
                            x0=list(CatboostRegressor.default_model_constructor_parameters.values()))

print('Best Accuracy: %.3f' % (search_result.fun))
print('Best Parameters: %s' % search_result.x)

# Отрисовка графиков
_ = plot_objective(result=search_result, n_points=50)
_ = plot_objective(result=search_result, sample_source='result', n_points=50)
plt.show()
