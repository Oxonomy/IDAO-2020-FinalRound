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
from models.random_forest_regression.random_forest_regression import RandomForestRegression
from utils.cuda import turn_off_gpu

# Отключение gpu
turn_off_gpu()

# Описание гиперпараметров модели
dimensions = [Integer(low=5, high=500, prior='log-uniform', name='n_estimators'),
              Categorical(categories=['mse', 'mae'], name='criterion'),
              Categorical(categories=['auto', 'sqrt', 'log2'], name='max_features'),
              Categorical(categories=[2, 3, 4, 5, 6], name='min_samples_split'),
              Categorical(categories=[1, 2, 3, 4, 5], name='min_samples_leaf'),
              Real(low=0.0, high=1.0, prior='uniform', name='min_impurity_decrease'),
              Real(low=0.0, high=5.0, prior='uniform', name='ccp_alpha')
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
    model = RandomForestRegression()
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
                            x0=list(RandomForestRegression.default_model_constructor_parameters.values()))

print('Best Accuracy: %.3f' % (search_result.fun))
print('Best Parameters: %s' % search_result.x)

# Отрисовка графиков
_ = plot_objective(result=search_result, n_points=50)
_ = plot_objective(result=search_result, sample_source='result', n_points=50)
plt.show()
