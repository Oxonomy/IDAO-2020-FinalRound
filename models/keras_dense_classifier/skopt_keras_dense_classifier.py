import random
import keras as k
import matplotlib.pyplot as plt
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective
from skopt.utils import use_named_args
from sklearn.datasets import fetch_california_housing, make_blobs

import config as c
from utils.cuda import turn_off_gpu
from models.keras_dense_classifier.keras_dense_classifier import KerasDenseClassifier as KDC

# Отключение gpu
turn_off_gpu()

# Описание гиперпараметров модели
dimensions = [Categorical(categories=['sigmoid', 'softmax', 'relu', 'softsign', 'tanh'], name='activation'),
              Categorical(categories=['sigmoid', 'softmax', 'relu', 'softsign', 'tanh'], name='output_node_activation'),
              Real(low=1e-6, high=1e2, prior='log-uniform', name='learning_rate'),
              Integer(low=1, high=5, name='num_dense_layers'),
              Integer(low=3, high=30, name='dense_shape'),
              Integer(low=2, high=10, name='early_patience')
              ]

# Для примера генерируется случайный датасет
x, y = make_blobs(n_samples=1000, centers=2, n_features=2, cluster_std=2)

# Глобальные переменные
best_score = 0.0
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
    c.SEED = random.randint(0, 3000)

    # Создание, обучение и тестирование модели
    model = KDC()
    score = model.fit_ensemble(10, 1, x, y, model_constructor_parameters)

    print("Score: {0:.2%}".format(score))
    print("Best score: {0:.2%}".format(best_score))
    print("Fitness iteration:", fit_iteration)
    print('Seed', c.SEED)
    print('--||--' * 10, '\n')
    fit_iteration += 1

    # Сохранение лучшей модели
    if score > best_score:
        model.save_ensemble('best_model')
        best_score = score

    # Очистка памяти
    del model
    k.backend.clear_session()

    # Возврат скора, так-как задача минимизации, то чем лучше модель - тем меньше результат
    return -score


# Подбор гиперпараметров, описание параметров смотри:
# https://scikit-optimize.github.io/stable/modules/generated/skopt.plots.plot_objective.html
search_result = gp_minimize(func=skopt_fit,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=30,
                            n_jobs=10,
                            x0=list(KDC.default_model_constructor_parameters.values()))

print('Best Accuracy: %.3f' % (-search_result.fun))
print('Best Parameters: %s' % search_result.x)

# Отрисовка графиков
_ = plot_objective(result=search_result, n_points=30)
_ = plot_objective(result=search_result, sample_source='result', n_points=30)
plt.show()
