import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective
from skopt.utils import use_named_args

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold

import keras as k
import matplotlib.pyplot as plt

dim_learning_rate = Real(low=1e-8, high=1e-1, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=6, name='num_dense_layers')
dim_dense_shape = Integer(low=2, high=512, name='dense_shape')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dim_batch_size = Integer(low=32, high=2048, name='batch_size')
dim_epochs = Integer(low=8, high=64, name='epochs')

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_dense_shape,
              dim_activation,
              dim_batch_size,
              dim_epochs]

best_accuracy = 0.0
ensembles_number = 8
fitness_iteration = 0


def create_model(input_shape, learning_rate, num_dense_layers, dense_shape, activation):
    model = k.Sequential()

    for _ in range(num_dense_layers):
        model.add(k.layers.Dense(dense_shape, activation=activation))
    model.add(k.layers.Dense(1, activation='relu'))

    opt = k.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, dense_shape,
            activation, batch_size, epochs):
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', dense_shape)
    print('activation:', activation)
    print('batch_size:', batch_size)
    print('epochs:', epochs)
    print()

    global data
    global best_accuracy
    global ensembles_number
    global fitness_iteration

    # Загрузка данных
    x, y = data
    x -= x.min(axis=0)
    x /= x.max(axis=0)

    models = []
    accuracy = 0

    kf = KFold(n_splits=ensembles_number, random_state=True)
    for train, test in kf.split(x):
        # Обучение модели
        model = create_model(input_shape=(8),
                             learning_rate=learning_rate,
                             num_dense_layers=num_dense_layers,
                             dense_shape=dense_shape,
                             activation=activation)

        history = model.fit(x=x[train],
                            y=y[train],
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(x[test], y[test]),
                            verbose=0)

        mean_absolute_error = history.history['val_mean_absolute_error'][-1] / 5
        models.append(model)
        print(1 - mean_absolute_error)
        accuracy += (1 - mean_absolute_error) / ensembles_number

    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print("Best accuracy: {0:.2%}".format(best_accuracy))
    print("Fitness iteration:", fitness_iteration)
    print()

    # Сохранение ансамбля
    if accuracy > best_accuracy:
        #for model in models:
        #    model.save('best_model')
        best_accuracy = accuracy

    del model
    k.backend.clear_session()

    fitness_iteration += 1

    return -accuracy


data = fetch_california_housing(return_X_y=True)

default_parameters = [1e-5, 1, 3, 'sigmoid', 128, 10]

search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI',
                            n_calls=100,
                            x0=default_parameters)

fig, ax = plot_objective(result=search_result)
plt.show()
