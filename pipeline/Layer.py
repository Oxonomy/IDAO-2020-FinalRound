import numpy as np
from sklearn.metrics import mean_squared_error


class Layer:
    def __init__(self, models: list):
        self.models = models

    def predict(self, x) -> dict:
        predictions = {}
        for i, model in enumerate(self.models):
            predictions[str(i) + '_' + model.model_name] = model.predict(x)

        return predictions

    def evaluate(self, x, y, metric=mean_squared_error):
        predictions = self.predict(x)
        score = {}
        for key in predictions.keys():
            score[key] = metric(y, predictions[key])
        return score

    def fit(self, x, y):
        for model in self.models:
            model.create_model(model.default_model_constructor_parameters)
            model.fit_model(x, y, test_size=0)
