import numpy as np
from sklearn.metrics import mean_squared_error


class Layer:
    def __init__(self, ensemble_models: list, models: list):
        self.ensemble_models = ensemble_models
        self.models = models

    def predict(self, x) -> dict:
        predictions = {}
        for i, model in enumerate(self.models):
            predictions[str(i) + '_' + model.model_name] = model.predict(x)

        for i, ensemble_model in enumerate(self.ensemble_models):
            predictions[str(i) + '_' + ensemble_model.model_name] = ensemble_model.predict(x)

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

        for ensemble_model in self.ensemble_models:
            ensemble_model.fit_ensemble(10, 5, x, y, ensemble_model.default_model_constructor_parameters)
