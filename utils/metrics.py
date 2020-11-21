import numpy as np
import keras.backend as K
from sklearn.metrics import roc_auc_score


def smape(satellite_predicted_values: np.array, satellite_true_values: np.array) -> float:
    """
    Симметричная средняя абсолютная процентная ошибка
    :param satellite_predicted_values: предсказанное значение
    :param satellite_true_values: истинное значение
    :return: величина ошибки
    """
    return float(np.mean(np.abs((satellite_predicted_values - satellite_true_values)
                                / (np.abs(satellite_predicted_values) + np.abs(satellite_true_values)))))


def score(satellite_predicted_values: np.array, satellite_true_values: np.array) -> float:
    """
    Скор на лидерборде
    :param satellite_predicted_values: предсказанное значение
    :param satellite_true_values: истинное значение
    :return: скор
    """
    return 100 * (1 - smape(satellite_predicted_values, satellite_true_values))


def smape_loss():
    """
    Функция, для передачи метрики керасу
    :return: функция потерь
    """

    def loss(satellite_predicted_values, satellite_true_values):
        return K.mean(K.abs((satellite_predicted_values - satellite_true_values)
                            / (K.abs(satellite_predicted_values) + K.abs(satellite_true_values))))

    return loss


def roc_auc_score_at_K(predicted_proba, target, rate=0.1):
    """
    Area under the ROC curve between the predicted probability and the observed target.
    The area under the ROC curve will be calculated only on the top 10 percent of predictions sorted descendingly.
    :param predicted_proba: предсказанное значение
    :param target: истинное значение
    :param rate: процент меток
    :return: roc auc score at k
    """
    order = np.argsort(-predicted_proba)
    top_k = int(rate * len(predicted_proba))
    return roc_auc_score(target[order][:top_k], predicted_proba[order][:top_k])
