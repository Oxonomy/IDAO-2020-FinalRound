from sklearn.datasets import make_blobs

import config as c
from pipeline.Layer import Layer
from utils.cuda import turn_off_gpu
from models.svm.svm import SVM
from models.keras_dense_classifier.keras_dense_classifier import KerasDenseClassifier as KDC


turn_off_gpu()

x, y = make_blobs(n_samples=10000, centers=2, n_features=2, cluster_std=3)

ensemble_models = []
models = []

ensemble_models.append(KDC())
models.append(SVM())

layer = Layer(ensemble_models, models)
layer.fit(x, y)
print(layer.evaluate(x, y))
