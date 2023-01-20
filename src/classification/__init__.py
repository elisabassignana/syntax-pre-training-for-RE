from .classifiers import *
from .losses import *


def load_classifier():
    return MultiLayerPerceptronClassifier, LabelLoss

