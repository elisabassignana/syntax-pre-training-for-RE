import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import compute_class_weight

#
# Loss Functions
#


class LabelLoss(nn.Module):
    def __init__(self, classes, target_train, weight_loss=False):
        super().__init__()
        self._classes = classes
        self._target_train = target_train
        if weight_loss:
            self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1, weight=self.compute_loss_weight())
        else:
            self._xe_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def __repr__(self):
        return f'<{self.__class__.__name__}: loss=XEnt, num_classes={len(self._classes)}>'

    def compute_loss_weight(self):

        weights = compute_class_weight(class_weight='balanced', classes=np.unique(self._target_train), y=self._target_train)

        # within sparse datasets not all the classes appear in the train set
        # add weight = 1 for classes which are not in the training set
        w = {l: w for l, w in zip(np.unique(self._target_train), weights)}
        for label in self._classes:
            if label not in w.keys():
                w[label] = 1
        ordered_w = sorted(w)
        weights = [w[elem] for elem in ordered_w]

        weights = torch.Tensor(weights)

        if torch.cuda.is_available():
            weights = weights.to(torch.device('cuda'))

        return weights

    def forward(self, logits, targets):

        target_labels = torch.LongTensor(targets).to(logits.device)
        loss = self._xe_loss(logits, target_labels)

        return loss

    def get_accuracy(self, predictions, targets):

        target_labels = torch.LongTensor(targets).to(predictions.device)

        # compute label accuracy
        num_label_matches = torch.sum(predictions == target_labels)
        accuracy = float(num_label_matches / predictions.shape[0])

        return accuracy
