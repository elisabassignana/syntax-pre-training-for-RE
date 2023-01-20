import torch
import torch.nn as nn
from src.classification.embeddings import get_marker_embeddings


#
# Base Classifier
#


class EmbeddingClassifier(nn.Module):
    def __init__(self, emb_model, lbl_model, classes):
        super().__init__()
        # internal models
        self._emb = emb_model
        self._emb_pooling = get_marker_embeddings
        self._lbl = lbl_model
        # internal variables
        self._classes = classes
        # move model to GPU if available
        if torch.cuda.is_available():
            self.to(torch.device('cuda'))

    def __repr__(self):
        return f'''<{self.__class__.__name__}:
                emb_model = {self._emb},
                num_classes = {len(self._classes)}
                >'''

    def train(self, mode=True):
        super().train(mode)
        return self

    def save(self, path):
        torch.save(self, path)

    @staticmethod
    def load(path):
        model = torch.load(path, map_location=torch.device('cpu'))
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
        return model

    def forward(self, sentences, entities_1, entities_2):
        # embed sentences (batch_size, seq_length) -> (batch_size, max_length, emb_dim)
        emb_tokens, att_tokens, encodings = self._emb(sentences)

        # prepare sentence embedding tensor (batch_size, emb_dim)
        emb_sentences = torch.zeros((emb_tokens.shape[0], emb_tokens.shape[2]*2), device=emb_tokens.device)
        # iterate over sentences and pool relevant tokens
        for sidx in range(emb_tokens.shape[0]):
            if len(self._emb_pooling(emb_tokens[sidx, :torch.sum(att_tokens[sidx]), :], encodings[sidx], entities_1[sidx], entities_2[sidx])) == (self._emb.emb_dim*2):
                emb_sentences[sidx, :] = self._emb_pooling(emb_tokens[sidx, :torch.sum(att_tokens[sidx]), :], encodings[sidx], entities_1[sidx], entities_2[sidx])

        # set embedding attention mask to cover each sentence embedding
        att_sentences = torch.ones((att_tokens.shape[0], 1), dtype=torch.bool)

        # logits for all tokens in all sentences + padding -inf (batch_size, max_len, num_labels)
        logits = torch.ones(
            (att_sentences.shape[0], att_sentences.shape[1], len(self._classes)),
            device=emb_sentences.device
        ) * float('-inf')

        # pass through classifier
        flat_logits = self._lbl(emb_sentences)  # (num_words, num_labels)
        logits[att_sentences, :] = flat_logits  # (batch_size, max_len, num_labels)

        labels = self.get_labels(logits.detach())

        results = {
            'labels': labels,
            'logits': logits,
            'flat_logits': flat_logits
        }

        return results

    def get_labels(self, logits):
        # get predicted labels with maximum probability (padding should have -inf)
        labels = torch.argmax(logits, dim=-1)  # (batch_size, max_len)
        # add -1 padding label for -inf logits
        labels[(logits[:, :, 0] == float('-inf'))] = -1

        return labels

#
# Head Classifiers
#

class MultiLayerPerceptronClassifier(EmbeddingClassifier):

    def __init__(self, emb_model, classes):

        lbl_model = nn.Sequential(
            nn.Linear(emb_model.emb_dim*2, len(classes))
        )

        super().__init__(
            emb_model=emb_model, lbl_model=lbl_model, classes=classes
        )
