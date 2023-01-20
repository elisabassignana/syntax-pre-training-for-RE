import logging
import os
import csv
import sys
import torch
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from src.preprocessing import prepare_data
from src.classification import load_classifier

load_dotenv()

class Stilt:

    def __init__(self, train, exp_path, embedding_model, epochs, batch_size, learning_rate, data_amount):
        self.train = train
        self.exp_path = exp_path
        self.embedding_model = embedding_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_amount = data_amount

    def run(self, classifier, criterion, optimizer, dataset):
        stats = defaultdict(list)

        # iterate over batches
        batch_idx = 0
        for sentences, entities_1, entities_2, labels in dataset:
            batch_idx += 1

            # zero out previous gradients
            optimizer.zero_grad()

            # forward pass
            predictions = classifier(list(sentences), entities_1, entities_2)

            # propagate loss
            loss = criterion(predictions['flat_logits'], labels)
            loss.backward()
            optimizer.step()

            # calculate and store accuracy metrics
            stats['loss'].append(float(loss.detach()))
            stats['accuracy'].append(float(criterion.get_accuracy(predictions['labels'].detach(), labels)))

            # print batch statistics
            sys.stdout.write(
                f"\r[Pre-train | Batch {batch_idx}] "
                f"Acc: {np.mean(stats['accuracy']):.4f}, "
                f" Loss: {np.mean(stats['loss']):.4f}")
            sys.stdout.flush()

        # clear line
        print("\r", end='')

        return stats

    def save_predictions(self, path, data, pred_labels):

        with open(path, 'w', encoding='utf8', newline='') as output_file:
            csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
            idx = 0

            csv_writer.writerow(['label', 'position-ent1-marker', 'position-ent2-marker', 'text'])
            for sentences, entities1, entities2, _ in data:
                for s, e1, e2 in zip(sentences, entities1, entities2):
                    csv_writer.writerow([pred_labels[idx], e1.item(), e2.item(), s])
                    idx += 1

    def do_stilt(self):

        logging.info(f"========== Preparing for STILT ==========")

        # setup labels
        label_types = {label: idx for idx, label in enumerate(os.getenv(f"RELATION_LABELS_SYNTAX").split(), 1)}
        label_types['no-rel'] = 0

        # setup data
        train_data, target_train = prepare_data(self.train, label_types, self.batch_size, False, True, self.data_amount)
        logging.info(f"Loaded {train_data} (train).")

        # load classifier and loss constructors based on identifier
        classifier_constructor, loss_constructor = load_classifier()

        # setup classifier
        classifier = classifier_constructor(
            emb_model=self.embedding_model, classes=list(label_types.values())
        )
        logging.info(f"Using classifier:\n{classifier}")

        # setup loss
        criterion = loss_constructor(list(label_types.values()), target_train)
        logging.info(f"Using criterion {criterion}.")

        # setup optimizer
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=self.learning_rate)
        logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {self.learning_rate}.")

        logging.info(f"----- Running STILT -----")

        # train loop
        for ep_idx in range(self.epochs):

            # iterate over training batches and update classifier weights
            ep_stats = self.run(
                classifier, criterion, optimizer, train_data,
            )

            # print statistics
            logging.info(
                f"[Epoch {ep_idx + 1}/{self.epochs}] Train completed with "
                f"Acc: {np.mean(ep_stats['accuracy']):.4f}, "
                f"Loss: {np.mean(ep_stats['loss']):.4f}"
            )

        logging.info(f"Training completed after {ep_idx + 1} epochs.")

        logging.info(f"----- STILT Ended -----")