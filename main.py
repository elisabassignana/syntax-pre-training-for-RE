import os
import sys
import csv
import argparse
import logging
import torch
import random
import numpy as np
from collections import defaultdict
from src.preprocessing import prepare_data
from src.classification import load_classifier
from src.classification.embeddings import TransformerEmbeddings
from src.run_stilt import Stilt
from dotenv import load_dotenv

load_dotenv()

def parse_arguments():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_path', help='path/paths to training data')
    arg_parser.add_argument('--dev_path', help='path to the dev data')
    arg_parser.add_argument('--test_path', help='path to the test data')
    arg_parser.add_argument('--exp_path', help='path to the experiment directory')

    arg_parser.add_argument('-lm', '--language_model', type=str, default='bert-base-cased')
    arg_parser.add_argument('-po', '--prediction_only', action='store_true', default=False, help='set flag to run prediction on the validation data and exit (default: False)')

    arg_parser.add_argument('--stilt_train', help='path/paths to training data')
    arg_parser.add_argument('--stilt_epochs', type=int, help='number of epochs to pre-train the encoder on', default=1)
    arg_parser.add_argument('--stilt_batch_size', type=int, default=12, help='maximum number of sentences per batch during pre-training (default: 12)')
    arg_parser.add_argument('--stilt_learning_rate', type=float, default=1e-5, help='learning rate during pre-training (default: 1e-5)')
    arg_parser.add_argument('--stilt_data_amount', type=int, default=680, help='amount of instances for pre-training')

    arg_parser.add_argument('-e', '--epochs', type=int, default=50, help='maximum number of epochs (default: 50)')
    arg_parser.add_argument('-bs', '--batch_size', type=int, default=12, help='maximum number of sentences per batch (default: 12)')
    arg_parser.add_argument('-lr', '--learning_rate', type=float, default=2e-5, help='learning rate (default: 2e-5)')
    arg_parser.add_argument('-wl', '--weight_loss', action='store_true', default=True, help='weight the loss in respect to the label distribution of the training set')
    arg_parser.add_argument('-es', '--early_stop', type=int, default=3, help='maximum number of epochs without improvement (default: 3)')
    arg_parser.add_argument('-rs', '--seed', type=int, default=1, help='seed for probabilistic components (default: None)')

    return arg_parser.parse_args()

def set_experiments(out_path, prediction=False):

    if not os.path.exists(out_path):
        if prediction:
            print(f"Experiment path '{out_path}' does not exist. Cannot run prediction. Exiting.")
            exit(1)

        # if output dir does not exist, create it (new experiment)
        print(f"Path '{out_path}' does not exist. Creating...")
        os.mkdir(out_path)
    # if output dir exist, check if predicting
    else:
        # if not predicting, verify overwrite
        if not prediction:
            response = None

            while response not in ['y', 'n']:
                response = input(f"Path '{out_path}' already exists. Overwrite? [y/n] ")
            if response == 'n':
                exit(1)

    # setup logging
    log_format = '%(message)s'
    log_level = logging.INFO
    if prediction:
        logging.basicConfig(filename=os.path.join(out_path, 'eval.log'), filemode='w', format=log_format, level=log_level)
    else:
        logging.basicConfig(filename=os.path.join(out_path, 'classify.log'), filemode='w', format=log_format, level=log_level)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

def run(classifier, criterion, optimizer, dataset, mode='train', return_predictions=False):
    stats = defaultdict(list)

    # set model to training mode
    if mode == 'train':
        classifier.train()
    # set model to eval mode
    elif mode == 'eval':
        classifier.eval()

    # iterate over batches
    batch_idx = 0
    for sentences, entities_1, entities_2, labels in dataset:
        batch_idx += 1

        # when training, perform both forward and backward pass
        if mode == 'train':
            # zero out previous gradients
            optimizer.zero_grad()

            # forward pass
            predictions = classifier(list(sentences), entities_1, entities_2)

            # propagate loss
            loss = criterion(predictions['flat_logits'], labels)
            loss.backward()
            optimizer.step()

        # when evaluating, perform forward pass without gradients
        elif mode == 'eval':
            with torch.no_grad():
                # forward pass
                predictions = classifier(list(sentences), entities_1, entities_2)

                # calculate loss
                loss = criterion(predictions['flat_logits'], labels)

        # calculate and store accuracy metrics
        stats['loss'].append(float(loss.detach()))
        stats['accuracy'].append(float(criterion.get_accuracy(predictions['labels'].detach(), labels)))

        # store predictions
        if return_predictions:
            # iterate over inputs items
            for sidx in range(predictions['labels'].shape[0]):
                # append non-padding predictions as list
                predicted_labels = predictions['labels'][sidx]
                stats['predictions'].append(predicted_labels[predicted_labels != -1].item())

        # print batch statistics
        sys.stdout.write(
                f"\r[{mode.capitalize()} | Batch {batch_idx}] "
                f"Acc: {np.mean(stats['accuracy']):.4f}, "
                f" Loss: {np.mean(stats['loss']):.4f}")
        sys.stdout.flush()

    # clear line
    print("\r", end='')

    return stats


def save_predictions(path, data, pred_labels):

    with open(path, 'w', encoding='utf8', newline='') as output_file:
        csv_writer = csv.writer(output_file, quoting=csv.QUOTE_ALL)
        idx = 0

        csv_writer.writerow(['label','position-ent1-marker','position-ent2-marker','text'])
        for sentences, entities1, entities2, _ in data:
            for s, e1, e2 in zip(sentences, entities1, entities2):
                csv_writer.writerow([pred_labels[idx], e1.item(), e2.item(), s])
                idx += 1

if __name__ == '__main__':

    args = parse_arguments()
    set_experiments(args.exp_path, prediction=args.prediction_only)

    # set random seeds
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    random.seed(args.seed)

    # setup labels
    label_types = {label: idx for idx, label in enumerate(os.getenv(f"RELATION_LABELS_CROSSRE").split(), 1)}
    label_types['no-rel'] = 0
    task_type = 'no-rel'

    # setup data
    if args.prediction_only:
        test_data, _ = prepare_data(args.test_path, label_types, args.batch_size, args.prediction_only)
        logging.info(f"Loaded {test_data} (test).")
        _, target_train = prepare_data(args.train_path, label_types, args.batch_size, args.prediction_only)
    else:
        train_data, target_train = prepare_data(args.train_path, label_types, args.batch_size, args.prediction_only)
        logging.info(f"Loaded {train_data} (train).")
        dev_data, _ = prepare_data(args.dev_path, label_types, args.batch_size, args.prediction_only)
        logging.info(f"Loaded {dev_data} (dev).")

    # load embedding model
    embedding_model = TransformerEmbeddings(
        args.language_model
    )
    logging.info(f"Loaded {embedding_model}.")

    if args.stilt_train != None:
        stilt = Stilt(args.stilt_train, args.exp_path, embedding_model, args.stilt_epochs,
                      args.stilt_batch_size, args.stilt_learning_rate, args.stilt_data_amount)
        stilt.do_stilt()

    # load classifier and loss constructors based on identifier
    classifier_constructor, loss_constructor = load_classifier()

    # setup classifier
    classifier = classifier_constructor(
        emb_model=embedding_model, classes=list(label_types.values())
    )
    logging.info(f"Using classifier:\n{classifier}")
    
    # load pre-trained model for prediction
    if args.prediction_only:
        classifier_path = os.path.join(args.exp_path, f'best.pt')
        if not os.path.exists(classifier_path):
            logging.error(f"[Error] No pre-trained model available in '{classifier_path}'. Exiting.")
            exit(1)
        classifier = classifier_constructor.load(classifier_path)
        logging.info(f"Loaded pre-trained classifier from '{classifier_path}'.")

    # setup loss
    criterion = loss_constructor(list(label_types.values()), target_train, args.weight_loss)
    logging.info(f"Using criterion {criterion}.")

    # main prediction call
    if args.prediction_only:
        stats = run(
            classifier, criterion, None, test_data,
            mode='eval', return_predictions=True
        )

        # convert label indices back to string labels
        idx_2_label = {idx: lbl for lbl, idx in label_types.items()}
        pred_labels = [idx_2_label[pred] for pred in stats['predictions']]

        name_file = 'pred'
        pred_path = os.path.join(args.exp_path, f'{os.path.splitext(os.path.basename(args.test_path))[0]}-{name_file}.csv')

        save_predictions(pred_path, test_data, pred_labels)

        logging.info(
            f"Prediction completed with Acc: {np.mean(stats['accuracy']):.4f}, "
            f"Loss: {np.mean(stats['loss']):.4f} (mean over batches).")
        logging.info(f"Saved results from '{pred_path}'. Exiting.")
        exit()

    # setup optimizer
    optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=args.learning_rate)
    logging.info(f"Optimizing using {optimizer.__class__.__name__} with learning rate {args.learning_rate}.")

    # main loop
    stats = defaultdict(list)
    for ep_idx in range(args.epochs):

        # iterate over training batches and update classifier weights
        ep_stats = run(
            classifier, criterion, optimizer,
            train_data, mode='train'
        )

        # print statistics
        logging.info(
            f"[Epoch {ep_idx + 1}/{args.epochs}] Train completed with "
            f"Acc: {np.mean(ep_stats['accuracy']):.4f}, "
            f"Loss: {np.mean(ep_stats['loss']):.4f}"
        )

        # iterate over batches in dev split
        ep_stats = run(
            classifier, criterion, None, 
            dev_data, mode='eval'
        )

        # store and print statistics
        for stat in ep_stats:
            stats[stat].append(np.mean(ep_stats[stat]))

        # print statistics
        logging.info(
            f"[Epoch {ep_idx + 1}/{args.epochs}] Eval completed with "
            f"Acc: {np.mean(ep_stats['accuracy']):.4f}, "
            f"Loss: {np.mean(ep_stats['loss']):.4f}"
        )

        cur_eval_loss = stats['loss'][-1]

        # save most recent model
        path = os.path.join(args.exp_path, f'newest.pt')
        classifier.save(path)
        logging.info(f"Saved model from epoch {ep_idx + 1} to '{path}'.")

        # save best model
        if cur_eval_loss <= min(stats['loss']):
            path = os.path.join(args.exp_path, f'best.pt')
            classifier.save(path)
            logging.info(f"Saved model with best loss {cur_eval_loss:.4f} to '{path}'.")

        # check for early stopping
        if (ep_idx - stats['loss'].index(min(stats['loss']))) >= args.early_stop:
            logging.info(f"No improvement since {args.early_stop} epochs ({min(stats['loss']):.4f} loss). Early stop.")
            break

    logging.info(f"Training completed after {ep_idx + 1} epochs.")
