import os
import csv
import json
import logging
import argparse
from dotenv import load_dotenv
from src.preprocessing import read_json_file_crossre, read_json_file_pretrain
from itertools import permutations
from sklearn.metrics import classification_report, f1_score

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

def parse_arguments():

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--gold_path', type=str, nargs='?', required=True, help='Path to the gold labels file.')
    arg_parser.add_argument('--pred_path', type=str, nargs='?', required=True, help='Path to the predicted labels file.')
    arg_parser.add_argument('--out_path', type=str, nargs='?', required=True, help='Path where to save scores.')
    arg_parser.add_argument('--summary_exps', type=str, nargs='?', required=True, help='Path to the summary of the overall experiments.')

    return arg_parser.parse_args()

def get_f1(gold_path, predicted_path):

    # get the labels
    label_types = {label: idx for idx, label in enumerate(os.getenv(f"RELATION_LABELS_CROSSRE").split(), 1)}
    label_types['no-rel'] = 0

    # get the gold
    _, _, _, gold = read_json_file_crossre(gold_path, label_types, multi_label=True)

    # get the predicted
    predicted = []
    with open(predicted_path) as predicted_file:
        predicted_reader = csv.reader(predicted_file, delimiter=',')
        next(predicted_reader)
        for line in predicted_reader:
            instance_labels = [0] * len(label_types.keys())
            for elem in line[0].split(' '):
                instance_labels[label_types[elem]] = 1
            predicted.append(instance_labels)

    # check gold and predicted lengths
    assert len(gold) == len(predicted), "Length of gold and predicted labels should be equal."

    # compute classification report and ignore no-rel class for the macro-f1 computation
    labels = ['no-rel'] + os.getenv(f"RELATION_LABELS_CROSSRE").split()
    report = classification_report(gold, predicted, target_names=labels, output_dict=True, zero_division=0)

    # do not consider the no-rel label and the classes with 0 instances in the test set in the macro-f1 computation
    macro_f1 = sum([elem[1]['f1-score'] if elem[0] in label_types.keys() and elem[0] != 'no-rel' and elem[1]['support'] > 0 else 0 for elem in report.items()]) / (sum([1 if elem[0] in label_types.keys() and elem[1]['support'] > 0 else 0 for elem in report.items()]) - 1)

    return macro_f1


if __name__ == '__main__':

    args = parse_arguments()
    logging.info(f"Evaluating {args.gold_path} and {args.pred_path}.")

    macro_f1 = get_f1(args.gold_path, args.pred_path)

    logging.info(f"Saving scores to {args.out_path} -> Macro F1: {macro_f1 * 100}")

    exp = os.path.splitext(os.path.basename(args.pred_path))[0]

    with open(args.summary_exps, 'a') as file:
        file.write(f"Macro F1: {macro_f1 * 100}\n")