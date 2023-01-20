import json
import random
from torch.utils.data import Dataset, DataLoader
from itertools import permutations


class DatasetMapper(Dataset):

    def __init__(self, sentences, entities_1, entities_2, relations):
        self.sentences = sentences
        self.entities_1 = entities_1
        self.entities_2 = entities_2
        self.relations = relations

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.entities_1[idx], self.entities_2[idx], self.relations[idx]


# if not given, data_amount is 100 (amount for dev and test)
def prepare_data(data_path, labels2id, batch_size, prediction, pretrain=False, data_amount=680):

    sentences_tot, entities_1_tot, entities_2_tot, relations_tot = [], [], [], []
    for path in data_path.split(' '):
        if pretrain:
            sentences, entities_1, entities_2, relations = read_json_file_pretrain(path, labels2id, data_amount=data_amount, test_set=prediction)
        else:
            sentences, entities_1, entities_2, relations = read_json_file_crossre(path, labels2id)
        sentences_tot += sentences
        entities_1_tot += entities_1
        entities_2_tot += entities_2
        relations_tot += relations

    # check lengths sentences, entities, labels
    assert len(sentences_tot) == len(entities_1_tot)
    assert len(entities_1_tot) == len(entities_2_tot)
    assert len(entities_2_tot) == len(relations_tot)

    do_shuffle = len(data_path.split(' ')) > 1 and not prediction
    data_loader = DataLoader(DatasetMapper(sentences_tot, entities_1_tot, entities_2_tot, relations_tot), batch_size=batch_size, shuffle=do_shuffle)
    return data_loader, relations_tot

# return sentences, idx within the sentence of entity-markers-start, relation labels
def read_json_file_crossre(json_file, labels2id, multi_label=False):

    sentences, entities_1, entities_2, relations = [], [], [], []

    with open(json_file) as data_file:
        for json_elem in data_file:
            abstract = json.loads(json_elem)

            # consider only the sentences with at least 2 entities
            if len(abstract["ner"]) > 1:

                # create all the possible entity pairs
                entity_pairs = permutations(abstract["ner"], 2)

                for entity_pair in entity_pairs:

                    # set the entity tokens to inject in the instance
                    ent1_start = f'<E1>'
                    ent1_end = f'</E1>'
                    ent2_start = f'<E2>'
                    ent2_end = f'</E2>'

                    # build the instance sentence for the model
                    sentence_marked = ''

                    for idx_token in range(len(abstract["sentence"])):

                        # nested entities begin
                        if idx_token == entity_pair[0][0] and idx_token == entity_pair[1][0]:
                            # entity 1 is the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][1] > entity_pair[1][1]:
                                sentence_marked += f'{ent1_start} {ent2_start} {abstract["sentence"][idx_token]} '
                                # entity 2 (the shortest one) is one token long
                                if idx_token == entity_pair[1][1]:
                                    sentence_marked += f'{ent2_end} '
                            # entity 2 is the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{ent2_start} {ent1_start} {abstract["sentence"][idx_token]} '
                                # entity 1 (the shortest one) is one token long
                                if idx_token == entity_pair[0][1]:
                                    sentence_marked += f'{ent1_end} '

                        # match begin entity 1
                        elif idx_token == entity_pair[0][0]:
                            sentence_marked += f'{ent1_start} {abstract["sentence"][idx_token]} '
                            # entity 1 is one token long
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '
                            # entity 1 is a nested entity encapsulated inside entity 2
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                        # match begin entity 2
                        elif idx_token == entity_pair[1][0]:
                            sentence_marked += f'{ent2_start} {abstract["sentence"][idx_token]} '
                            # entity 2 is one token long
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                            # entity 2 is a nested entity encapsulated inside entity 1
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '

                        # nested entities end
                        elif idx_token == entity_pair[0][1] and idx_token == entity_pair[1][1]:
                            # entity 1 in the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][0] < entity_pair[1][0]:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} {ent1_end} '
                            # entity 2 in the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} {ent2_end} '

                        # match end entity 1
                        elif idx_token == entity_pair[0][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} '
                        # match end entity 2
                        elif idx_token == entity_pair[1][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} '

                        # regular token
                        else:
                            sentence_marked += f'{abstract["sentence"][idx_token]} '

                    # retrieve relation label
                    dataset_relations = [(e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) for (e1_s, e1_e, e2_s, e2_e, rel, exp, ns, sa) in abstract["relations"] if e1_s == entity_pair[0][0] and e1_e == entity_pair[0][1] and e2_s == entity_pair[1][0] and e2_e == entity_pair[1][1]]

                    # prepare data
                    if len(dataset_relations) > 0:
                        if multi_label:
                            instance_labels = [0] * len(labels2id.keys())
                            for elem in dataset_relations:
                                instance_labels[labels2id[elem[4]]] = 1
                            relations.append(instance_labels)
                        else:
                            relations.append(labels2id[dataset_relations[0][4]])
                    else:
                        if multi_label:
                            instance_labels = [0] * len(labels2id.keys())
                            instance_labels[labels2id['no-rel']] = 1
                            relations.append(instance_labels)
                        else:
                            relations.append(labels2id['no-rel'])                            
                    sentences.append(sentence_marked.strip())
                    entities_1.append(sentence_marked.split(' ').index(f'{ent1_start}'))
                    entities_2.append(sentence_marked.split(' ').index(f'{ent2_start}'))

    return sentences, entities_1, entities_2, relations


# return sentences, idx within the sentence of entity-markers-start, relation labels
def read_json_file_pretrain(json_file, labels2id, data_amount, test_set=False):

    sentences, entities_1, entities_2, relations = [], [], [], []

    with open(json_file) as data_file:
        # for json_elem in data_file:
        for i in range(data_amount):
            json_elem = data_file.readline()

            # list per sentence: we sample a max of 10 each at the end
            sentences_sent, entities_1_sent, entities_2_sent, relations_sent = [], [], [], []

            abstract = json.loads(json_elem)

            # consider only the sentences with at least 2 entities
            if len(abstract["ner"]) > 1:

                # create all the possible entity pairs
                entity_pairs = permutations(abstract["ner"], 2)

                for entity_pair in entity_pairs:

                    # set the entity tokens to inject in the instance
                    ent1_start = f'<E1>'
                    ent1_end = f'</E1>'
                    ent2_start = f'<E2>'
                    ent2_end = f'</E2>'

                    # build the instance sentence for the model
                    sentence_marked = ''

                    for idx_token in range(len(abstract["sentence"])):

                        # nested entities begin
                        if idx_token == entity_pair[0][0] and idx_token == entity_pair[1][0]:
                            # entity 1 is the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][1] > entity_pair[1][1]:
                                sentence_marked += f'{ent1_start} {ent2_start} {abstract["sentence"][idx_token]} '
                                # entity 2 (the shortest one) is one token long
                                if idx_token == entity_pair[1][1]:
                                    sentence_marked += f'{ent2_end} '
                            # entity 2 is the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{ent2_start} {ent1_start} {abstract["sentence"][idx_token]} '
                                # entity 1 (the shortest one) is one token long
                                if idx_token == entity_pair[0][1]:
                                    sentence_marked += f'{ent1_end} '

                        # match begin entity 1
                        elif idx_token == entity_pair[0][0]:
                            sentence_marked += f'{ent1_start} {abstract["sentence"][idx_token]} '
                            # entity 1 is one token long
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '
                            # entity 1 is a nested entity encapsulated inside entity 2
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                        # match begin entity 2
                        elif idx_token == entity_pair[1][0]:
                            sentence_marked += f'{ent2_start} {abstract["sentence"][idx_token]} '
                            # entity 2 is one token long
                            if idx_token == entity_pair[1][1]:
                                sentence_marked += f'{ent2_end} '
                            # entity 2 is a nested entity encapsulated inside entity 1
                            if idx_token == entity_pair[0][1]:
                                sentence_marked += f'{ent1_end} '

                        # nested entities end
                        elif idx_token == entity_pair[0][1] and idx_token == entity_pair[1][1]:
                            # entity 1 in the biggest: entity 1 encapsulates entity 2
                            if entity_pair[0][0] < entity_pair[1][0]:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} {ent1_end} '
                            # entity 2 in the biggest: entity 2 encapsulates entity 1
                            else:
                                sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} {ent2_end} '

                        # match end entity 1
                        elif idx_token == entity_pair[0][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent1_end} '
                        # match end entity 2
                        elif idx_token == entity_pair[1][1]:
                            sentence_marked += f'{abstract["sentence"][idx_token]} {ent2_end} '

                        # regular token
                        else:
                            sentence_marked += f'{abstract["sentence"][idx_token]} '

                    # retrieve relation label
                    relation = [(e1_s, e1_e, e2_s, e2_e, rel) for (e1_s, e1_e, e2_s, e2_e, rel) in
                                abstract["relations"] if
                                e1_s == entity_pair[0][0] and e1_e == entity_pair[0][1] and e2_s ==
                                entity_pair[1][0] and e2_e == entity_pair[1][1]]

                    # prepare data according to the task (rc only keep positive samples)
                    if len(relation) > 0:
                        relations_sent.append(labels2id[relation[0][4]])
                    else:
                        relations_sent.append(labels2id['no-rel'])
                    sentences_sent.append(sentence_marked.strip())
                    entities_1_sent.append(sentence_marked.split(' ').index(f'{ent1_start}'))
                    entities_2_sent.append(sentence_marked.split(' ').index(f'{ent2_start}'))

            # if train data, for each json sample a max of 5 instances
            if test_set:
                sentences += sentences_sent
                entities_1 += entities_1_sent
                entities_2 += entities_2_sent
                relations += relations_sent
            else:
                ids_samples_to_keep = random.sample(range(len(sentences_sent)), min(5, len(sentences_sent)))
                for id in ids_samples_to_keep:
                    if id < len(sentences_sent):
                        sentences.append(sentences_sent[id])
                        entities_1.append(entities_1_sent[id])
                        entities_2.append(entities_2_sent[id])
                        relations.append(relations_sent[id])

    return sentences, entities_1, entities_2, relations