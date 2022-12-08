import configparser
import os
import re
import string
import pickle
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import Dataset
from dataloader import TrainDataLoader
from utils import padding, batch_padding
from sentence_transformers import SentenceTransformer


def _parse_list(data_path, list_name):
    domain = set()
    with open(os.path.join(data_path, list_name), 'r', encoding='utf-8') as f:
        for line in f:
            domain.add(line.strip('\n'))
    return domain


def get_domains(data_path, filtered_name, target_name):
    all_domains = _parse_list(data_path, filtered_name)
    test_domains = _parse_list(data_path, target_name)
    train_domains = all_domains - test_domains
    print('train domains', len(train_domains), 'test_domains', len(test_domains))
    return sorted(list(train_domains)), sorted(list(test_domains))


def _parse_data(data_path, filename):
    neg = {
        'filename': filename,
        'data': [],
        'target': []
    }
    pos = {
        'filename': filename,
        'data': [],
        'target': []
    }
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if line[-2:] == '-1':
                neg['data'].append(line[:-2])
                neg['target'].append(0)
            else:
                pos['data'].append(line[:-1])
                pos['target'].append(1)
    # check
    print(filename, 'neg', len(neg['data']), 'pos', len(pos['data']))
    return neg, pos


def _process_data(data_dict):
    for i in range(len(data_dict['data'])):
        text = data_dict['data'][i]
        # ignore string.punctuation
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        # string.whitespace -> space
        text = re.sub('[%s]' % re.escape(string.whitespace), ' ', text)
        # lower case
        text = text.lower()
        # split by whitespace
        # text = text.split()
        # replace
        data_dict['data'][i] = model.encode(text)
    return data_dict


def _get_data(data_path, domains, usage):
    # usage in ['train', 'dev', 'test']
    data = {}
    for domain in domains:
        for t in ['t2', 't4', 't5']:
            filename = '.'.join([domain, t, usage])
            neg, pos = _parse_data(data_path, filename)
            neg = _process_data(neg)
            pos = _process_data(pos)
            data[filename] = {'neg': neg, 'pos': pos}
    return data


def _combine_data(support_data, data):
    for key in data:
        key_split = key.split('.')[0:-1] + ['train']
        support_key = '.'.join(key_split)
        for value in data[key]:
            data[key][value]['support_data'] = copy.deepcopy(support_data[support_key][value]['data'])
            data[key][value]['support_target'] = copy.deepcopy(support_data[support_key][value]['target'])
    return data


def get_test_data(data_path, domains):
    # get dev, test data
    support_data = _get_data(data_path, domains, 'train')
    test_data = _get_data(data_path, domains, 'test') # data[filename] = {'neg': neg, 'pos': pos}
    test_data = _combine_data(support_data, test_data)
    print('test data', len(test_data))
    return test_data


# def get_vocabulary(data, min_freq):
#     # train data -> vocabulary
#     vocabulary = Vocabulary(min_freq=min_freq, padding='<pad>', unknown='<unk>')
#     for filename in data:
#         for value in data[filename]:
#             for word_list in data[filename][value]['data']:
#                 vocabulary.add_word_lst(word_list)
#     vocabulary.build_vocab()
#     print('vocab size', len(vocabulary), 'pad', vocabulary.padding_idx, 'unk', vocabulary.unknown_idx)
#     return vocabulary


# def _idx_text(text_list, vocabulary):
#     for i in range(len(text_list)):
#         for j in range(len(text_list[i])):
#             text_list[i][j] = vocabulary.to_index(text_list[i][j])
#     return text_list


# def idx_all_data(data, vocabulary):
#     for filename in data:
#         for value in data[filename]:
#             for key in data[filename][value]:
#                 if key in ['data', 'support_data']:
#                     data[filename][value][key] = _idx_text(data[filename][value][key], vocabulary)
#     return data

def get_test_loader2(full_data, support, query):
    loader = []
    for filename in full_data:
        # support
        support_data = full_data[filename]['neg']['support_data'][0:support] + full_data[filename]['pos']['support_data'][0:support]
        support_data = torch.tensor(np.array(support_data))
        # support data should return embeddings
        support_target = full_data[filename]['neg']['support_target'][0:support] + full_data[filename]['pos']['support_target'][0:support]
        support_target = torch.tensor(support_target)
        # query
        neg_dl = DataLoader(Dataset(full_data[filename]['neg']), batch_size=query * 2, shuffle=False, drop_last=False)
        pos_dl = DataLoader(Dataset(full_data[filename]['pos']), batch_size=query * 2, shuffle=False, drop_last=False)
        # combine
        for dl in [neg_dl, pos_dl]:
            for batch_data, batch_target in dl:
                support_data_cp, support_target_cp = copy.deepcopy(support_data), copy.deepcopy(support_target)
                data = torch.cat([support_data_cp, batch_data], dim=0)
                target = torch.cat([support_target_cp, batch_target], dim=0)
                loader.append((data, target))
    print('test loader length', len(loader))
    return loader

def main():
    train_domains, test_domains = get_domains(data_path, config['data']['filtered_list'], config['data']['target_list'])

    test_data = get_test_data(data_path, test_domains)

    # print(dev_data['books.t2.dev']['neg']['support_data'])
    # print(dev_data['books.t2.dev']['neg']['support_target'])

    support = int(config['model']['support'])
    query = int(config['model']['query'])
    test_loader2 = get_test_loader2(test_data, support, query)
    
    pickle.dump(test_loader2, open(os.path.join(config['data']['path'], config['data']['test_loader2']), 'wb'))

if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['data']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_path = config['data']['path']
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    main()
