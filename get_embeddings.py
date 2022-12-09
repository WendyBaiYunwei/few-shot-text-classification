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
from fastNLP import Vocabulary
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
        'data': []
    }
    pos = {
        'filename': filename,
        'data': []
    }
    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            if line[-2:] == '-1':
                neg['data'].append(line[:-2])
            else:
                pos['data'].append(line[:-1])
    # check
    neg['data'] = neg['data'][:10] # select only 10 data points from each domain #to-do
    pos['data'] = pos['data'][:10]
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


def get_data(data_path, domains, usage='train'):
    # usage in ['train', 'dev', 'test']
    data = {}
    all_pos = []
    all_neg = []
    for domain in domains:
        for t in ['t2', 't4', 't5']:
            filename = '.'.join([domain, t, usage])
            neg, pos = _parse_data(data_path, filename)
            neg = _process_data(neg)
            pos = _process_data(pos)
            all_pos.append(pos)
            all_neg.append(neg)
    return all_pos, all_neg

def main():
    train_domains, test_domains = get_domains(data_path, config['data']['filtered_list'], config['data']['target_list'])

    pos, neg = get_data(data_path, train_domains)

    with open('../data/embedding_nlp_pos.pkl', 'wb') as f:
        pickle.dump(pos, f) #dict{filename:, embeddings:[]}
    
    with open('../data/embedding_nlp_neg.pkl', 'wb') as f:
        pickle.dump(neg, f)

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
