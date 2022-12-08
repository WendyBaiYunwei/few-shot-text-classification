import configparser
import pickle
import os
import torch
from torch import optim
from criterion import Criterion
from rf_classifier import RFClassifier
import random
import numpy as np


def test():
    correct = 0.
    count = 0.
    for data, target in test_loader:
        predict = model._rf_classify(data)
        _, acc = criterion(predict, target.cuda())
        amount = len(target) - support * 2
        correct += acc * amount
        count += amount
    acc = correct / count
    print('Test Acc: {}'.format(acc))
    return acc


def main():
    # train
    # preprocess dataset, load
    # fit, predict
    test()


if __name__ == "__main__":
    # config
    config = configparser.ConfigParser()
    config.read("config.ini")

    # seed
    seed = int(config['model']['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # data loaders
    test_loader = pickle.load(open(os.path.join(config['data']['path'], config['data']['test_loader2']), 'rb'))

    # word2vec weights
    weights = pickle.load(open(os.path.join(config['data']['path'], config['data']['weights']), 'rb'))

    # model & optimizer & criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    support = int(config['model']['support'])
    model = RFClassifier()
    criterion = Criterion(way=int(config['model']['class']),
                          shot=int(config['model']['support']))

    # writer
    main()
