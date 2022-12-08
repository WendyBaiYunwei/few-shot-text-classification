import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

class RFClassifier():
    def __init__(self):
        data_dir = '../data/'
        print('loading dataset and preparing classifier')
        with open(data_dir + 'rf_train_x_nlp.pkl', 'rb') as f:
            trainX = pickle.load(f)

        with open(data_dir + 'rf_train_y_nlp.pkl', 'rb') as f:
            trainY = pickle.load(f)

        self.shot = 5
        self.way = 2

        print(trainX.shape, trainY.shape)
        print('start RF training')
        self.classifier = RandomForestRegressor(n_estimators = 200, random_state = 0, max_features = 4)
        # self.classifier = SVR(max_iter = 100)
        self.classifier.fit(trainX, trainY)
        # self.classifier.fit(trainX, trainY, weights)
        print('done RF training')
        # preds = self.classifier.predict(testX)
        # print(accuracy_score(preds, testY))
        del trainX
        del trainY
    
    def _rf_classify(self, data):
        spt1 = torch.mean(data[:5], 0)
        spt2 = torch.mean(data[5:10], 0)
        spt = [spt1, spt2]
        qry = data[10:]
        diffs = []

        # print(sEmbeddings[0])
        for qEmbedding in qry:
            for class_i in range(self.way):
                avgEmebdding = spt[class_i]
                diff = (avgEmebdding - qEmbedding) ** 2
                diffs.append(diff)  
        # diffs = np.stack(diffs).round(2).reshape(-1, 512)
        diffs = np.stack(diffs).reshape(len(qry)*2, -1)
        preds = self.classifier.predict(diffs)
        # preds *= 100
        # preds = preds.astype(int)
        # preds = np.log(preds / ((1 - preds) + 1e-4))
        preds = preds.reshape(len(qry), -1)
        
        # preds = preds.flatten()
        return torch.from_numpy(preds).cuda()