import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pickle
import json
from sklearn.ensemble import RandomForestRegressor
from model.models import FewShotModel
from sklearn.svm import SVR

class RFClassifier(FewShotModel):
    def __init__(self):
        data_dir = '../data/'
        print('loading dataset and preparing classifier')
        with open(data_dir + 'rf_trainX.pkl', 'rb') as f:
            trainX = pickle.load(f)

        with open(data_dir + 'rf_trainY.pkl', 'rb') as f:
            trainY = pickle.load(f)

        self.shot = 5
        self.way = 2

        print(trainX.shape, trainY.shape)
        print('start RF training')
        # self.classifier = RandomForestRegressor(n_estimators = 200, random_state = 0, max_features = 4)
        self.classifier = SVR(max_iter = 100)
        self.classifier.fit(trainX, trainY)
        # self.classifier.fit(trainX, trainY, weights)
        print('done RF training')
        # preds = self.classifier.predict(testX)
        # print(accuracy_score(preds, testY))
        del trainX
        del trainY

    def _forward(self, instance_embs, support_idx, query_idx):
        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,))).\
        squeeze(0).permute([1, 0, 2]) # [1, 1, 5, 512] -> [5, 1, 512]
        query   = instance_embs[query_idx.flatten()].view(  *(query_idx.shape   + (-1,))).\
        flatten(0, 2) # [1, 15, 5, 512] -> [75, 512]

        # print(support.shape)
        # print(query.shape)
        
        diffs = []
        sEmbeddings = torch.mean(support, dim = 1).squeeze()

        for i, qEmbedding in enumerate(query):
            for class_i in range(self.way):
                avgEmebdding = sEmbeddings[class_i]
                diff = (avgEmebdding - qEmbedding) ** 2
                diffs.append(diff)
        # diffs = np.stack(diffs).round(2).reshape(-1, 512)
        diffs = torch.stack(diffs).reshape(-1, 512).cpu().numpy()
        # print(diffs.shape)
        # exit()
        preds = self.classifier.predict(diffs)
        # preds = np.log(preds / ((1 - preds) + 1e-4))
        # preds *= 100
        # preds = preds.astype(int)
        preds = preds.reshape(-1, self.way)
        # preds = preds.flatten()
        return torch.from_numpy(preds).cuda()
    
    def _rf_classify(self, names):
        supportNames = names[:5*self.shot]
        batchQueryNames = names[5*self.shot:]
        diffs = []
        sEmbeddings = []
        for class_i in range(5):
            classEmbeddings = []
            for shot_i in range(self.shot):
                sName = supportNames[shot_i * 5 + class_i]
                sEmbedding = self.embeddingsTest[self.nameToIdxTest[sName]]
                classEmbeddings.append(sEmbedding)
            avgEmebdding = np.mean(classEmbeddings, axis = 0)
            sEmbeddings.append(avgEmebdding)

        # print(sEmbeddings[0])
        for qName in batchQueryNames:
            qEmbedding = self.embeddingsTest[self.nameToIdxTest[qName]]
            for class_i in range(self.way):
                avgEmebdding = sEmbeddings[class_i]
                diff = (avgEmebdding - qEmbedding) ** 2
                diffs.append(diff)  
        # diffs = np.stack(diffs).round(2).reshape(-1, 512)
        diffs = np.stack(diffs).reshape(-1, 512)
        preds = self.classifier.predict(diffs)
        # preds *= 100
        # preds = preds.astype(int)
        # preds = np.log(preds / ((1 - preds) + 1e-4))
        preds = preds.reshape(len(batchQueryNames), -1)
        
        # preds = preds.flatten()
        return torch.from_numpy(preds).cuda()