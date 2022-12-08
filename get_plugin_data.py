import numpy as np
import random
import pickle

random.seed(0)

def get_data(pos_embeddings, neg_embeddings):
    train_x = []
    train_y = []
    for i in range(len(pos_embeddings)):
        pos = pos_embeddings[i]['data']
        neg = neg_embeddings[i]['data']

        both = [pos, neg]
        # get same class embs
        for i, cur in enumerate(both):
            for same_class_i in range(1, 10): #18 +ve samples
                diff = (cur[0] - cur[same_class_i]) ** 2
                train_x.append(diff)
                train_y.append(1)

            other = both[1 - i]
            for j in range(1, 10): # 18 -ve samples
                diff = (cur[0] - other[j]) ** 2
                train_x.append(diff)
                train_y.append(0)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

data_dir = '../data/'

with open('../data/embedding_nlp_pos.pkl', 'rb') as f:
    pos_embeddings = pickle.load(f)

with open('../data/embedding_nlp_neg.pkl', 'rb') as f:
    neg_embeddings = pickle.load(f)


trainX, trainY = get_data(pos_embeddings, neg_embeddings)

with open(data_dir + 'rf_train_x_nlp.pkl', 'wb') as f: # embedding difference vector
    pickle.dump(trainX, f)

with open(data_dir + 'rf_train_y_nlp.pkl', 'wb') as f: 
    pickle.dump(trainY, f)

print('SimForest data successfully created.')
