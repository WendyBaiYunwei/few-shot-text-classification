import torch
from skimage import io
import random
import seq_tg as tg
import json
import torchvision.transforms as transforms
from random import choices
import numpy as np
import pickle

# understand dataset composition
# output torch.Size([64, 510])
    # torch.Size([64]) (binary label)

class Sampler():
    def __init__(self, difficulty_level, batch_size, shot_size, class_size, max_difficulty):
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        self.transform = transforms.Compose([transforms.ToTensor(),normalize])
        self.difficulty_level = difficulty_level
        self.metatrain_folders,metatest_folders = tg.mini_imagenet_folders()
        self.batch_size = batch_size
        self.shot_size = shot_size
        self.class_size = class_size
        self.max_difficulty = max_difficulty
        with open('data/embedding_new.pkl', 'rb') as f:
            self.embeddings =  pickle.load(f)
        with open('data/imgNameToIdx.json', 'r') as f:
            self.nameToIdx =  json.load(f)
        with open('data/idxToImgname.json', 'r') as f:
            self.idxToName =  json.load(f)
        with open('data/embedding_sim.pkl', 'rb') as f:
            self.sortedAdj = pickle.load(f)
        with open('data/class_sim.json') as f:
            self.classAdj = json.load(f)
        with open('data/class_sim_matrix.json') as f:
            self.class_matrix = json.load(f)
        with open('data/centroid_by_class.json') as f:
            self.getCentroidByClass = json.load(f)
        with open('data/label_map.json', 'r') as f: # compress label
            self.idxToLableName = json.load(f)
        # with open('nameToLableIdx.json', 'r') as f: # compress label
        #     self.nameToLableIdx = json.load(f)
        
    def getImgByRoot(self, root):
        query = self.transform(io.imread(root))
        return query

    def getAvgName(self, shots, classI):
        # l = len('n0279516900000371.jpg')
        embeddings = []
        for shot in shots:
            # shotName = shot[:-l]
            idx = self.nameToIdx[shot]
            embedding = self.embeddings[idx]
            embeddings.append(embedding)
        avgEmbedding = sum(embeddings) / len(embeddings)

        classStart = classI * 600
        classEmbeddings = self.embeddings[classStart:classStart + 600]
        bestDiff = float('inf')
        bestIdx = classStart
        for i, embedding in enumerate(classEmbeddings):
            diff = np.average((avgEmbedding - embedding) ** 2)
            if diff < bestDiff:
                bestDiff = diff
                bestIdx = i + classStart

        return self.idxToName[str(bestIdx)][0]


    def get_query_set(self, ss, labels, classIsChosen):
        query_set = []
        query_labels = [] # convert root to identifier
        mix = []
        for i in range(len(ss)):#0...10,
            if i % self.shot_size == 0:
                # s_root is the name of the picture that resembles the average embedding the most
                # get average embedding, loop through all embeddings and get the most similar one
                oneClassShots = ss[i:i + self.shot_size]
                averageName = self.getAvgName(oneClassShots, classIsChosen[i//self.shot_size])
                names = self.sortedAdj[averageName][self.difficulty_level : self.difficulty_level+self.batch_size] ##batch size
                # names = self.sortedAdj[ss[i]][self.difficulty_level : self.difficulty_level+self.batch_size] ##batch size
                
                for name in names:
                    name = name[1]
                    folder = name[:9] + '/'
                    name = './train/' + folder + name
                    query = self.getImgByRoot(name)
                    mix.append((query, labels[i], name))

        random.shuffle(mix)
        query_set = torch.stack([img_label[0] for img_label in mix])
        query_labels = torch.stack([img_label[1] for img_label in mix])
        names = [img_label[2] for img_label in mix]
        return query_set, query_labels, names

    def getClasses(self, classSize):
        idxes = [i for i in range(64)]
        newClassIs = random.sample(idxes, k = classSize)
        newClasses = [self.idxToLableName[idx] for idx in newClassIs]
        return newClassIs, newClasses

    # get support set images
    def getSupportSetInfo(self, classSize, supportSize):
        classIsChosen, classesChosen = self.getClasses(classSize)
        # classIsChosen, classesChosen = self.getClasses(classSize, difficulty_level=self.difficulty_level)
        ssNames = []
        ssImgs = []
        ssLabels = []

        for oneClass in classesChosen:
            imageName = self.getCentroidByClass[oneClass]
            
            support = random.sample(self.sortedAdj[imageName], k = 1)[0]
            supports = [support]
            query = self.sortedAdj[support[1]][self.difficulty_level]
            supports.extend(random.sample(self.sortedAdj[query[1]]\
                [self.difficulty_level+self.batch_size:], k = self.shot_size - 1))

            for support in supports:
                name = support[1]
                ssNames.append(name)
                folder = name[:9] + '/'
                name = './train/' + folder + name
                img = self.getImgByRoot(name)
                ssImgs.append(img)
                idx = self.idxToLableName.index(oneClass)
                label = torch.tensor(idx, dtype=torch.long)
                ssLabels.append(label)
        return torch.stack(ssImgs), torch.stack(ssLabels), ssNames, classIsChosen

    def getBatch(self):
        support_set, support_set_labels, ss_roots, classIsChosen = self.getSupportSetInfo(self.class_size,self.shot_size)
        query_set, query_set_label, query_names = self.get_query_set(ss_roots, support_set_labels, classIsChosen) # ss_roots: batch_size, class_num, shot_num

        qLabels = []
        
        for ql in query_set_label: # 30
            for i, ssLabel in enumerate(support_set_labels): # 10
                if ssLabel == ql:
                    qLabels.append(torch.tensor(i // self.shot_size, dtype=torch.long))
                    break

        qLabels = torch.stack(qLabels)
        # expandedQL = torch.zeros((qLabels.size(0), self.shot_size), dtype=torch.long)
        # for i in range(expandedQL.size(0)):
        #     expandedQL[i] = qLabels[i].repeat(1, self.shot_size) 
        # expandedQL = torch.reshape(expandedQL, (expandedQL.size(0) * expandedQL.size(1), )) #(30x5)x1

        # one_hot = torch.zeros((expandedQL.size(0), torch.max(expandedQL)+1), dtype=torch.long)
        # one_hot[torch.arange(expandedQL.size(0)), expandedQL] = torch.tensor(1, dtype=torch.long) #30 x 5 x 2
        # one_hot = torch.reshape(one_hot, (-1, self.shot_size*self.class_size))

        return support_set, query_set, qLabels, ss_roots, query_names#, one_hot 