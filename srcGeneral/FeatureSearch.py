import numpy as np
from itertools import combinations
from DIClasses import DIANNSetup, DIBDTSetup, Init, DISample
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import MultiTaskLassoCV
import SampleHandler
import NeuralNetwork
import OutputGrabber
import re


class SBS():                #Sequential Backward Selection
    def __init__(self,MaxDim,VarList,DIANNSetup):
        self.MaxDim   = MaxDim
        self.VarList  = VarList
        self.ANNSetup = ANNSetup

    def Main(self, train, test, vali):
        """
        self.MaxDim:    The dim to which the array should be reduced'
        self.indices:   the current indcies (of variables) that are
                        considered
        self.Removed:   Removed Variables from the VarList
        self.scores   :   List of the best scores achieved
        self.FinalScore: The score after reducing the array to self.MaxDim
        self.call   :   Amount of ANNs trained
        """

        dim = train.Events.shape[1]
        self.call = 0
        self.indices = tuple(range(dim))
        self.Removed = []
        self.scores = []
        print(self.VarList)

        while dim > self.MaxDim:
            scores   = []
            subsets = []

            for p in combinations(self.indices, r=dim-1):
                rm = tuple(set(self.indices) - set(p))[0]
                print(self.VarList[rm])
                score = self.CalcScore(train, test, vali, p)
                scores.append(score)
                subsets.append(p)

            print("The scores are: {}".format(scores))
            best = np.argmax(scores)
            rm = tuple(set(self.indices) - set(subsets[dim-best]))[0]
            self.Removed.append(self.VarList[rm])
            print("The removed element is: "+self.VarList[rm])
            print(scores[best])
            self.indices = subsets[best]
            print("new Indicies: {0}".format(self.indices))
            dim -= 1
            self.scores.append(scores[best])

        self.FinalScore = self.scores[-1]
        self.FinalList  = self.GetList(self.indices)

        return self

    def CalcScore(self, train, test, vali, indices):
        """ Training a ANN for the indices given and calculating the AUC """
        trainRed   = DISample(train.Events[:,indices],train.Weights,train.Events,train.OutTrue,train.DISetup)
        testRed    = DISample(test.Events[:,indices],test.Weights,test.Events,test.OutTrue,test.DISetup)
        RedVarList = self.GetList(indices)
        self.ANNSetup.InputDim  = len(RedVarList)
        self.ANNSetup.ModelName = 'PyKeras'+str(self.call)

        print(self.call)
        aucs = np.array([])
        for i in range(3):
            auc = ''
            flag = 0
            BDTSetup = DIBDTSetup(0,0,0,0,0,0)
            with OutputGrabber.stdout_redirected('FeatureSearch.txt'):
                NeuralNetwork.Main(self.ANNSetup, BDTSetup, testRed, trainRed, RedVarList)
            with open('FeatureSearch.txt',"r") as f:
                for line in f:
                    if flag == 1:
                        auc = line
                        break
                    if 'ROC-integ' in line:
                        flag = 1
            auc = float(re.findall("\d+\.\d+", auc)[0])
            print(auc)
            aucs = np.append(aucs,auc)

        self.call += 1
        return np.mean(aucs)

    def GetList(self, indices):
        RedVarList = []
        for i in indices:
            RedVarList.append(self.VarList[i])
        return RedVarList


def Lasso(train, test):                                     #Selecting features using Lasso (L1) regularisation
    print(len(train.Events[0]))
    sfm = SelectFromModel(MultiTaskLassoCV())
    sfm.fit(train.Events[:270010], test.Events)
    trainE = sfm.transform(train.Events[:270010])
    testE  = sfm.transform(test.Events)
    print(len(testE[0]))




# Basic Sample informations
ListSamples = Init('Five')
# Data Preparation
Sampler = SampleHandler.SampleHandler(ListSamples)
Sampler.norm = 'ZScore'
Sampler.valSize = 0.2
Sampler.Split = 'EO'
Sampler.Scale = False
train, test, vali = Sampler.GetANNInput()

# Lasso(train,test)
            

ANNSetup = DIANNSetup('80','test_model.h5','8192','PyKeras')
RedFeature = SBS(4,ListSamples[0].VarName,ANNSetup)
RedFeature.Main(train, test, vali)
print(RedFeature.Removed)
print(RedFeature.FinalScore)
print(RedFeature.FinalList)
print(RedFeature.scores)
print(RedFeature.call)

