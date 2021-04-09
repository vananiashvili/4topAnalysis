import numpy as np
import ROOT
import Utils
import PlotService
import SampleHandler
import DIClasses
import math
import os
from sklearn.metrics import confusion_matrix, roc_auc_score



class EvalMulti:

    def __init__(self, SavePath, ModelName, DataSet, ModelEven, ModelOdd, RocEven, RocOdd):
        """ 
        SavePath  :  Path to the Model files
        ModelNames:  List of ModelNames
        DataSet   :  DataSet (train, test, validation)
        ModelEven :  The trained Model for the even Eventnumbers
        ModelOdd  :  The trained Model for the odd Eventnumbers
        """

        if(isinstance(ModelName,str)):
            ModelName = [ModelName]
        self.SavePath    = SavePath
        self.ModelNames  = ModelName
        self.DataSet     = DataSet
        self.ModelEven   = ModelEven
        self.ModelOdd    = ModelOdd
        self.RocEven     = RocEven
        self.RocOdd      = RocOdd

    def EvaluateFNN(self):
        LOutPreOther = []
        for Name in self.ModelNames:
            if("BDT" in Name):                                                                                  #Import the BDT Scores as comperison
                OutPreOther, OutPreSame = self.GetOutPreFromFile(Name)
            else:
                OutPreOther, OutPreSame = self.GetOutPreFromRoc(Name)
            PreLabelsOther, PreLabelsSame = self.PredictClasses(OutPreOther,OutPreSame)
            self.MakeConfusionMatrix(Name, PreLabelsOther, PreLabelsSame)
            PlotService.Score("./plots/",OutPreOther[:,0],OutPreSame[:,0],self.DataSet,Name)
            PlotService.MultiScore("./plots/",OutPreOther[:,0],OutPreSame[:,0],self.DataSet,Name,0)               #Score for Sig
            LOutPreOther.append(OutPreOther[:,0])
        PlotService.RocCurve("./plots/",LOutPreOther,self.DataSet,self.ModelNames)


    def PreTrainedFNN(self):
        LOutPreOther = []
        for Name in self.ModelNames:
            OutPreOther, OutPreSame = self.GetOutPreFromFile(Name)
            PreLabelsOther, PreLabelsSame = self.PredictClasses(OutPreOther,OutPreSame)
            #self.MakeConfusionMatrix(Name, PreLabelsOther, PreLabelsSame)
            Classnum = 13
            PlotService.MultiScore("./plots/",OutPreOther[:,Classnum],OutPreSame[:,Classnum],self.DataSet,Name,Classnum)               #Score for Sig
            PlotService.Score("./plots/",OutPreOther[:,Classnum],OutPreSame[:,Classnum],self.DataSet,Name)
            #Fit(OutPreOther)
            LOutPreOther.append(OutPreOther[:,0])
        PlotService.RocCurve("./plots/",LOutPreOther,self.DataSet,self.ModelNames)


    def GetOutPreFromFile(self, Name):
        OutPreOther = np.load(self.SavePath+Name+"_OutPreOther.npy")              
        OutPreSame  = np.load(self.SavePath+Name+"_OutPreSame.npy")          
        return OutPreOther, OutPreSame

    def GetOutPreFromModel(self,Name):
        train, test = self.DataSet.GetInput(Name)
        if("Even" in Name):
            OutPreOther, OutPreSame = self.ModelEven.predict(test.Events), self.ModelEven.predict(train.Events)
        elif("Odd" in Name):
            OutPreOther, OutPreSame = self.ModelOdd.predict(test.Events), self.ModelOdd.predict(train.Events)
        np.save(self.SavePath+Name+"_OutPreOther.npy",OutPreOther)   
        np.save(self.SavePath+Name+"_OutPreSame.npy",OutPreSame)
        return OutPreOther, OutPreSame


    def GetOutPreFromRoc(self,Name):
        train, test = self.DataSet.GetInput(Name)
        if("Even" in Name):
            OutPreOther, OutPreSame = self.RocEven.MaxPre[0], self.RocEven.MaxPre[1]
        elif("Odd" in Name):
            OutPreOther, OutPreSame = self.RocOdd.MaxPre[0], self.RocOdd.MaxPre[1]
        np.save(self.SavePath+Name+"_OutPreOther.npy",OutPreOther)   
        np.save(self.SavePath+Name+"_OutPreSame.npy",OutPreSame)

        Auctest  = roc_auc_score(test.OutTrue, OutPreOther[:,0], sample_weight=test.Weights)
        Auctrain = roc_auc_score(train.OutTrue, OutPreSame[:,0], sample_weight=train.Weights)

        print(Name)
        print("AUC train: "+str(Auctrain))
        print("AUC test: "+str(Auctest))

        return OutPreOther, OutPreSame


    def PredictClasses(self,OutPreOther,OutPreSame):
        """ Each event is a signed the class for which it has the highest score """
        PreLabelsOther, PreLabelsSame = np.array([]), np.array([])
        OutPreOther, OutPreSame = np.copy(OutPreOther), np.copy(OutPreSame)
        #Count = 0
        for i in range(len(OutPreOther)):
            # if(np.argmax(OutPreOther[i]) == 0 and OutPreOther[i][0] < 0.8):
            #     Count += 1
            #     OutPreOther[i][0] = 0.
            PreLabelsOther = np.append(PreLabelsOther,np.argmax(OutPreOther[i]))
        #print(Count)
        #Count = 0
        for i in range(len(OutPreSame)):
            # if(np.argmax(OutPreSame[i]) == 0 and OutPreSame[i][0] < 0.8):
            #     Count += 1
            #     OutPreSame[i][0] = 0.
            PreLabelsSame = np.append(PreLabelsSame,np.argmax(OutPreSame[i]))
        # print(Count)
        # print(len(OutPreSame))
        return PreLabelsOther, PreLabelsSame

    def MakeConfusionMatrix(self,Name,OutPreOther,OutPreSame):
        train, test = self.DataSet.GetInput(Name)
        cmOther, cmSame = confusion_matrix(test.MultiClass,OutPreOther), confusion_matrix(train.MultiClass,OutPreSame)
        cmOther = np.array(cmOther, dtype=np.float32)
        cmSame = np.array(cmSame, dtype=np.float32)
        for row in range(cmOther.shape[0]):
            if(np.sum(cmOther[row] != 0)):
                cmOther[row] = cmOther[row]/np.sum(cmOther[row])
        for row in range(cmSame.shape[0]):
            if(np.sum(cmSame[row] != 0)):
                cmSame[row] = cmSame[row]/np.sum(cmSame[row])

        if(cmOther.shape[0] == 14):
            Names = ['tttt', 'others', 'vjets', 'vv', 'tt(X)', 'tt others', 'tt light', 'tt HF', 'tt CO', 'tt Qmis', 'ttH', 'ttZ', 'ttWW', 'ttW']
        elif(cmOther.shape[0] == 3):
            Names = ['tttt','ttV','rest']
        elif(cmOther.shape[0] == 2):
            Names = ['tttt','Bkg']
        PlotService.ConfusionMatrix(cmOther,Names,len(Names),tag='Other')
        PlotService.ConfusionMatrix(cmSame,Names,len(Names),tag='Same')
        #print(np.sum(np.diag(cmOther)))

def Fit(OutPre):
    V = 'HT_all'
    Var = [V]
    Utils.stdinfo("Getting the variable from the root file")
    ListSamples = DIClasses.Init(Var,Cuts=True)     #HT_all
    Sampler = SampleHandler.SampleHandler(ListSamples)
    Sampler.norm    = False                                     #Y axis norm to one
    Sampler.valSize = 0.2
    Sampler.Split   = 'EO'
    Sampler.Scale   = None                                      #X axis scaling for NN
    Sampler.SequenceLength  = 0
    Sampler.Plots = False
    DataSet       = Sampler.GetANNInput(verbose=0)
    train, test = DataSet.GetInput("Even")
    vali = DataSet.vali
    PlotService.VarHist(DataSet,'NLO','./plots/','test', V, r"H_{T} [GeV]",500,2200,30,Norm=True)
    os.system("mv plots/"+V+".png plots/"+V+"_wo.png")
    OutPreSig = OutPre[:,0]
    cut, senSig = FindCut(OutPreSig,test.OutTrue,test.Weights)
    print(cut, senSig)         
    Events = test.Events[OutPreSig > cut]
    Weights = test.Weights[OutPreSig > cut]
    OutTrue = test.OutTrue[OutPreSig > cut]
    MultiClass = test.MultiClass[OutPreSig > cut]
    NewSample = DIClasses.DISample(Events,Weights,OutTrue,MultiClass,vali.LVariables,vali.Names)
    PlotService.VarHist(NewSample,'NLO','./plots/','test', V, r"H_{T} [GeV]",500,2200,30,Norm=True)  #H_{\text{T}}^{\text{all}} [GeV]



def FindCut(OutPre,OutTrue,Weights,InitSen=0):
    print("started cut search")

    sensitivity, cut = InitSen, 0
    for i in range(1,100):
        curCut = i/100.
        CutTrue    = OutTrue[OutPre > curCut]
        CutWeights = Weights[OutPre > curCut]
        Sig = np.sum(CutWeights[CutTrue == 1])
        Bkg = np.sum(CutWeights[CutTrue == 0])
    
        sen = Sig/math.sqrt(Sig+Bkg)
        if(sen > sensitivity):
            sensitivity = sen
            cut = curCut

    return cut, sensitivity

def PreCutScore(OutPreOther,OutPreSame,DataSet,Name):
    """ Plots a Score distribution after cutting on the signal Score distribution """
    cut      = 0.8
    ScoreNum = 11
    train, test   = DataSet.GetInput("Even")
    SigScoreOther = OutPreOther[:,0]                    #Choosing the signal Score
    SigScoreSame = OutPreSame[:,0]
    SecScoreOther = (OutPreOther[:,ScoreNum])[SigScoreOther > 0.8]
    SecScoreSame  = (OutPreSame[:,ScoreNum])[SigScoreSame > 0.8]
    test = DIClasses.DISample(None,test.Weights[SigScoreOther > 0.8],test.OutTrue[SigScoreOther > 0.8],test.MultiClass[SigScoreOther > 0.8])
    train = DIClasses.DISample(None,train.Weights[SigScoreSame > 0.8],train.OutTrue[SigScoreSame > 0.8],train.MultiClass[SigScoreSame > 0.8])
    DataSet = DIClasses.DIDataSet(train,test,None,None)


    PlotService.MultiScore("./plots/",SecScoreOther,SecScoreSame,DataSet,Name,ScoreNum)







    