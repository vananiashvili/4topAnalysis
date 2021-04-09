import PlotService
import SampleHandler
import numpy as np
from DIClasses import Init, DISample
from Utils import stderr, stdwar, stdinfo
from Utils import roc_auc_score
from array import array
from math import log, sqrt, isnan, log
from root_numpy.tmva import evaluate_reader
from tensorflow.keras.models import load_model
import ROOT

class EvalRNN:

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
        self.SavePath     = SavePath
        self.ModelNames   = ModelName
        self.DataSet      = DataSet
        self.ModelEven    = ModelEven
        self.ModelOdd     = ModelOdd
        self.RocEven      = RocEven
        self.RocOdd       = RocOdd

    def EvaluateRNN(self):
        """ Evaluation LSTM model """
        LOutPreOther = []                       # Predicted Output Scores on the sample used for training
        LOutPreSame  = []                       # Predicted Output Scores on the sample used for testing
        for Name in self.ModelNames:
            if("BDT" in Name):                                                                                  #Import the BDT Scores as comperison
                #OutPreOther, OutPreSame = self.GetOutPreFromFile(Name)
                pass
            else:
                OutPreOther, OutPreSame = self.GetOutPreFromRoc(Name)
            LOutPreOther.append(OutPreOther)
            LOutPreSame.append(OutPreSame)
        AucOthers, AucSames = self.MakePlots(LOutPreOther,LOutPreSame)

        return

    def PreTrainedRNN(self):
        """ Evaluation LSTM model """
        LOutPreOther = []                       # Predicted Output Scores on the sample used for training
        LOutPreSame  = []                       # Predicted Output Scores on the sample used for testing
        for Name in self.ModelNames:
            if("BDT" in Name):                                                                                  #Import the BDT Scores as comperison
                OutPreOther, OutPreSame = self.GetOutPreFromFile(Name)
            else:
                OutPreOther, OutPreSame = self.GetOutPreFromFile(Name)
            LOutPreOther.append(OutPreOther)
            LOutPreSame.append(OutPreSame)
        AucOthers, AucSames = self.MakePlots(LOutPreOther,LOutPreSame)


        return
                                                                        

    # def ImportHFive(self,ModelName):
    #     """ Import the given h5 model """
    #     train, test = self.DataSet.GetInput("Even")
    #     model = load_model("./ANNOutput/RNN/"+ModelName+".h5")
    #     return model, model.predict(test.Events).ravel(), model.predict(train.Events).ravel()

    def MakePlots(self, OutPreOther, OutPreSame):
        """ Plotting the ROC, Score of the Model """
        AucOthers, AucSames = [], []
        PlotService.RocCurve("./plots/",OutPreOther,self.DataSet,self.ModelNames)
        if(len(self.ModelNames) == 1):
            AucOther, AucSame = PlotService.Score("./plots/",OutPreOther[0],OutPreSame[0],self.DataSet,self.ModelNames[0])
        elif(len(self.ModelNames) > 1):
            for i,Name in enumerate(self.ModelNames):
                AucOther, AucSame = PlotService.Score("./plots/",OutPreOther[i],OutPreSame[i],self.DataSet,Name)
        AucOthers.append(AucOther)
        AucSames.append(AucSame)            

        return AucOthers, AucSames

    def GetOutPreFromFile(self, Name):
        OutPreOther = np.load(self.SavePath+Name+"_OutPreOther.npy")              
        OutPreSame  = np.load(self.SavePath+Name+"_OutPreSame.npy")        
        return OutPreOther, OutPreSame

    def GetOutPreFromModel(self,Name):
        train, test = self.DataSet.GetInput(Name)
        if("Even" in Name):
            OutPreOther, OutPreSame = self.ModelEven.predict(test.Events).ravel(), self.ModelEven.predict(train.Events).ravel()
        elif("Odd" in Name):
            OutPreOther, OutPreSame = self.ModelOdd.predict(test.Events).ravel(), self.ModelOdd.predict(train.Events).ravel()
        np.save(self.SavePath+Name+"_OutPreOther.npy",OutPreOther)   
        np.save(self.SavePath+Name+"_OutPreSame.npy",OutPreSame)
        return OutPreOther, OutPreSame

    def GetOutPreFromRoc(self,Name):
        train, test = self.DataSet.GetInput(Name)
        if("Even" in Name):
            OutPreOther, OutPreSame = self.RocEven.MaxPre[0].ravel(), self.RocEven.MaxPre[1].ravel()
        elif("Odd" in Name):
            OutPreOther, OutPreSame = self.RocOdd.MaxPre[0].ravel(), self.RocOdd.MaxPre[1].ravel()
        np.save(self.SavePath+Name+"_OutPreOther.npy",OutPreOther)
        np.save(self.SavePath+Name+"_OutPreSame.npy",OutPreSame)

        Auctest  = roc_auc_score(test.OutTrue, OutPreOther, sample_weight=test.Weights)
        Auctrain = roc_auc_score(train.OutTrue, OutPreSame, sample_weight=train.Weights)

        print(Name)
        print("AUC train: "+str(Auctrain))
        print("AUC test: "+str(Auctest))

        return OutPreOther, OutPreSame

