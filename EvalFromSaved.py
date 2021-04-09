import sys
sys.path.insert(1,'./srcRNN')
sys.path.insert(1,'./srcGeneral')

import SampleHandler
import DIClasses
import PlotService
import HighScoreResults
import numpy as np
from tensorflow import keras

Samples    = 'nomLoose'                     # 'nomLoose' All samples, else List of Samples needed
ModelName   = 'LSTM'

# Basic Sample informations
ListSamples = DIClasses.Init(ModelName,Samples,Cuts=True)
# Data Preparation
Mode = 'Fast'
if(Samples != 'nomLoose'):
    Mode = 'Save'
Sampler = SampleHandler.SampleHandler(ListSamples,mode=Mode+ModelName)
Sampler.norm    = False                                   # Y axis norm to one 
Sampler.valSize = 0.2                                     # Size of the validation sample
Sampler.Split   = 'EO'                                    # Use EO (even odd splitting)
Sampler.Scale   = 'ZScoreLSTM'                            # Kind of Norm use ZScoreLTSM
Sampler.SequenceLength  = 7                               # Length of the Sequence of a Bach (at the momement this controlls amount of jets)
Sampler.Plots = False                                     # Should the plots be done?

DataSet       = Sampler.GetANNInput()



class EvalRNN:

    def __init__(self, SavePath, ModelName, DataSet):
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
        self.ModelEven    = None
        self.ModelOdd     = None


    def PreTrainedRNN(self):
        """ Evaluation LSTM model """
        LOutPreOther = []                       # Predicted Output Scores on the sample used for training
        LOutPreSame  = []                       # Predicted Output Scores on the sample used for testing
        for Name in self.ModelNames:
            if('Even' in Name):
                self.ModelEven = self.ImportHFive(Name)
            elif('Odd' in Name):
                self.ModelOdd = self.ImportHFive(Name)
            OutPreOther, OutPreSame = self.GetOutPreFromModel(Name)
            LOutPreOther.append(OutPreOther)
            LOutPreSame.append(OutPreSame)
        AucOthers, AucSames = self.MakePlots(LOutPreOther,LOutPreSame)


        return


    def BootStrap(self):
         """ perform bootstrap on validation set """
         AucEven, AucOdd = np.array([]), np.array([])
         for Name in self.ModelNames:
              print(Name)
              Bootstrap = ('test',None)
              train, test, vali = GetSamples(Bootstrap,self.DataSet,Name)
              if('Even' in Name):
                   Model = self.ImportHFive(Name)
              elif('Odd' in Name):
                   Model = self.ImportHFive(Name)
              OutPreOther, OutPreSame = Model.predict(test.Events).ravel(), Model.predict(train.Events).ravel()
              Auctrain = roc_auc_score(train.OutTrue, OutPreSame, sample_weight=train.Weights)
              Auctest = roc_auc_score(test.OutTrue, OutPreOther, sample_weight=test.Weights)
              print("Auc on test, train: {0}, {1}".format(Auctest,Auctrain))
              for Seed in range(50):
                   Bootstrap = ('vali',Seed)
                   train, test, vali = GetSamples(Bootstrap,self.DataSet,Name)
                   OutPre = Model.predict(vali.Events).ravel()
                   if('Even' in Name):
                        AucEven = np.append(AucEven,roc_auc_score(vali.OutTrue, OutPre, sample_weight=vali.Weights))
                   elif('Odd' in Name):
                        AucOdd = np.append(AucOdd,roc_auc_score(vali.OutTrue, OutPre, sample_weight=vali.Weights))
         AucMean = (AucEven+AucOdd)/2
         print(np.mean(AucMean))

                                                                        

    def ImportHFive(self,ModelName):
        """ Import the given h5 model """
        print("Importing Model: "+ModelName)
        train, test = self.DataSet.GetInput("Even")
        model = keras.models.load_model("./"+ModelName+".h5")
        print("Import done")
        return model

    def MakePlots(self, OutPreOther, OutPreSame):
        """ Plotting the ROC, Score of the Model """
        AucOthers, AucSames = [], []
        PlotService.RocCurve("./",OutPreOther,self.DataSet,self.ModelNames)
        if(len(self.ModelNames) == 1):
            AucOther, AucSame = PlotService.Score("./",OutPreOther[0],OutPreSame[0],self.DataSet,self.ModelNames[0])
        elif(len(self.ModelNames) > 1):
            for i,Name in enumerate(self.ModelNames):
                AucOther, AucSame = PlotService.Score("./",OutPreOther[i],OutPreSame[i],self.DataSet,Name)
        AucOthers.append(AucOther)
        AucSames.append(AucSame)            

        return AucOthers, AucSames


    def GetOutPreFromModel(self,Name):
        train, test = self.DataSet.GetInput(Name)
        if("Even" in Name):
            OutPreOther, OutPreSame = self.ModelEven.predict(test.Events).ravel(), self.ModelEven.predict(train.Events).ravel()
        elif("Odd" in Name):
            OutPreOther, OutPreSame = self.ModelOdd.predict(test.Events).ravel(), self.ModelOdd.predict(train.Events).ravel()
        np.save(self.SavePath+Name+"_OutPreOther.npy",OutPreOther)   
        np.save(self.SavePath+Name+"_OutPreSame.npy",OutPreSame)
        return OutPreOther, OutPreSame


if(ModelName == 'LSTM'):
    ModelNames = ['LSTMEven_4096_0.8_1.0','LSTMOdd_4096_0.8_1.0']
if(ModelName == 'FNN19'):
    ModelNames = ['FNN19Even','FNN19Odd']

Eval = EvalRNN('./',ModelNames,DataSet)
Eval.PreTrainedRNN()
HNNRegion = HighScoreResults.SBOptimization(Eval.ModelEven,Eval.ModelOdd,ModelNames)
HNNRegion.Main()
