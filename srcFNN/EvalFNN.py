import numpy as np
import os
from tensorflow.keras.models import load_model

import EvalTMVA
import EvalMulti
import PlotService
from PlotService import VarHist
from Utils import roc_auc_score



class EvalFNN:                                                                                                                        # Creates class EvalFNN

    def __init__(self, Type, SavePath, ModelName, DataSet, ModelEven, ModelOdd, RocEven, RocOdd):
       
        """ 
                Type      :  TMVA, Keras only or MultiClass
                SavePath  :  Path to the Model files
                ModelNames:  List of ModelNames
                DataSet   :  DataSet (train, test, validation)
                ModelEven :  The trained Model for the even Eventnumbers
                ModelOdd  :  The trained Model for the odd Eventnumbers
        """

        if isinstance(ModelName, str):                                                                                                # Checks if ModelName is a str
            ModelName = [ModelName]                                                                                                   # Turrns ModelName into a List with the string ModelName as the 1st entry

        self.Type        = Type
        self.SavePath    = SavePath
        self.ModelNames  = ModelName
        self.DataSet     = DataSet
        self.ModelEven   = ModelEven
        self.ModelOdd    = ModelOdd
        self.RocEven     = RocEven
        self.RocOdd      = RocOdd

    def EvaluateFNN(self):                                                                                                            # Define class method EvaluateFNN
        """ Evaluation of the FNN and the BDT model if given"""
        
        if self.Type == 'TMVA':                                                                                                       # Creates instance of class EvalTMVA defined in EvalTMVA.py ...
            Eval = EvalTMVA.EvalTMVA(self.SavePath,self.ModelNames,self.DataSet)                                                      # ... and feeds in the variables 
            Eval.EvaluateFNN()                                                                                                        # acts with the method EvaluateFNN from EvalTMVA on Eval
        
        elif self.Type == 'FNNMulti':                                                                                                 # Does the same for FNNMulti as for TMVA above
            Eval = EvalMulti.EvalMulti(self.SavePath,self.ModelNames,self.DataSet,self.ModelEven,self.ModelOdd,self.RocEven,self.RocOdd)
            Eval.EvaluateFNN()
        
        else:
            LOutPreOther = []                                                                                                         # Predicted Output Scores on the sample used for training
            LOutPreSame  = []                                                                                                         # Predicted Output Scores on the sample used for testing
            
            for Name in self.ModelNames:
                
                if "BDT" in Name:                                                                                                     # Import the BDT Scores as comperison ???
                    OutPreOther, OutPreSame = self.GetOutPreFromFile(Name)                                                            # Calls method GetOutPreFromFile from bellow, ...
                                                                                                                                      # ... which loads the np arrays into: OutPreOther, OutPreSame   
                else:
                    OutPreOther, OutPreSame = self.GetOutPreFromRoc(Name)                                                             # Calls method GetOutPreFromRoc from bellow
                   # Fit(OutPreOther,Name)
               
                LOutPreOther.append(OutPreOther)
                LOutPreSame.append(OutPreSame)   
            
            AucOthers, AucSames = self.BinaryPlots(LOutPreOther, LOutPreSame)

        return


    def PreTrainedFNN(self):
        
        if self.Type == 'FNNMulti':
            Eval = EvalMulti.EvalMulti(self.SavePath, self.ModelNames, self.DataSet, self.ModelEven, self.ModelOdd, self.RocEven, self.RocOdd)
            Eval.PreTrainedFNN()
        
        else:
            ROOT.TMVA.Tools.Instance()
            ROOT.TMVA.PyMethodBase.PyInitialize()
            LOutPreOther = []                       # Predicted Output Scores on the sample used for training
            LOutPreSame  = []                       # Predicted Output Scores on the sample used for testing  
            
            for Name in self.ModelNames:
                OutPreOther, OutPreSame = self.GetOutPreFromFile(Name)
                LOutPreOther.append(OutPreOther)
                LOutPreSame.append(OutPreSame)
                # Fit(OutPreOther,Name)
            AucOthers, AucSames = self.BinaryPlots(LOutPreOther, LOutPreSame)

        return


    def BinaryPlots(self, OutPreOther, OutPreSame):
        """ Plotting the ROC, Score of the Model """
        
        AucOthers, AucSames = [], []
        PlotService.RocCurve("./plots/", OutPreOther, self.DataSet, self.ModelNames)
        
        if len(self.ModelNames) == 1:
            AucOther, AucSame = PlotService.Score("./plots/", OutPreOther[0], OutPreSame[0], self.DataSet, self.ModelNames[0])
        
        elif len(self.ModelNames) > 1:
            
            for i,Name in enumerate(self.ModelNames):
                AucOther, AucSame = PlotService.Score("./plots/", OutPreOther[i], OutPreSame[i], self.DataSet, Name)
        AucOthers.append(AucOther)
        AucSames.append(AucSame)            

        return AucOthers, AucSames


    def GetOutPreFromFile(self, Name):
        
        OutPreOther = np.load(self.SavePath + Name + "_OutPreOther.npy")              
        OutPreSame  = np.load(self.SavePath + Name + "_OutPreSame.npy")          
        
        return OutPreOther, OutPreSame


    def GetOutPreFromRoc(self, Name):
        
        train, test = self.DataSet.GetInput(Name)
        
        if "Even" in Name:
            OutPreOther, OutPreSame = self.RocEven.MaxPre[0].ravel(), self.RocEven.MaxPre[1].ravel()
        
        elif "Odd" in Name :
            OutPreOther, OutPreSame = self.RocOdd.MaxPre[0].ravel(), self.RocOdd.MaxPre[1].ravel()
        
        np.save(self.SavePath + Name + "_OutPreOther.npy", OutPreOther)   
        np.save(self.SavePath + Name + "_OutPreSame.npy", OutPreSame)

        Auctest  = roc_auc_score(test.OutTrue, OutPreOther, sample_weight=test.Weights)
        Auctrain = roc_auc_score(train.OutTrue, OutPreSame, sample_weight=train.Weights)

        print(Name)
        print("AUC train: " + str(Auctrain))
        print("AUC test: " + str(Auctest))

        return OutPreOther, OutPreSame


    def GetOutPreFromModel(self, Name):
        
        train, test = self.DataSet.GetInput(Name)
        vali = self.DataSet.vali
        
        if "Even" in Name:
            OutPreOther, OutPreSame = self.ModelEven.predict(test.Events).ravel(), self.ModelEven.predict(train.Events).ravel()
            OutPreVali = self.ModelEven.predict(vali.Events).ravel()
        
        elif "Odd" in Name:
            OutPreOther, OutPreSame = self.ModelOdd.predict(test.Events).ravel(), self.ModelOdd.predict(train.Events).ravel()
            OutPreVali = self.ModelOdd.predict(vali.Events).ravel()
        
        np.save(self.SavePath+Name + "_OutPreOther.npy", OutPreOther)   
        np.save(self.SavePath+Name + "_OutPreSame.npy", OutPreSame)

        Auctest  = roc_auc_score(test.OutTrue, OutPreOther, sample_weight=test.Weights)
        Auctrain = roc_auc_score(train.OutTrue, OutPreSame, sample_weight=train.Weights)
        Aucvali  = roc_auc_score(vali.OutTrue, OutPreVali, sample_weight=vali.Weights)

        print(Name)
        print("AUC train: " + str(Auctrain))
        print("AUC test: " + str(Auctest))
        # print("AUC vali: "+str(Aucvali))
        # with open("SearchOut.txt","a+") as Out:
        #     Out.write(Name+"\n")
        #     Out.write("AUC train: "+str(Auctrain)+"\n")
        #     Out.write("AUC test: "+str(Auctest)+"\n")
        #     Out.close()
        return OutPreOther, OutPreSame
