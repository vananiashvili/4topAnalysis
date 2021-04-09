import PlotService
import SampleHandler
import numpy as np
from DIClasses import Init, DISample
from Utils import stderr, stdwar, stdinfo
from sklearn.metrics import roc_auc_score
from PlotService import VarHist
from array import array
from math import log, sqrt
from root_numpy.tmva import evaluate_reader
from tensorflow.keras.models import load_model
import ROOT

class EvalTMVA:

    def __init__(self, SavePath, ModelName, DataSet):
        """ 
        SavePath  :  Path to the Model files
        ModelNames:  List of ModelNames
        DataSet   :  DataSet (train, test, validation)
        """

        if(isinstance(ModelName,str)):
            ModelName = [ModelName]
        self.SavePath   = SavePath
        self.ModelNames = ModelName
        self.DataSet    = DataSet                                                                                                     # Dataset is an instance of class DIDataSet

    def EvaluateFNN(self):                                                                                                            # Creates method used in EvalFNN.py, ...
        """ Evaluation the FNN and the BDT model if given"""
        LOutPreOther = []                                                                                                             # Predicted Output Scores on the sample used for training
        LOutPreSame  = []                                                                                                             # Predicted Output Scores on the sample used for testing
        
        for Name in self.ModelNames:
            train, test = self.DataSet.GetInput(Name)                                                                                 # Uses method GetInput of class DIDataSet which returns: train, test
            OutPreOther, OutPreSame = self.ImportXml(train,test,Name)                                                                 # Uses method ImportXml defined below
            np.savetxt(self.SavePath+Name+"_OutPreOther.txt",OutPreOther)                                                             # Saves Numpy array OutPreOther to a .txt file
            np.savetxt(self.SavePath+Name+"_OutPreSame.txt",OutPreSame)                                                               # Saves Numpy array OutPresame to a .txt file
            LOutPreOther.append(OutPreOther)                                                                                          # Appends the list LOutPreOther with OutPreOther                                                                                      
            LOutPreSame.append(OutPreSame)                                                                                            # Appends the list LOutPreSame with OutPreSame
        AucOthers, AucSames = self.MakePlots(LOutPreOther, LOutPreSame)                                                               # Uses the method MakePlots defined below, which returns, AucOthers, AucSames

        return

    def PreTrainedFNN(self):
        ROOT.TMVA.Tools.Instance()
        ROOT.TMVA.PyMethodBase.PyInitialize()
        LOutPreOther = []                       # Predicted Output Scores on the sample used for training
        LOutPreSame  = []                       # Predicted Output Scores on the sample used for testing  
        for Name in self.ModelNames:
            OutPreOther, OutPreSame = self.GetOutPreFromFile(Name)
            LOutPreOther.append(OutPreOther)
            LOutPreSame.append(OutPreSame)
            #Fit(OutPreOther)
        AucOthers, AucSames = self.MakePlots(LOutPreOther, LOutPreSame)

        return
        

    def ImportXml(self,train,test,ModelName):
        """ Import the given Xml model """
        reader = ROOT.TMVA.Reader()
        #Filling reader used for eval
        for Var in self.DataSet.LVariables:
            reader.AddVariable(Var,array('f', [0.]))
        xml = './dataset/weights/TMVAClassification_'+ModelName+'.weights.xml'                          
        reader.BookMVA(ModelName,xml)
        return evaluate_reader(reader,ModelName,test.Events), evaluate_reader(reader,ModelName,train.Events)                   # retuns the Classifcation Score

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
        OutPreOther = np.array([])
        PreFile = open(self.SavePath+Name+"_OutPreOther.txt")                
        for line in PreFile:
            OutPreOther = np.append(OutPreOther,float(line))
        OutPreSame = np.array([])
        PreFile = open(self.SavePath+Name+"_OutPreSame.txt")                
        for line in PreFile:
            OutPreSame = np.append(OutPreSame,float(line))

        return OutPreOther, OutPreSame


def Fit(OutPre):
    Var = ['met_met']
    stdinfo("Getting the variable from the root file")
    ListSamples = Init(Var,Cuts=True)     #HT_all
    Sampler = SampleHandler.SampleHandler(ListSamples)
    Sampler.norm    = False                                     #Y axis norm to one
    Sampler.valSize = 0.2
    Sampler.Split   = 'EO'
    Sampler.Scale   = 'None'                                    #X axis scaling for NN
    Sampler.SequenceLength  = 0
    Sampler.Plots = False
    DataSet       = Sampler.GetANNInput(verbose=0)
    train, test = DataSet.GetInput("Even")
    #VarHist(Sampler.ListAnaSetup,'./plots/','test', "HT_all", r"H_{\text{T}}^{\text{all}} [GeV]",500,2000,30,Norm=False)            
    cut  = 0.31                                                                     #TODO: Implement best cut search    
    PlotService.SBVar('./plots/',Var[0],test, r"H_{\text{T}}^{\text{all}} [GeV]",30,0,500,tag='_after_'+str(cut)+'_cut',Norm=False)
    Events = test.Events[OutPre > cut]
    Weights = test.Weights[OutPre > cut]
    OutTrue = test.OutTrue[OutPre > cut]
    NewSample = DISample(Events,Weights,None,OutTrue,None)
    PlotService.SBVar('./plots/',Var[0],NewSample, r"H_{\text{T}}^{\text{all}} [GeV]",30,0,500,tag='No_cut',Norm=False)
    assert 0 == 1   #TODO:Check plot
        
        



