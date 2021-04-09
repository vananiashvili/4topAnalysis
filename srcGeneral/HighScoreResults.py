import DIClasses
import Utils
import SampleHandler
import PlotService
import numpy as np
import math, os

class SBOptimization:

    def __init__(self, ModelEven, ModelOdd, Names, Var='HT_all'):
        self.ModelEven = ModelEven
        self.ModelOdd = ModelOdd
        self.ModelNames = Names
        self.Var = Var

    def Main(self):
        #importing the Variable to plot
        Samples    = 'nomLoose'
        Utils.stdinfo("Getting the variable from the root file")
        ListSamples = DIClasses.Init([self.Var],Samples,Cuts=True)     #HT_all
        Sampler = SampleHandler.SampleHandler(ListSamples)
        Sampler.norm    = False                                     #Y axis norm to one
        Sampler.valSize = 0.2
        Sampler.Split   = 'EO'
        Sampler.Scale   = None                                      #X axis scaling for NN
        Sampler.SequenceLength  = 0
        Sampler.Plots = False
        DataSet       = Sampler.GetANNInput(verbose=0)

        for Name in self.ModelNames:
            self.Name = Name
            # Plotting the variable with any NN Score cuts
            PlotService.VarHist(DataSet,'NLO','./plots/','All', self.Var, r"H_{T} [GeV]",500,2200,30,Norm=True)
            if("Even" in Name):
                os.system("mv plots/"+self.Var+".png plots/"+self.Var+"_wo_Even.png")
            elif("Odd" in Name):
                os.system("mv plots/"+self.Var+".png plots/"+self.Var+"_wo_Odd.png")

            #Optimizing a cut based on S/sqrt(S+B) and apllying it to the variable
            OutPre = MakePrediction(DataSet)
            cut, senSig = FindCut(OutPre,DataSet)
            print(cut, senSig)         
            test = ApllyCuts(test,OutPre[1],cut)
            train = ApllyCuts(train,OutPre[0],cut)
            vali = ApllyCuts(vali,OutPre[2],cut)
            NewDataSet = DIClasses.DIDataSet(test,train,vali)
            PlotService.VarHist(NewDataSet,'NLO','./plots/','All', self.Var, r"H_{T} [GeV]",500,2200,30,Norm=True)  #H_{\text{T}}^{\text{all}} [GeV]
            if("Even" in Name):
                os.system("mv plots/"+self.Var+".png plots/"+self.Var+"_Even.png")
            elif("Odd" in Name):
                os.system("mv plots/"+self.Var+".png plots/"+self.Var+"_Odd.png")


    def MakePrediction(self,DataSet):
        # Returns list of predictions as a list [train,test,vali]
        OutPre = []
        train, test = DataSet.GetInput(self.Name)
        vali = DataSet.vali
        OutPre.append(self.Model.predict(train.Events))
        OutPre.append(self.Model.predict(test.Events))
        OutPre.append(self.Model.predict(vali.Events))

        return OutPre


    def FindCut(self,OutPre,DataSet,InitSen=0):
        print("started cut search")
        Sample=DataSet.OneSample(self.Name)


        Sig = np.sum(Sample.Weights[Sample.OutTrue == 1])
        Bkg = np.sum(Sample.Weights[Sample.OutTrue == 0])
        print("Initial sens: "+str(Sig/math.sqrt(Sig+Bkg)))

        sensitivity, cut = InitSen, 0
        for i in range(1,100):
            curCut = i/100.
            CutTrue    = Sample.OutTrue[OutPre > curCut]
            CutWeights = Sample.Weights[OutPre > curCut]
            Sig = np.sum(CutWeights[CutTrue == 1])
            Bkg = np.sum(CutWeights[CutTrue == 0])
        
            sen = Sig/math.sqrt(Sig+Bkg)
            if(sen > sensitivity):
                sensitivity = sen
                cut = curCut

        return cut, sensitivity


def ApllyCuts(self,Sample,OutPre,cut):
        """ Apllies the found cut and returns a new DataSample """
        Events = Sample.Events[OutPre > cut]
        Weights = Sample.Weights[OutPre > cut]
        OutTrue = Sample.OutTrue[OutPre > cut]
        MultiClass = Sample.MultiClass[OutPre > cut]
        return DIClasses.DISample(Events,Weights,OutTrue,MultiClass,Sample.LVariables,Sample.Names)

