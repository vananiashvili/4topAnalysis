import DIClasses
import SampleHandler
import Utils
import ROOT
import numpy as np
import math
import PlotService
from scipy.stats import pearsonr
from scipy.stats import entropy
from keras import backend as K

class FNNAnalysis:

    def __init__(self,Model,DataSet):
        self.Model   = Model
        self.DataSet = DataSet

    def PhyNNCompare(self, layer,LCompareVar):
        """
        Comparesion of the neuron output in a given layer to variables that are build by using physics intuition
        layer:              Which Layer should be used for the comparision?
        LCompareVar:        List of Variables to which the neuron Output is compared
        """
        ListSamples = DIClasses.Init(LCompareVar,Cuts=True)
        Sampler = SampleHandler.SampleHandler(ListSamples)
        Sampler.norm    = False
        Sampler.SequenceLength  = 1
        Sampler.Plots = False
        CompareData = Sampler.GetANNInput(verbose=0)
        CompareTrain, CompareTest = CompareData.GetInput("Even")
        train, test = self.DataSet.GetInput("Even")

        input1 = self.Model.input                                           # input placeholder
        output1 = [l.output for l in self.Model.layers]                     # all layer outputs
        fun = K.function([input1, K.learning_phase()],output1)              # evaluation function
        LayerOutput = fun([train.Events, 1.])

        Layer = LayerOutput[layer]
        LCompareVar = CompareData.LVariables
        print(pearsonr(train.Events[:,0],CompareTrain.Events[:,0]))

        # for i in range(Layer.shape[1]):
        #     for j in range(CompareTrain.Events.shape[1]):
        #         print(pearsonr(Layer[:,i],CompareTrain.Events[:,j]))
        #         NeuronOutput = self.PrePlot(Layer[:,i], CompareTrain)
        #         CompareVar   = self.PrePlot(CompareTrain.Events[:,j], CompareTrain) 
                               
        #         PlotService.SigBkgHist(NeuronOutput,'Neuron ('+str(i)+') [a.u.]',40,0,1,tag='_Neuron_'+str(i))
        #         PlotService.SigBkgHist(CompareVar, LCompareVar[j], 40,0,1,tag='LCompareVar'+str(j))
        #         Hist2D = PlotService.SigBkg2D(CompareVar,NeuronOutput,'Neuron ('+str(i)+') [a.u.]',LCompareVar[j],20,20,0,1,0,1)
        #         Utils.stdinfo("The mutual information of Neuron {0} and {1} is : {2}".format(i,LCompareVar[j],self.GetNormedMi(Hist2D)))


    def PrePlot(self, Arr, DITrain):
        Events    = Utils.Transform(Arr,'MinMax')

        return DIClasses.DISample(Events, DITrain.Weights, DITrain.OutTrue,None,None)

    def GetNormedMi(self,Hist2D):
        """
        Compute the mutual information (normed to the entropy of X and Y) from a 2D histogramm
        """
        for i in range(30):
            col = np.array([])
            for j in range(30):
                col = np.append(col,Hist2D.GetBinContent(i,j))
            if(i == 0):
                HistArr = col
            else:
                HistArr = np.c_[HistArr,col]
        pxy = HistArr / float(np.sum(HistArr))                         #Convert to probability
        nztotal = pxy > 0
        px = np.sum(pxy, axis=1)                                       #marginal for x over y
        py = np.sum(pxy, axis=0)                                       #marginal for y over x
        pxpy = px[:, None] * py[None, :]
        nzcomb = pxpy > 0
        EntrX = entropy(px[px > 0])
        EntrY = entropy(py[py > 0])

        pxy = pxy[nztotal * nzcomb]
        pxpy = pxpy[nztotal * nzcomb]
    
        return 1/math.sqrt(EntrX* EntrY) * np.sum(pxy * np.log(pxy / pxpy))
