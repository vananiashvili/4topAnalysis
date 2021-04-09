import Utils    
import numpy as np
from math import sqrt

class Trafo:

    def __init__(self,TrafoFlag):
        self.TrafoFlag = TrafoFlag
        self.Means = []
        self.Stds = []
    
    def DefaultTrafo(self,Events,Sampletype):
        if(self.TrafoFlag == 'MinMax'):
            Utils.stdwar("Min not implemented for LSTM")
            for i in range(len(Events[0])):
                min, max = self.GetMinMax(Events,i)                 #TODO: use training min max for test
                for j in range(len(Events)):
                    Events[j][i] = float(Events[j][i])
                    Events[j][i] = (Events[j][i] - min)/(max - min) - 0.5



        elif(self.TrafoFlag == 'ZScoreLSTM'):
            for iVar in range(Events.shape[1]):                                                  #Loop over Variable
                if(Sampletype == 'train'):                                                #Use same var and mean for all sets
                    Var = np.ma.masked_equal(Events[:,iVar],-3.4*pow(10,38))
                    self.Means.append(np.mean(Var.flatten()))
                    self.Stds.append(np.var(Var.flatten()))
                for iSeq in range(Events.shape[2]):                                           #Loop over Sequence
                        for Event in range(len(Events)):                                                #Loop over Batch
                            if(Events[Event][iVar][iSeq] != -3.4*pow(10,38)):
                                Events[Event][iVar][iSeq] = float(Events[Event][iVar][iSeq])
                                Events[Event][iVar][iSeq] = (Events[Event][iVar][iSeq] - self.Means[iVar])/sqrt(self.Stds[iVar])


        elif(self.TrafoFlag == 'ZScore'):
            for iVar in range(Events.shape[1]):                                                  #Loop over Variable
                if(Sampletype == 'train'):
                    mean, variance = np.mean(Events[:,iVar]), np.var(Events[:,iVar])
                    self.Means.append(mean)
                    self.Stds.append(variance)
                else:
                    mean = self.Means[iVar]
                    variance = self.Stds[iVar]
                if(variance == 0 and mean !=0):
                    Utils.stdwar("All Entries in this variable are the same and not equal 0. Skipping!")
                    Utils.stdwar("Variableidx {0}".format(iVar))
                elif(variance != 0):
                    for iBatch in range(len(Events)):                                                #Loop over Batch
                        Events[iBatch][iVar] = float(Events[iBatch][iVar])
                        Events[iBatch][iVar] = (Events[iBatch][iVar] - mean)/sqrt(variance)
            
        elif(self.TrafoFlag != None):
            Utils.stdwar("This norm is not in implemented!")
            assert 0 == 1

        if(np.ndim(Events) == 3):                                                    # events, t, Var (the ordering was incorrect up until now)
            Events = np.swapaxes(Events,1,2)
        Events = np.where(Events != -3.4*pow(10,38), Events, 0)                      # replacing all padded element with 0

        return Events



    def GetMinMax(self,Arr,col):
        return Arr.min(0)[col], Arr.max(0)[col]













# def Transform(Arr,kind):
    
#     if(kind == 'ZScore'):  
#         print("enter")           
#         scaler = StandardScaler(with_mean=True, with_std=True)
#         if(np.ndim(Arr) == 1):
#             Arr = Arr.reshape(-1,1)
#         scaler.fit(Arr)
#         Arr = scaler.transform(Arr)
#         Arr = Arr.ravel()
        
#     if(kind == 'MinMax'):
#         scaler = MinMaxScaler()
#         if(np.ndim(Arr) == 1):
#             Arr = Arr.reshape(-1,1)
#         scaler.fit(Arr)
#         Arr = scaler.transform(Arr)
#         Arr = Arr.ravel()

#     return Arr