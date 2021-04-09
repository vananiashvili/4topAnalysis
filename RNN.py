import sys
sys.path.insert(1,'./srcRNN')
sys.path.insert(1,'./srcGeneral')

import SampleHandler
import RecurrentNeuralNet
import EvalRNN
import numpy as np
import os
import DIClasses
import tensorflow as tf
import Utils

np.random.seed(15)
#Parameters
Samples    = 'nomLoose'                     # 'nomLoose' All samples, else List of Samples needed
ModelName   = 'LSTM'
EvalFlag    = True                           # Should the models be evaluated?
Odd         = False                            # Should the Odd combination be trained?
PreTrained  = False
SavePath    = './ANNOutput/RNN/' 

LayerType = 'LSTM'
Optimizer='Adam'
Epochs  = 1
Batch   = 2000
Neurons = [[1,1],[]]                 # Last Layer needs to be have 1 Neuron
Dropout = [0]
Regu = 0.001
Bootstrap = ('test',None)
Dropout.extend([0] * (len(Neurons[0])-len(Dropout)))
LearnRate  = DIClasses.DILrSchedule('poly',0.001,factor=2,StepSize=Epochs)
#LearnRate = DIClasses.DILrSchedule('normal',0.001)

print("LayerType: {}".format(LayerType))
print("Neurons: {}".format(Neurons))
print("Epochs: {0}".format(Epochs))
print("Batch: {0}".format(Batch))
print("Dropout: {0}".format(Dropout))
print("Regu: {0}".format(Regu))
LearnRate.Print()

# Basic Sample informations
ListSamples = DIClasses.Init(ModelName,Samples,Cuts=True)
# Data Preparation
GPU  = False                                                # Enable GPU training
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

if(GPU != True):
    tf.config.optimizer.set_jit(True)
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
DeviceTyp = tf.config.experimental.list_physical_devices()
if('GPU' in str(DeviceTyp)):
    Utils.stdwar("GPU training enabled!")
else:
    Utils.stdwar("CPU training!")






DataSet       = Sampler.GetANNInput()
#Neural Network Hyperparameters
ANNSetupEven = DIClasses.DIANNSetup(LayerType,Epochs,SavePath+ModelName+'Even.h5',Batch,ModelName+'Even',Neurons,Dropout=Dropout,LearnRate=LearnRate,Optimizer=Optimizer,Regu=Regu)
ANNSetupOdd  = DIClasses.DIANNSetup(LayerType,Epochs,SavePath+ModelName+'Odd.h5',Batch,ModelName+'Odd',Neurons,Dropout=Dropout,LearnRate=LearnRate,Optimizer=Optimizer,Regu=Regu)
ModelNames = [ANNSetupEven.ModelName,'BDTEven5','BDTEven19']

if(PreTrained == False):
    #Even: Training and Evaluation
    ModelEven, RocEven = RecurrentNeuralNet.Main(ANNSetupEven, DataSet, BootStrap=Bootstrap)
    ModelOdd, RocOdd = None, None

    #Odd: Training and Evaluation
    if(Odd == True):
        ModelOdd, RocOdd  = RecurrentNeuralNet.Main(ANNSetupOdd, DataSet, BootStrap=Bootstrap)                              
        ModelNames.append(ANNSetupOdd.ModelName)


    if(EvalFlag == True):
        Eval = EvalRNN.EvalRNN(SavePath,ModelNames,DataSet,ModelEven,ModelOdd,RocEven,RocOdd)
        Eval.EvaluateRNN()

elif(PreTrained == True):
        Eval = EvalRNN.EvalRNN(SavePath,ModelNames,DataSet,None,None,None,None)
        Eval.PreTrainedRNN()
        













