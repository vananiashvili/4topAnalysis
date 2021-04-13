import tensorflow as tf
import numpy as np
import os
import sys
sys.path.insert(1,'./srcFNN')
sys.path.insert(1,'./srcGeneral')

import SampleHandler                                                                                              # Niklas' Script in /srcGeneral
import DIClasses                                                                                                  # Niklas' Script in /srcGeneral
import Utils                                                                                                      # Niklas' Script in /srcGeneral
import FeedforwardNeuralNet as FNN                                                                                # Niklas' Script in /srcFNN
import EvalFNN                                                                                                    # Niklas' Script in /srcFNN



""" Analysis Setup """
np.random.seed(15)                                                                                                # Random seed used for splitting. Used to generate the same random value everytime.

# Parameters
ModelName  = 'FNN18'                                                                                              # Set of Variables
BDT        = None                                                                                                 # BDT model Name. If None, no BDT is trained (TMVA only)
EvalFlag   = True                                                                                                 # Should the models be evaluated?
Odd        = False                                                                                                # Should the Odd combination be trained?
PreTrained = False                                                                                                # Was the Model already evaluated before?



""" Neural Network Parameters """
Type       = 'FNN'                                                                                                # FNN, TMVA, FNNMulti
Opti       = 'Adam'                                                                                               # Adam, Rmsprop, SGD, Adagrad
Winit      = 'GlorotNormal'                                                                                       # Lecun, Glorot, He (Normal or Uniform)
activation = 'elu'                                                                                                # if number => alpha leaky relu elif selu, relu, elu etc.

if activation.replace('.','',1).isdigit():                                                                        # string.replace('a','b',#): Replace 'a' in string with 'b' #-times.
    activation = float(activation)

Epochs     = 80
Batch      = 50000
# Neurons    = [int(num) for num in sys.argv[1].split(',')]
Neurons    = [35, 35, 35, 1]                                                                                      # Last Neuron should be 1 for FNN, 2 for TMVA, or # of classes for FNNMulti
Dropout    = None                                                                                                 # No dropout -> None
Regu       = 0                                                                                                    # l2
Bootstrap  = ('test', None)                                                                                       # (Sample, Random seed for Bootstrap)



"""" Learning rate schedule """                                                                                   # Create instance of class DILrSchedule defined in DIClasses.py

# LearnRate = DIClasses.DILrSchedule('poly',0.004,factor=2,StepSize=Epochs)
# LearnRate = DIClasses.DILrSchedule('drop',0.005,factor=0.25,StepSize=10.)
# LearnRate = DIClasses.DILrSchedule('cycle',0.005,cycle='triangular',MinLr=0.001,StepSize=400.,factor=0.9)

LearnRate = DIClasses.DILrSchedule('normal', 0.001)                                                               # DILrSchedule(mode, Lr, factor=1., cycle='triangular', MinLr=0.006, StepSize=0)
                                                                                                                  #    modes = {cycle, poly, drop, normal}
                                                                                                                  #    Lr = Initial learnrate or max learnrate (cycle)
                                                                                                                  #    cycle = {triangular, triangular2, exp_range(factor=gamma)}


""" Display the properties of NN before running """

Utils.stdinfo("Neurons: {}".format(Neurons))                                                                      # Uses function stdinfo from Utils.py
Utils.stdinfo("Epochs: {0}".format(Epochs))                                                                       #    def stdinfo(message):
Utils.stdinfo("Batch: {0}".format(Batch))                                                                         #        print(OKBLUE + message + ENDC)   OKGREEN = '\033[92m'   ENDC = '\033[0m'
Utils.stdinfo("Dropout: {0}".format(Dropout))                                                                     # str.format(value) Returns a string with placeholder {} replaced by value.
Utils.stdinfo("Regu: {0}".format(Regu))
LearnRate.Print()


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#                                                                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

Samples = 'nomLoose'                                                                                              # 'nomLoose' All samples, else List of Samples needed

ListSamples = DIClasses.Init(ModelName, Samples, Cuts=True)                                                       # Feeds the vairables to function Init defined in DIClasses.py

GPU  = False                                                                                                      # Enables GPU training
Mode = 'Slow'                                                                                                     # Fast, Slow or Save

if Samples != 'nomLoose':                                                                                         # Fast and Save modes only available for nomLoose
    Mode = 'Slow'


Sampler = SampleHandler.SampleHandler(ListSamples, mode=Mode+ModelName)                                           # Creates instance of class SampleHandler defined in SamplerHandler.py

Sampler.NormFlag = False                                                                                          # y axis to one
Sampler.valSize  = 0.2                                                                                            # Size of the validation sample
Sampler.Split    = 'EO'                                                                                           # Use EO (even odd splitting)
Sampler.Plots    = False                                                                                          # either False, 'LO' or 'NLO's


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#                                                                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

if GPU != True:
    tf.config.optimizer.set_jit(True)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DeviceTyp = tf.config.experimental.list_physical_devices()

if 'GPU' in str(DeviceTyp):
    Utils.stdwar("GPU training enabled!")                                                                         # Using function stdwar defined in Utils.py
else:                                                                                                             #    def stdwar(message):
    Utils.stdwar("CPU training!")                                                                                 #       print(WARNING + message + ENDC)    WARNING = '\033[93m'   ENDC = '\033[0m'


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#                                                                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

DataSet  = Sampler.GetANNInput()
SavePath = './ANNOutput/FNN/'                                                          # NN output save path


"""BDT Network Hyperparameters"""
BDTSetupEven = DIClasses.DIBDTSetup('BDTEven', 800, 3, 5, 2, 30, 8)
BDTSetupOdd  = DIClasses.DIBDTSetup('BDTOdd',  800, 3, 5, 2, 30, 8)

# Hand optiomalization
ModelNames = []


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#                                                                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

"""Neural Network Hyperparameters"""
ANNSetupEven = DIClasses.DIANNSetup(Type, Epochs, SavePath+ModelName+'Even.h5', Batch, ModelName+'Even',
                                    LearnRate=LearnRate, Neurons=Neurons, Dropout=Dropout, Regu=Regu,
                                    Optimizer=Opti, Winit=Winit, Activ=activation)

ANNSetupOdd  = DIClasses.DIANNSetup(Type, Epochs, SavePath+ModelName+'Odd.h5', Batch, ModelName+'Odd',
                                    LearnRate=LearnRate, Neurons=Neurons, Dropout=Dropout, Regu=Regu,
                                    Optimizer=Opti, Winit=Winit, Activ=activation)


if PreTrained == False:

    if BDT != None:
        ModelNames.append(BDTSetupEven.ModelName)
        ModelNames.append(ANNSetupEven.ModelName)
        ModelEven, RocEven = FNN.Main(ANNSetupEven, DataSet, BDTSetup=BDTSetupEven, BootStrap=Bootstrap)
        ModelOdd, RocOdd = None, None

        if Odd == True:
            ModelNames.append(BDTSetupOdd.ModelName)
            ModelNames.append(ANNSetupOdd.ModelName)
            ModelOdd, RocOdd = FNN.Main(ANNSetupOdd, DataSet, BDTSetup=BDTSetupOdd, BootStrap=Bootstrap)

    elif BDT == None:
        ModelNames.append(ANNSetupEven.ModelName)
        ModelEven, RocEven = FNN.Main(ANNSetupEven, DataSet, BootStrap=Bootstrap)
        ModelOdd, RocOdd = None, None

        if Odd == True:
            ModelNames.append(ANNSetupOdd.ModelName)
            ModelOdd, RocOdd = FNN.Main(ANNSetupOdd, DataSet, BootStrap=Bootstrap)

    if EvalFlag == True:
        Eval = EvalFNN.EvalFNN(Type, SavePath, ModelNames, DataSet, ModelEven, ModelOdd, RocEven, RocOdd)
        Eval.EvaluateFNN()

    if Type == 'TMVA':
        os.system("mv ./dataset/weights/*.xml ./ANNOutput/FNN/")
        os.system("mv ./dataset/weights/*.h5 ./ANNOutput/FNN/")

else:

    if BDT != None:
        ModelNames.append(BDTSetupEven.ModelName)
        ModelNames.append(ANNSetupEven.ModelName)

        if Odd == True:
            ModelNames.append(BDTSetupOdd.ModelName)
            ModelNames.append(ANNSetupOdd.ModelName)

    elif BDT == None:
        ModelNames.append(ANNSetupEven.ModelName)

        if Odd == True:
            ModelNames.append(ANNSetupOdd.ModelName)

    if EvalFlag == True:
        Eval = EvalFNN.EvalFNN(Type, SavePath, ModelNames, DataSet, None, None, None, None)
        Eval.PreTrainedFNN()