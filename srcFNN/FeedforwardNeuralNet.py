import numpy as np
import ROOT
import time                                                                                                       # Used [201, 210]
from root_numpy.tmva import add_classification_events                                                             # Used [103, 104]
from sklearn.utils.class_weight import compute_class_weight

import PlotService
from Guard import GuardFNN
from Callbacks import *
from Utils import *

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential                                                                    # Used [114, 174]
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2, l1



NUM_THREADS = 1
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#                                                                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

def Main(ANNSetup, DataSet, BDTSetup=None, BootStrap=('vali', None)):
    
    np.random.seed(5)                                                                                             # Fix Seeds to ensure that the random weight init stays the same for each training                                   
    tf.compat.v1.set_random_seed(5)
    
    GuardFNN(ANNSetup)                                                                                            # Ensure that the NN are not to large (just some conventions for the baf)


    # Training using TMVA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    if ANNSetup.Architecture == 'TMVA':
        
        train, test, vali = GetSamples(BootStrap, DataSet, ANNSetup.ModelName, DoTrafo=False)                     # Defined in Utils
        dataloader, factory, output = Init(train, test, DataSet.LVariables)                                       # Defined below
        TMVAFNN(ANNSetup, dataloader, factory)                                                                    # Defined below
        
        if BDTSetup != None:
            BDT(BDTSetup, dataloader, factory)                                                                    # Defined below
                                                    # GetRocs(factory, dataloader,"BDTEven")
        
        Finialize(factory, output)                                                                                # Defined below
                                                    # GetRocs(factory, dataloader,"FNN19Even")


    # Training using FNN  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # Direct Keras implementation
    elif ANNSetup.Architecture == 'FNN':
        
        train, test, vali = GetSamples(BootStrap, DataSet, ANNSetup.ModelName, DoTrafo=True)
        ANNSetup.InputDim = len(DataSet.LVariables)                                                               # Sets the N-dim of the 1st layer of FNN
        
        if BDTSetup != None:
            stdwar("BDT is only supported using TMVA")                                                            # Defined in Utils
        
        return FNN(ANNSetup, test, train)                                                                         # Defined below                                                                 


    # Training using FNN Multi Classifier - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # Direct Keras implementation
    elif ANNSetup.Architecture == 'FNNMulti':
        train, test, vali = GetSamples(BootStrap, DataSet, ANNSetup.ModelName, DoTrafo=True)                      # Defined in Utils
        ANNSetup.InputDim = len(DataSet.LVariables)
        
        if BDTSetup != None:
            stdwar("BDT is only supported using TMVA")                                                            # Defined in Utils
        
        Model, Roc = MultiFNN(ANNSetup, test, train)                                                              # Defined Below
        
        if BootStrap[1] != None:
            stdwar("BootStrap for multi has to be implemented")
            assert 0 == 1
        
        return Model, Roc



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   T M V A                                                                                                       #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

def Init(train, test, VarList):                                                                                   # Used in Main above[46]
    # Setup TMVA
    ROOT.TMVA.Tools.Instance()
    ROOT.TMVA.PyMethodBase.PyInitialize()

    output = ROOT.TFile.Open('~/Data/NNOutput.root', 'RECREATE')
    factory = ROOT.TMVA.Factory('TMVAClassification', output,'!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')
    dataloader = ROOT.TMVA.DataLoader('dataset')

    for Var in VarList:
        dataloader.AddVariable(Var)

    add_classification_events(dataloader, train.Events, train.OutTrue, weights=train.Weights, signal_label=1)     # from root_numpy.tmva
    add_classification_events(dataloader, test.Events, test.OutTrue, weights=test.Weights, signal_label=1, test=True)

    dataloader.PrepareTrainingAndTestTree(ROOT.TCut(''), 'SplitSeed=100')   # :NormMode=None
    #CrossCheck(dataloader)

    return dataloader, factory , output


def TMVAFNN(ANNSetup, dataloader, factory):                                                                       # Used in Main above[47]
    
    model = Sequential()                                                                                          # from tensorflow.keras.models
    model.add(Dense(ANNSetup.Neurons[0], activation='selu', input_dim=ANNSetup.InputDim))
    
    if ANNSetup.Dropout != None:
        model.add(Dropout(ANNSetup.Dropout))
    
    for i in range(1,len(ANNSetup.Neurons)):
        
        if i == len(ANNSetup.Neurons)-1:
            model.add(Dense(ANNSetup.Neurons[i], activation='sigmoid'))
        
        else:
            model.add(Dense(ANNSetup.Neurons[i], activation='selu'))

    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate.Lr)                                                      # Defined below
    
    model.compile(optimizer=Opti, loss='binary_crossentropy', metrics=['accuracy'])                               # Compiling the model

    model.save(ANNSetup.SavePath)
    # model.summary()
    
    factory.BookMethod(dataloader, ROOT.TMVA.Types.kPyKeras, ANNSetup.ModelName,
                       '!H:!V:FilenameModel=' + ANNSetup.SavePath + ':NumEpochs='
                       + ANNSetup.Epochs + ':BatchSize=' + ANNSetup.Batch + ":VarTransform=G")

    return


def BDT(BDTSetup, dataloader, factory):                                                                           # Used in Main above[50] 

    factory.BookMethod(dataloader, ROOT.TMVA.Types.kBDT, BDTSetup.ModelName,
                       "!H:!V:NTrees=" + BDTSetup.TreeNumber + ":MinNodeSize="
                       + BDTSetup.NMinActual + "%:BoostType=Grad:Shrinkage=" + BDTSetup.Shrinkage
                       + ":UseBaggedBoost:BaggedSampleFraction=" + BDTSetup.BaggingActual
                       + ":nCuts=" + BDTSetup.NCutActual + ":MaxDepth=" + BDTSetup.MaxDepth
                       + ":IgnoreNegWeightsInTraining=True" )

    return

def Finialize(factory, output):                                                                                   # Used in Main above[53]

    factory.TrainAllMethods()
    factory.TestAllMethods()
    factory.EvaluateAllMethods()
    output.Close()

    return


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   F N N                                                                                                         #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

def FNN(ANNSetup, test, train):
    
    # ClassWeights = GetClassWeights(train.OutTrue, train.Weights)
    TrainWeights = GetTrainWeights(train.OutTrue, train.Weights)                                                  # Defined below           

    # tf.debugging.set_log_device_placement(True)                                                                 # Check if system is running on the correct device
    
    model = Sequential()                                                                                          # from tensorflow.keras.models                                                                                          #
    model.X_train = train.Events
    model.Y_train = train.OutTrue
    model.W_train = train.Weights                                                                                 # Original weights!
    model.X_test  = test.Events
    model.Y_test  = test.OutTrue
    model.W_test  = test.Weights

    model = BuildModel(ANNSetup, model)                                                                           # Defined below

    Opti = GetOpti(ANNSetup.Optimizer, ANNSetup.LearnRate.Lr)                                                     # Defined below
    
    lrate = GetLearnRate(ANNSetup.LearnRate, ANNSetup.Epochs)                                                     # Defined below
    
    Roc = Histories()                                                                                             # Defined in Callbacks.py
    # Roc = RedHistory()
    
    if lrate == None:
        Lcallbacks = [Roc]
        # Lcallbacks = []
    else:
        Lcallbacks = [Roc, lrate]
        # Lcallbacks = [lrate]

    model.summary()
    model.compile(optimizer=Opti, loss='binary_crossentropy', metrics=['accuracy'])
    
    start = time.clock()                                                                                          # from time
    
    history = model.fit(train.Events, train.OutTrue, sample_weight=TrainWeights,
                        validation_data=(test.Events, test.OutTrue, test.Weights),
                        epochs=int(ANNSetup.Epochs), batch_size=int(ANNSetup.Batch),
                        verbose=2, callbacks=Lcallbacks)   # sample_weight=TrainWeights
    
    # history = model.fit(train.Events, train.OutTrue, batch_size=4000, epochs=2)
    
    end = time.clock()                                                                                            # from time
    
    print("The training took {} seconds".format(end-start))

    LAuc = Roc.TestAucs
    LTrainAuc = Roc.TrainAucs
    
    print("Best Test Auc {0:.4f} at Epoch {1}".format(max(LAuc), (LAuc.index(max(LAuc))+1)))   # 0:.4f
    print("Best Train Auc {0:.4f}".format(LTrainAuc[LAuc.index(max(LAuc))]))

    for i in range(len(LAuc)):
        print("Auc at Epoch {0}: {1:.4f} Ov: {2:.3f}".format(i,LAuc[i],1-LAuc[i]/LTrainAuc[i]))

    model.save(ANNSetup.SavePath)

    return model, Roc


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   M u l t i   F N N                                                                                             #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

def MultiFNN(ANNSetup, test, train):
    TrainMultiClass   = to_categorical(train.MultiClass)
    TestMultiClass   = to_categorical(test.MultiClass)
    #ClassWeights = GetClassWeights(train.MultiClass,train.Weights)
    TrainWeights = GetTrainWeights(train.MultiClass,train.Weights)

    model = Sequential()
    model.Y_test  = TestMultiClass[:,0]
    model.X_train = train.Events
    model.Y_train = TrainMultiClass[:,0]
    model.W_train = train.Weights       #Original weights!

    model.add(Dense(ANNSetup.Neurons[0], activation='selu', input_dim=ANNSetup.InputDim))
    if(ANNSetup.Dropout != None):
        model.add(Dropout(ANNSetup.Dropout))
    for i in range(1,len(ANNSetup.Neurons)):
        if(i == len(ANNSetup.Neurons)-1):
            model.add(Dense(ANNSetup.Neurons[i], activation='softmax'))
        else:
            model.add(Dense(ANNSetup.Neurons[i], activation='selu'))

    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate.Lr)
    lrate = GetLearnRate(ANNSetup.LearnRate,ANNSetup.Epochs)
    Roc = Histories()
    Lcallbacks = [Roc,lrate]

    model.compile(optimizer=Opti, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train.Events, TrainMultiClass, sample_weight=TrainWeights, validation_data=(test.Events, TestMultiClass, test.Weights), epochs=int(ANNSetup.Epochs),
                        batch_size=int(ANNSetup.Batch), verbose=2, callbacks=Lcallbacks)            #, sample_weight=TrainWeights

    LAuc = Roc.TestAucs
    LTrainAuc = Roc.TrainAucs
    print("Best Roc {0:.4f} at Epoch {1}".format(max(LAuc),LAuc.index(max(LAuc))+1))
    print("Train Auc {0:.4f}".format(LTrainAuc[LAuc.index(max(LAuc))]))
    # print("Test Rocs: {0}".format(LAuc))
    # print("Test Loss: {0}".format(Roc.TestLosses))
    # print("Train Rocs: {0}".format(LTrainAuc))
    # print("Train Loss: {0}".format(Roc.TrainLosses))

    model.save(ANNSetup.SavePath)

    return model, Roc


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   L e v e l - 2                                                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

def GetOpti(Optimizer,LearnRate):                                                                                 # Used above in: TMVAFNN[128], FNN[184]
    if(Optimizer == 'SGD'):
        Opti = optimizers.SGD(lr=LearnRate, momentum=0.0, nesterov=False)
    elif(Optimizer == 'Rmsprop'):
        Opti = optimizers.RMSprop(lr=LearnRate, rho=0.9)
    elif(Optimizer == 'Adagrad'):
        Opti = optimizers.Adagrad(lr=LearnRate)
    elif(Optimizer == 'Adam'):
        Opti = optimizers.Adam(lr=LearnRate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    return Opti


def BuildModel(ANNSetup, model):                                                                                  # Used in FNN above[182]
    
    if isinstance(ANNSetup.Activ, str):
        model.add(Dense(ANNSetup.Neurons[0], kernel_regularizer=l2(ANNSetup.Regu), activation=ANNSetup.Activ, kernel_initializer=Winit(ANNSetup.Winit), input_dim=ANNSetup.InputDim))
        
        if ANNSetup.Dropout != None:
            model.add(Dropout(ANNSetup.Dropout))
        
        for i in range(1,len(ANNSetup.Neurons)):
            
            if i == len(ANNSetup.Neurons)-1:
                model.add(Dense(ANNSetup.Neurons[i], kernel_initializer=Winit(ANNSetup.Winit), activation='sigmoid'))
           
            else:
                model.add(Dense(ANNSetup.Neurons[i], kernel_initializer=Winit(ANNSetup.Winit), activation=ANNSetup.Activ))
   
    else:
        model.add(Dense(ANNSetup.Neurons[0], kernel_regularizer=l2(ANNSetup.Regu), kernel_initializer=Winit(ANNSetup.Winit), input_dim=ANNSetup.InputDim))
        model.add(LeakyReLU(alpha=ANNSetup.Activ))
       
        if ANNSetup.Dropout != None:
            model.add(Dropout(ANNSetup.Dropout))
        
        for i in range(1,len(ANNSetup.Neurons)):
            
            if i == len(ANNSetup.Neurons)-1:
                model.add(Dense(ANNSetup.Neurons[i], kernel_initializer=Winit(ANNSetup.Winit), activation='sigmoid'))
            
            else:
                model.add(Dense(ANNSetup.Neurons[i], kernel_initializer=Winit(ANNSetup.Winit)))
                model.add(LeakyReLU(alpha=ANNSetup.Activ))

    return model


def GetLearnRate(DILr,Epochs):                                                                                    # Used above in: FNN[186]
    if(DILr.mode == 'poly'):
        ScheduelLr = PolynomialDecay(maxEpochs=DILr.StepSize,initAlpha=DILr.Lr,power=DILr.factor)
        ScheduelLr.plot(range(1,int(Epochs)+1))
        lrate = LearningRateScheduler(ScheduelLr)
    elif(DILr.mode == 'cycle'):
        lrate = CyclicLR(step_size=DILr.StepSize,mode=DILr.cycle,gamma=DILr.factor,base_lr=DILr.MinLr,max_lr=DILr.Lr)
    elif(DILr.mode == 'drop'):
        ScheduelLr = StepDecay(initAlpha=DILr.Lr, factor=DILr.factor, dropEvery=DILr.StepSize)
        ScheduelLr.plot(range(1,int(Epochs)+1))
        lrate = LearningRateScheduler(ScheduelLr)
    elif(DILr.mode == 'normal'):
        lrate = None

    return np.asarray(lrate)


def GetRocs(factory, dataloader,name):
    #This function still needs some fixing TODO
    print(name)
    f = ROOT.TFile('/gpfs/share/home/s6nsschw/Data/NNOutput.root')
    TH_Train_S = f.Get("/dataset/Method_"+name+"/"+name+"/MVA_"+name+"_Train_S")
    TH_Train_B = f.Get("/dataset/Method_"+name+"/"+name+"/MVA_"+name+"_Train_B")

    RocSame = ROOT.TMVA.ROCCalc(TH_Train_S,TH_Train_B)
    AucSame = RocSame.GetROCIntegral()
    print("Auc test")
    print(factory.GetROCIntegral(dataloader,name))       #Auc test
    print("Auc train")
    print(AucSame)                                              #Auc train

    del RocSame
    f.Close()

    return


def CrossCheck(dataloader):

    DataSet = dataloader.GetDataSetInfo().GetDataSet()                
    EventCollection = DataSet.GetEventCollection()
    BkgW, SigW = np.zeros([]), np.zeros([])
    Bkg, Sig = np.zeros([]), np.zeros([])
    for Event in EventCollection:
        if(Event.GetClass() == 1):
            Bkg  = np.append(Bkg, Event.GetValue(1)) 
            BkgW = np.append(BkgW, Event.GetWeight())
        elif(Event.GetClass() == 0):
            Sig = np.append(Sig, Event.GetValue(1))
            SigW = np.append(SigW, Event.GetWeight())
    PlotService.VarCrossCheck(Sig,Bkg,SigW,BkgW,'njets',-6,4,10)

    return


def GetClassWeights(Labels,Weights):
    ClassWeight = {}
    for Class in np.unique(Labels):
        TotalWeight = np.sum(Weights[Labels == Class])
        CWeight = np.sum(Weights)/(len(np.unique(Labels))*TotalWeight)
        ClassWeight.update({int(Class):CWeight})

    print(ClassWeight)
    return ClassWeight


def GetTrainWeights(Labels,Weights):
    """ In some cases event weights are given by Monte Carlo generators, and may turn out to be overall very small or large number.  To avoid artifacts due to this use renormalised weights.
        The  event  weights  are  renormalised such  that  both,  the  sum  of  all  weighted  signal  training  events  equals  the  sum  of  all  weights  of the background training events
        [from tmva guide] """
    Weights = np.where(Weights > 0, Weights, 0)                 #Setting negative weights to zero for training
    ReferenceLength = len(Labels[Labels == 0])
    for Class in np.unique(Labels):
        CWeight = np.sum(Weights[Labels == Class])
        RenormFactor = ReferenceLength/CWeight
        Weights = np.where(Labels != Class,Weights,Weights*RenormFactor)
    
    return Weights


def Winit(Weightinit):
    DicWinit = {'LecunNormal':tf.keras.initializers.lecun_normal(seed=None),
            'LecunUniform':tf.keras.initializers.lecun_uniform(seed=None),
            'GlorotNormal':tf.keras.initializers.GlorotNormal(seed=None),
            'GlorotUniform':tf.keras.initializers.GlorotUniform(seed=None),
            'HeNormal':tf.keras.initializers.he_normal(seed=None),
            'HeUniform':tf.keras.initializers.he_uniform(seed=None)}
    return DicWinit[Weightinit]


def FastAUC(model):
    
    train_pred = model.predict(model.X_train)
    test_pred  = model.predict(model.X_test)
    
    return roc_auc_score(model.Y_train, train_pred), roc_auc_score(model.Y_test, test_pred)


