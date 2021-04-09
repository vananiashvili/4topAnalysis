import numpy as np
import PlotService
from Utils import *
from Guard import GuardRNN
from Callbacks import *
from tensorflow.keras import optimizers
import ROOT
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras import backend as K

import tensorflow as tf

NUM_THREADS=1
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

def Main(ANNSetup,DataSet,BootStrap=('vali',None)):
    np.random.seed(5)
    train, test, vali = GetSamples(BootStrap,DataSet,ANNSetup.ModelName,DoTrafo=True)
    GuardRNN(ANNSetup)
    if(ANNSetup.Architecture == 'LSTM'):
        return LSTMNN(ANNSetup, test, train, DataSet.LVariables)
    elif(ANNSetup.Architecture == 'GRU'):
        return GRUNN(ANNSetup, test, train, DataSet.LVariables)


def LSTMNN(ANNSetup, test, train, VarList):
    TrainWeights = GetTrainWeights(train.OutTrue,train.Weights)

    model = Sequential()
    model.X_train = train.Events
    model.Y_train = train.OutTrue
    model.W_train = train.Weights       #Original weights!
    model.X_test  = test.Events
    model.Y_test  = test.OutTrue
    model.W_test  = test.Weights

    LSTMNeurons  = ANNSetup.Neurons[0]
    DenseNeurons = ANNSetup.Neurons[1]
    width = train.Events.shape[1]
    Seq = train.Events.shape[2]
    model.add(LSTM(LSTMNeurons[0], input_shape=(width, Seq),kernel_regularizer=l1(ANNSetup.Regu), return_sequences=True))  # kernel_regularizer=l2(ANNSetup.Regu) 
    if(ANNSetup.Dropout[0] != 0):
        model.add(Dropout(ANNSetup.Dropout[0]))
    for i in range(1,len(LSTMNeurons)):
        if(i == len(LSTMNeurons)-1):                                               # Add last LSTMLayer
            model.add(LSTM(LSTMNeurons[i]))
        else:
            model.add(LSTM(LSTMNeurons[i],return_sequences=True,dropout=ANNSetup.Dropout[i], recurrent_regularizer=l2(ANNSetup.Regu)))
        if(ANNSetup.Dropout[i] != 0):
            model.add(Dropout(ANNSetup.Dropout[i]))
    for j in range(len(DenseNeurons)):
        model.add(Dense(DenseNeurons[j], activation='selu'))

    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate.Lr)
    lrate = GetLearnRate(ANNSetup.LearnRate,ANNSetup.Epochs)
    Roc = Histories()
    if(lrate == None):
        Lcallbacks = [Roc]
    else:
        Lcallbacks = [Roc,lrate]


    model.summary()
    model.compile(optimizer=Opti, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(train.Events, train.OutTrue, sample_weight=TrainWeights, validation_data=(test.Events, test.OutTrue, test.Weights), epochs=int(ANNSetup.Epochs),
                        batch_size=int(ANNSetup.Batch), verbose=2, callbacks=Lcallbacks)

    LAuc = Roc.TestAucs
    LTrainAuc = Roc.TrainAucs
    print("Best Roc {0:.4f} at Epoch {1}".format(max(LAuc),LAuc.index(max(LAuc))+1)) #0:.4f
    print("Train Auc {0:.4f}".format(LTrainAuc[LAuc.index(max(LAuc))]))
    #print("Test Rocs: {0}".format(LAuc))

    for i in range(len(LAuc)):
        print("Auc at Epoch {0}: {1:.4f} Ov: {2:.3f}".format(i,LAuc[i],1-LAuc[i]/LTrainAuc[i]))

    model.save(ANNSetup.SavePath)

    return model, Roc


def GRUNN(ANNSetup, test, train, VarList):              #TODO: revise this!
    assert 0 == 1
    model = Sequential()
    GRUNeurons  = ANNSetup.Neurons[0]
    DenseNeurons = ANNSetup.Neurons[1]
    for i in range(len(GRUNeurons)):
        print(GRUNeurons[i])
        if(i == len(GRUNeurons)-1):
            model.add(GRU(GRUNeurons[i],activation='tanh', recurrent_activation='sigmoid'))     #,dropout=ANNSetup.Dropout[i]
        else:
            model.add(GRU(GRUNeurons[i],activation='tanh', recurrent_activation='sigmoid', return_sequences=True))                                                 
    for j in range(len(DenseNeurons)):
        model.add(Dense(DenseNeurons[j], activation='relu'))
    Opti = GetOpti(ANNSetup.Optimizer,ANNSetup.LearnRate)


    model.compile(optimizer=Opti, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train.Events,train.OutTrue, sample_weight=train.Weights, nb_epoch=int(ANNSetup.Epochs), batch_size=int(ANNSetup.Batch), verbose=2)


    # model.save(ANNSetup.SavePath)
    # model.summary()

    return model


def GetOpti(Optimizer,LearnRate):
    if(Optimizer == 'SGD'):
        Opti = optimizers.SGD(lr=LearnRate, momentum=0.0, nesterov=False)
    elif(Optimizer == 'Rmsprop'):
        Opti = optimizers.RMSprop(lr=LearnRate, rho=0.9)
    elif(Optimizer == 'Adagrad'):
        Opti = optimizers.Adagrad(lr=LearnRate)
    elif(Optimizer == 'Adam'):
        Opti = optimizers.Adam(lr=LearnRate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    return Opti

def GetLearnRate(DILr,Epochs):
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

    return lrate

def GetTrainWeights(Labels,Weights):
    """ In some cases event weights are given by Monte Carlo generators, and may turn out to be overallvery small or large number.  To avoid artifacts due to this use renormalised weights.
        The  event  weights  are  renormalised such  that  both,  the  sum  of  all  weighted  signal  training  events  equals  the  sum  of  all  weights  ofthe background training events
        [from tmva guide] """
    Weights = np.where(Weights > 0, Weights, 0)                 #Setting negative weights to zero for training
    ReferenceLength = len(Labels[Labels == 0])
    for Class in np.unique(Labels):
        CWeight = np.sum(Weights[Labels == Class])
        RenormFactor = ReferenceLength/CWeight
        Weights = np.where(Labels != Class,Weights,Weights*RenormFactor)

    return Weights




class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: {} - roc-auc_val: {} \n'.format(round(roc,4),round(roc_val,4)))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return










