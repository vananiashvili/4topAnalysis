import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import resample                                                                                # Used below[159] in GetSamples

import Transform
import PlotService
import DIClasses

HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def stderr(message):
    print(FAIL + message + ENDC)

def stdwar(message):
    print(WARNING + message + ENDC)                                                                               # Used in FNN.py[61-65]

def stdinfo(message):
    print(OKBLUE + message + ENDC)                                                                                # Used in FNN.py[61-65]

def dlen(obj):
    """ returns the length of the obj if the obj is an int or a float len=1 """
    if(isinstance(obj,list)):
        Length = len(obj)
    elif(isinstance(obj,np.ndarray)):
        Length = len(obj)
    elif(isinstance(obj,int) or isinstance(obj,float)):
        Length = 1
    else:
        stderr("object type not supported")
        print(obj, type(obj))
        assert 0 == 1

    return Length





def ContainsList(Arr):
    """ If array containce list True if not Flase """
    Flag = False
    for element in Arr:
        if(isinstance(element,list)):
            Flag = True
    return Flag

def PadArr(obj,MaxLength,PadNum=-3.4*pow(10,38)):
    """ Pads an the obj given to an MaxLength long list """
    if(hasattr(obj,'__len__')):
        if(isinstance(obj,np.ndarray)):
            obj = obj.tolist()
        diff = MaxLength - len(obj)
        if(diff > 0):
            for i in range(diff):
                obj.append(PadNum)
        elif(diff < 0):
                obj = obj[:MaxLength]
    else:
        obj = [obj]
        for i in range(MaxLength-1):
            obj.append(PadNum)
    
    return obj

def HasSequence(Arr):
    for obj in Arr[0]:
        if(isinstance(obj,np.ndarray)):
            print(obj,isinstance(obj,np.ndarray))
            return True
    return False


def ConvertArr(Arr,SequenceLength,PadNum=-3.4*pow(10,38)):
    """ Converts the input array into an 3dim array with constant Sequence length (using padding)"""
    if(SequenceLength == 1):
        for irow in range(Arr.shape[0]):
            for icol in range(Arr.shape[1]):
                if(hasattr(Arr[irow][icol],'__len__')):
                    if(len(Arr[irow][icol]) == 0):
                        Arr[irow][icol] = PadNum
                    else:
                        Arr[irow][icol] = Arr[irow][icol][0]

    else:
        Batchs = []
        for j,Batch in enumerate(Arr):                               #Event loop
            for i, element in enumerate(Batch):         #Variable loop
                element = PadArr(element,SequenceLength,PadNum=PadNum)
                if(i == 0):
                    BatchArr = element
                else:
                    BatchArr = np.vstack((BatchArr,element))
            Batchs.append(BatchArr)
        Arr = np.array(Batchs)
    
    return Arr

def CheckSame(Arr1,Arr2):
    if(len(Arr1) != len(Arr2)):
        stdinfo("Not the same (length)")
    else:
        Truth = Arr1 == Arr2
        if(False in Truth):
            #stdinfo("Not the same")
            return False
        else:
            #stdinfo("They are the same")
            return True



def GetSamples(SampleAndSeed, DataSet, Name, DoTrafo=False):                                                                          # Used in FeedForwardNeuralNet: Main
# train, test, vali = GetSamples(BootStrap, DataSet, ANNSetup.ModelName, DoTrafo=False)
    # Transform, Reodering, and adjust padding
    train, test = DataSet.GetInput(Name)                                                                                              # Defined in DIClasses.py[360]          
    vali = DataSet.vali

    if DoTrafo == True:
       
        if np.ndim(train.Events) == 2:
            TrafoFlag = 'ZScore'
        elif np.ndim(train.Events) == 3:
            TrafoFlag = 'ZScoreLSTM'
        else:
            print("Event array wrong dim")
            assert 0 == 1

        Trafo = Transform.Trafo(TrafoFlag)
        train.Events = Trafo.DefaultTrafo(train.Events,'train')
        test.Events = Trafo.DefaultTrafo(test.Events,'test')
        vali.Events = Trafo.DefaultTrafo(vali.Events,'vali')
        DataSet = DIClasses.DIDataSet(train,test,vali)
        print(np.mean(test.Events[:,1]),np.var(test.Events[:,1]))

        # Comment in to check trafo + comment out padding replacement in Trafo 
        # PlotService.VarHist(DataSet, 'NLO',"./plots/","All", "lep_0_pt", r"Trafo(p_{T}(\ell)^{\text{leading}})",-5,5,10,Norm=False)
        # PlotService.VarHist(DataSet, 'NLO',"./plots/","All", "nJets", "Trafo(Jet multiplicity)",-5,5,10,Norm=False)
        # assert 0 == 1 
    
    else:
        train.Events = np.where(train.Events != -3.4*pow(10,38), train.Events, 0)
        test.Events = np.where(test.Events != -3.4*pow(10,38), test.Events, 0)
        vali.Events = np.where(vali.Events != -3.4*pow(10,38), vali.Events, 0)


    # BootStrap
    Sample = SampleAndSeed[0]
    Seed   = SampleAndSeed[1]
    
    if Seed != None:
        Samples = {'train':train, 'test':test, 'vali':vali}
        Sample = Samples[Sample]
        
        Comb = np.hstack((Sample.Events, Sample.Weights.reshape(-1,1), Sample.OutTrue.reshape(-1,1)))             # np.hstack() stacks the sequence of input arrays horizontally (i.e. column wise)
                                                                                                                  # .reshape(-1,1) reshapes the array into an array with 1 column and necessary number of rows
        
        Comb = resample(Comb, replace=True, n_samples=len(Comb), random_state=Seed)                               # sklearn.resample                                 
        
        Sample.Events = Comb[:,:-2]
        Sample.Weights = Comb[:,-2:-1].ravel()
        Sample.OutTrue = np.array(Comb[:,-1], dtype=int)

    return train, test, vali








    # ------------------------------------ AUC -------------------------------------------------------

def roc_auc_score(classes : np.ndarray,
               predictions : np.ndarray,
               sample_weight : np.ndarray = None) -> float:
    """
    Calculating ROC AUC score as the probability of correct ordering
    """

    if sample_weight is None:
        sample_weight = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(sample_weight)
    assert classes.ndim == predictions.ndim == sample_weight.ndim == 1
    class0, class1 = sorted(np.unique(classes))

    data = np.empty(
            shape=len(classes),
            dtype=[('c', classes.dtype),
                   ('p', predictions.dtype),
                   ('w', sample_weight.dtype)]
        )
    data['c'], data['p'], data['w'] = classes, predictions, sample_weight

    data = data[np.argsort(data['c'])]
    data = data[np.argsort(data['p'], kind='mergesort')] # here we're relying on stability as we need class orders preserved

    correction = 0.
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data['p'][1:] == data['p'][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        ids, = mask2.nonzero()
        correction = sum([((dsplit['c'] == class0) * dsplit['w'] * msplit).sum() * 
                          ((dsplit['c'] == class1) * dsplit['w'] * msplit).sum()
                          for dsplit, msplit in zip(np.split(data, ids), np.split(mask1, ids))]) * 0.5
 
    sample_weight_0 = data['w'] * (data['c'] == class0)
    sample_weight_1 = data['w'] * (data['c'] == class1)
    cumsum_0 = sample_weight_0.cumsum()

    return ((cumsum_0 * sample_weight_1).sum() - correction) / (sample_weight_1.sum() * cumsum_0[-1])




