import ROOT
import copy
import numpy as np
from math import sqrt
import PlotService
from DIClasses import DISample, DIDataSet
import Utils
import os
from root_numpy import tree2array, root2array, array2root
from sklearn.model_selection import train_test_split



class SampleHandler:                                                                                              # Used in FNN.py[81]

    valSize         = 0.2
    Split           = 'EO'
    NormFlag        = True
    SequenceLength  = 1
    Plots           = False

    def __init__(self, ListAnaSetup, mode='Slow', Single=False):                                                  # Sampler = SampleHandler(ListSamples, mode=Mode+ModelName) 
                                                                                                                  #     ListSamples = DIClasses.Init(ModelName, Samples, Cuts=True)
        self.ListAnaSetup = ListAnaSetup                                                                          #                                 ('FNN18', 'nomLoose')
        self.mode         = mode                                                                                  #     mode = Mode + ModelName = 'slowFNN18'
        self.Single       = Single                                                                                # ? ? ? ? ?


    def GetANNInput(self):

        # SLOW or SAVE Modes  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ # 
        if self.mode[:4] == 'Slow' or self.mode[:4] == 'Save':
            
            Utils.stdinfo("Setting up input vectors")
            
            # Import Variables as a NumPy array 
            for DISetup in self.ListAnaSetup:
                
                print("Processing Sample: " + DISetup.Name)
                
                ListOfVariables = DISetup.LVars[:]

                ListOfVariables.extend(DISetup.WeightList)
                
                Arr = self.GetArray(ListOfVariables, DISetup)                                                                         # Defined below
                
                np.random.shuffle(Arr)                                                                                                # Ensures randomization of event order. Seed fixed in FNN.py[17]
                
                DISetup.Samples = self.MakeSplit(Arr, DISetup)                                                                        # Split samples
                
                self.Info(DISetup)
                
                        
            # Combine the different sets into a DataSet
            Utils.stdinfo("Finalising input Preparation")
            
            ListSamples = [DISetup.Samples for DISetup in self.ListAnaSetup]
            
            train = self.Finalise("train", ListSamples)
            test  = self.Finalise("test", ListSamples)
            vali  = self.Finalise("validation", ListSamples)
            
            DataSet = DIDataSet(train, test, vali)


        # Save the numpy arrays to load them faster later
        if self.mode[:4] == 'Save':
            if self.verbose == 1:
                Utils.stdinfo("Saving data for Fast mode")
            self.SaveNpy('TrainSet' + self.mode[4:], train)
            self.SaveNpy('TestSet' + self.mode[4:], test)
            self.SaveNpy('ValiSet' + self.mode[4:], vali)
            
        # If stored as numpy arrays load the arrays
        elif self.mode[:4] == 'Fast':
            train = self.GetSampleNpy('TrainSet' + self.mode[4:])
            test = self.GetSampleNpy('TestSet' + self.mode[4:])
            vali = self.GetSampleNpy('ValiSet' + self.mode[4:])
            DataSet = DIDataSet(train, test, vali)

        # Make Plots of the Variables
        if self.Plots != False:
            PlotService.VarHists(DataSet, key='All', Norm=self.NormFlag, Sig=self.Plots) 

        return DataSet



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   G e t   A r r a y   f r o m   R O O T                                                                         #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

    def GetArray(self, ListOfVariables, DISetup):                                                                                     # Used above in GetANNInput
        
        """ Import the Variables from the ROOT file """

        for i, path in enumerate(DISetup.Path):
            if DISetup.Cuts == '':
                selection = DISetup.McChannel
            elif DISetup.McChannel == '':
                selection = DISetup.Cuts
            else:
                selection = "(" + DISetup.McChannel + ") && (" + DISetup.Cuts + ")"

            # get the array from the tree
            rfile  = ROOT.TFile(path)
            intree = rfile.Get(DISetup.Tree)
            Arr = tree2array(intree, branches=ListOfVariables, selection=selection)
            Arr = np.array(Arr.tolist(), dtype=object)
            
            if i == 0:
                TotalArr = Arr
            elif Arr.shape != (0,):
                TotalArr = np.append(TotalArr, Arr, axis=0)

        TotalArr = Utils.ConvertArr(TotalArr, self.SequenceLength)                                                                    # Convert into a format that is NumPy compatible
        
        return np.array(TotalArr, dtype=np.float64)



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   S p l i t   A r r a y   i n t o:   T r a i n,   T e s t,   V a l i d a t i o n                                #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

    def MakeSplit(self, Arr, DISetup):
        
        """ Split the array into Training, Testing and Validation sets """
        
        # Validation
        if DISetup.Name != "tttt":                                                                                # Except for LO Sample (tttt), 20% of all samples are used for validation
            validation, Arr      = self.MakeValidation(Arr, self.valSize)                                         # Defined below
        else:                                                                       
            validation = None
        
        # Test, Train
        if DISetup.Name != "tttt" and DISetup.Name != "NLO":                                                      # Split background samples into 40% test 40% train
            test, train          = self.MakeTestTrain(Arr, self.Split)                                            # Defined below
        
        elif DISetup.Name == "tttt":                                                                              # Only training on LO Sample
            test  = None
            train = Arr
        
        elif DISetup.Name == "NLO":                                                                               # Only testing on NLO Sample
            test  = Arr
            train = None
        
        else:
            Utils.stderr("Not a sample!")
            assert 0 == 1
        
        Sample               = {"train": train, "test": test, "validation": validation}
        Sample["train"]      = self.MakeDISample(train, DISetup)
        Sample["test"]       = self.MakeDISample(test, DISetup)
        Sample["validation"] = self.MakeDISample(validation, DISetup)
        
        return Sample



    def MakeValidation(self, Arr, valSize=0.2):                                                                   # Used above in MakeSplit
        
        splitIndex = int(round(Arr.shape[0] * (1-valSize)))                                                       # Determines the index of splitting 
        
        return Arr[splitIndex:], Arr[:splitIndex]                                                                 # Splits the array according to the index
    


    def MakeTestTrain(self, Arr, Split='EO'):                                                                     # Used above in MakeSplit
        
        """ Different ways to split the samples """
        
        # Splitting according to Even/Odd event number
        if Split == 'EO':
            train = Arr[::2]
            test  = Arr[1::2]
        
        # Splitting in Half
        elif Split == 'H':                                   
            train  = Arr[:len(Arr)/2]
            test = Arr[len(Arr)/2:]
        
        # Splitting with a Keras function
        elif Split == 'K':
            train, test = train_test_split(Arr)
        
        return test, train



    def MakeDISample(self, Arr, DISetup):
        
        """ Split the imported Array into Events (the Variables) and the Weights """
        
        if not isinstance(Arr, np.ndarray):
            return None
        
        else:
            Events   = Arr[:, :len(DISetup.LVars)]
            Weights  = Arr[:, len(DISetup.LVars):]
            Weights  = self.GetWeightSum(Weights, DISetup)                                                        # Defined below
            
            if DISetup.Name == "tttt" or DISetup.Name == "NLO":
                OutTrue = np.ones([len(Events)])

            else:
                OutTrue = np.zeros([len(Events)])
            
            Names = [DISetup.Name for DISetup in self.ListAnaSetup]
            
            return DISample(Events, Weights, OutTrue, None, self.ListAnaSetup[0].LVars, Names)



    def GetWeightSum(self, Weights, DISetup):                                                                      # Used above in MakeDISample
        
        """ Calculates the weights for each event from the weight expression """
        
        # Imported directly as expression in GetArray
        if len(DISetup.WeightList) == 1:
            
            if np.ndim(Weights) == 2:
                Weight = Weights.flatten()
            
            elif np.ndim(Weights) == 3:
                Weight = Weights[:,:,0]
                Weight = Weight.flatten()


        # Calculation "by hand" (old)
        else:
            SingleWeights = [(36207.7,284500), (44307.4,300000), (58450.1,310000)]
            
            if len(DISetup.WeightList) == 9:
                Weight = Weights[:,0]*Weights[:,1]*Weights[:,2]*Weights[:,3]*Weights[:,4]*Weights[:,5]/Weights[:,6]*Weights[:,7]
            
            else:
                Weight = Weights[:,0]*Weights[:,1]*Weights[:,2]*Weights[:,3]*Weights[:,4]*Weights[:,5]

            
            for i in range(len(Weights)):
                for k in range(len(SingleWeights)):
                    if(Weights[i][DISetup.WeightList.index('runNumber')] == SingleWeights[k][1]):
                        Weight[i] *= SingleWeights[k][0]


        return Weight



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   F i n a l i s e                                                                                               #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

    def Finalise(self, key, ListSamples):
        
        # concatenate all Bkg and Signal Samples for the input
       
        LSample = [Samples[key] for Samples in ListSamples]
        LSample = list(filter(None, LSample))
        
        listAllEvents     = [Sample.Events for Sample in LSample]
        listAllWeights    = [Sample.Weights for Sample in LSample]
        listAllOutTrue    = [Sample.OutTrue for Sample in LSample]
        listAllMultiClass = []
        
        Names = [DISetup.Name for DISetup in self.ListAnaSetup]                            #For Multiclass tttt = 0 (in Binary tttt = 1)                                   
        # for i,Sample in enumerate(listAllOutTrue):
        #     if(Names[i] in ['ttW','ttH','ttZ']):
        #         MultiClass = np.full((len(Sample)),1)
        #     elif(Names[i] in ["tttt"]):
        #         MultiClass = np.full((len(Sample)),0)
        #     else:
        #         MultiClass = np.full((len(Sample)),2)
        #     listAllMultiClass.append(MultiClass)

        ClassNum = 0         
        
        for i, Sample in enumerate(listAllOutTrue):                                                    #14 Classes
            MultiClass = np.full((len(Sample)), ClassNum)
            listAllMultiClass.append(MultiClass)                                                      #NLO and LO are both singal
            ClassNum += 1


        AllWeights      = np.concatenate(listAllWeights,axis=0)
        AllOutTrue      = np.concatenate(listAllOutTrue,axis=0)
        AllMultiClass   = np.concatenate(listAllMultiClass,axis=0)
        AllEvents       = np.concatenate(listAllEvents,axis=0)

        if key == "train":                                                             #Checking for negative Weights in the Sample                                                            
            for i, Sample in enumerate(LSample):
                NWeights = 0
                for weight in Sample.Weights:
                    if weight < 0:
                        NWeights += 1
                if NWeights != 0 and self.verbose == 1:
                    Utils.stdwar("We have negative weights! {0} in {1} (train)".format(NWeights, self.ListAnaSetup[i].Name))

        Sample = DISample(AllEvents, AllWeights, AllOutTrue, AllMultiClass, self.ListAnaSetup[0].LVars, Names)
        
        return Sample



# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   D e a l i n g   w i t h   N u m P y   A r r a y s                                                             #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

    def SaveNpy(self,fname,Sample):
                                                                                                                                      
        """ Saves the NumPy Array containing the data to the indicated location """                                                   
                                                                                                                                      
        fpath = '/mnt/c/Users/Vakho/Desktop/'
                                                    # fpath = os.path.expanduser("~") 
                                                    # if(fpath == '/jwd'):
                                                    #   fpath = '/cephfs/user/s6nsschw'                                                                                      
        
        np.save(fname + '_Events.npy', Sample.Events)
        os.system("mv " + fname + "_Events.npy " + fpath) # + "/Data/Fast/")
        Other = np.hstack((Sample.Weights.reshape(-1,1), Sample.OutTrue.reshape(-1,1), Sample.MultiClass.reshape(-1,1)))
        np.save(fname+'_Other.npy', Other)
        os.system("mv " + fname + "_Other.npy " + fpath)  # +"/Data/Fast/")



    def GetSampleNpy(self, fname):

        """ Loads the date from previously saved NumPy Arrays """

        fpath = "/mnt/c/Users/Vakho/Desktop/4tNNFrameWork/SourceArrays/"                            
                                                    # fpath = os.path.expanduser("~")
                                                    # if fpath == "/jwd":
                                                    #     fpath = "/cephfs/user/s6vaanan"

        Events     = np.load(fpath + fname + "_Events.npy", allow_pickle=True)
        Other      = np.load(fpath + fname + "_Other.npy", allow_pickle=True)
        Weights    = Other[:,0]
        OutTrue    = np.array(Other[:,1], dtype=int)
        MultiClass = np.array(Other[:,2], dtype=int)
        
        Names = [DISetup.Name for DISetup in self.ListAnaSetup]
        
        return DISample(Events, Weights, OutTrue, MultiClass, self.ListAnaSetup[0].LVars, Names)


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#   D i s p l a y   t h e   I n f o r m a t i o n                                                                 #
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #

    def Info(self, DISetup):
        
        """" Displays the information about the sample sizes """

        train = DISetup.Samples["train"]
        test  = DISetup.Samples["test"]
        vali  = DISetup.Samples["validation"]
        
        if DISetup.Name == "tttt":
            print("The whole sample has {0} events".format(len(train.Events)))
            print("The training sample contains {0} events".format(len(train.Events)))
            Yield = np.sum(train.Weights)
        
        elif DISetup.Name == "NLO":
            print("The whole sample has {0} events".format(len(test.Events) + len(vali.Events)))
            print("The testing sample contains {0} events".format(len(test.Events)))
            print("The validation sample contains {0} events".format(len(vali.Events)))
            Yield = np.sum(test.Weights)+np.sum(vali.Weights)
        
        else:
            print("The whole sample has {0} events".format(len(train.Events) + len(test.Events) + len(vali.Events)))
            print("The training sample contains {0} events".format(len(train.Events)))
            print("The testing sample contains {0} events".format(len(test.Events)))
            print("The validation sample contains {0} events".format(len(vali.Events)))
            Yield = np.sum(train.Weights) + np.sum(test.Weights) + np.sum(vali.Weights)
        
        Utils.stdinfo("The total Yield amounts to: {0}".format(Yield))

        








