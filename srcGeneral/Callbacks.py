import numpy as np
import ROOT
from root_numpy import fill_hist
from tensorflow.keras.callbacks import *
from Utils import roc_auc_score

class Histories(Callback):                                                                                        # Used in: FeedforwardNeuralNet.py[188]
    def on_train_begin(self, logs={}):
        self.TestAucs = []
        self.TestLosses = []
        self.TrainAucs = []
        self.TrainLosses = []
 
    def on_train_end(self, logs={}):
        # Evalute Loss and Auc for test
        #self.TestLosses.append(logs.get('val_loss'))
#        y_pred = self.model.predict(self.model.X_test)
#        if(y_pred.shape[1] == 1):
#            self.TestAucs.append(roc_auc_score(self.model.Y_test, y_pred, sample_weight=self.model.W_test))
#        else:
#            self.TestAucs.append(roc_auc_score(self.model.Y_test, y_pred[:,0], sample_weight=self.model.W_test))

        # Evalute Loss and Auc for train
        #self.TrainLosses.append(logs.get('loss'))
#        y_pred_Train = self.model.predict(self.model.X_train)
#        if(y_pred.shape[1] == 1):
#            self.TrainAucs.append(roc_auc_score(self.model.Y_train, y_pred_Train, sample_weight=self.model.W_train))
#        else:
#            self.TrainAucs.append(roc_auc_score(self.model.Y_train, y_pred_Train[:,0], sample_weight=self.model.W_train))

        #Save the Score if the Auc is the current highest
#        if(self.TestAucs[-1] == max(self.TestAucs)):
#            self.MaxPre = [y_pred,y_pred_Train]

        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        # # Evalute Loss and Auc for test
        # self.TestLosses.append(logs.get('val_loss'))
            y_pred = self.model.predict(self.model.X_test)
            if(y_pred.shape[1] == 1):
                y_pred = y_pred.flatten()
                self.TestAucs.append(roc_auc_score(self.model.Y_test, y_pred, sample_weight=self.model.W_test))
            else:
                self.TestAucs.append(roc_auc_score(self.model.Y_test, y_pred[:,0], sample_weight=self.model.W_test))

        # # Evalute Loss and Auc for train
        # self.TrainLosses.append(logs.get('loss'))
            y_pred_Train = self.model.predict(self.model.X_train)
            if(y_pred_Train.shape[1] == 1):
                y_pred_Train = y_pred_Train.flatten()
                self.TrainAucs.append(roc_auc_score(self.model.Y_train, y_pred_Train, sample_weight=self.model.W_train))
            else:
                self.TrainAucs.append(roc_auc_score(self.model.Y_train, y_pred_Train[:,0], sample_weight=self.model.W_train))

            #Save the Score if the Auc is the current highest
            if(self.TestAucs[-1] == max(self.TestAucs)):
                self.MaxPre = [y_pred,y_pred_Train]
        

            return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

        




class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
		# compute the set of learning rates for each corresponding
		# epoch
        lrs = [self(i) for i in epochs]
		# the learning rate schedule
        c1 = ROOT.TCanvas("c1","Canvas",800,600)
        ROOT.gStyle.SetOptStat(0)

        combined = np.vstack((epochs,lrs)).T
        h1 = ROOT.TH2F("h1", title,len(epochs)*10,0,max(epochs)+1,len(lrs)*10,min(lrs)-min(lrs)*0.1,max(lrs)+max(lrs)*0.1)
        fill_hist(h1,combined)
        h1.GetXaxis().SetTitle("Epoch")
        h1.GetYaxis().SetTitle("Learn Rate")
        h1.SetMarkerStyle(21)
        h1.Draw("L")

        c1.SaveAs("./plots/LearnRateScheduel.png")

class StepDecay(LearningRateDecay):
	def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=10):
		# store the base initial learning rate, drop factor, and
		# epochs to drop every
		self.initAlpha = initAlpha
		self.factor = factor
		self.dropEvery = dropEvery

	def __call__(self, epoch):
		# compute the learning rate for the current epoch
		exp = np.floor((1 + epoch) / self.dropEvery)
		alpha = self.initAlpha * (self.factor ** exp)

		# return the learning rate
		return float(alpha)

class PolynomialDecay(LearningRateDecay):
	def __init__(self, maxEpochs=100, initAlpha=0.01, power=1.0):
		# store the maximum number of epochs, base learning rate,
		# and power of the polynomial
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power

	def __call__(self, epoch):
		# compute the new learning rate based on polynomial decay
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay

		# return the new learning rate
		return float(alpha)


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        """ Calculates the next learning rate for the next batch """
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))                      # Im wievielten cycyle befinden wir uns gerade?
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)                    # procentage between maximum and minimum value
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr())



class RedHistory(Callback):
    def on_train_begin(self, logs={}):
        self.TestAucs = []
        self.TrainAucs = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        if(epoch == 64 or epoch == 59):
            y_pred = self.model.predict(self.model.X_test)
            self.TestAucs.append(roc_auc_score(self.model.Y_test, y_pred, sample_weight=self.model.W_test))

            y_pred_Train = self.model.predict(self.model.X_train)
            self.TrainAucs.append(roc_auc_score(self.model.Y_train, y_pred_Train, sample_weight=self.model.W_train))

            #Save the Score if the Auc is the current highest
            if(self.TestAucs[-1] == max(self.TestAucs)):
                self.MaxPre = [y_pred,y_pred_Train]
        
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

