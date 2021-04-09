Niklas Schwan 16.11.2020

/*******************************************\
/*** tttt analysis using Neural Networks ***\
/*******************************************\

This Framework was not made for sharing it with other people. If there is anything unclear ask me.


------
Usage
------

First steps:
1. Adjust import path in DIClasses.py line 116 - 125
2. Adjust Fast save path in SampleHandler line 260 ff.
3. Adjust SavePath for Evaluation etc. line 82


python FNN.py	(For Feedforward Neural Networks)
python RNN.py	(For Recurrent Neural Networks)

Usage on baf: Set additional flag for treat controll (https://confluence.team.uni-bonn.de/x/b4e7)


------
FNN.py
------
I will give some commands on the flags etc. in this file now:
Samples			Use 'nomLoose' if you want to use all samples in the anaylsis else give it a list (which samples are implemented you can see in DIClasses.py)
ModelName		Name of the model also the flag that is used to find which variables should be used in the Neural Network
BDT			BDT model Name if None no Bdt is trained (can only be used if the "Type" is TMVA)
EvalFlag		Online evaluation of the NN you can also load a save model and use the script "Evalfromsave.py"
Odd			As you may have read in my thesis I train two Neural Networks one on the odd and one on the even backgrounds. If this flag is false only the Even NN is trained
			this saves time but allways train both when you optimize
PreTrained		This is a flag from the Evaluation script if the evaluation was done before the predictions on the test dataset are saved and can be reused by setting this flag to True
SavePath		Where to save prediction and models
Type			I implemented the NN in two different ways directly in keras (FNN or FNNMulti) and using the TMVA framework. TMVA was there to compare and for the BDT I would not recomment to use
			it for opimization its not flexable enough. FNN is for binary classification and FNNMulti for multi class how much classes you want you need to definie in SampleHandeler.py default 				are 14 Classes (all backgrounds as one class)
Opti			Optimizer of the Neural Network (implemented SGD, Adam, Rmsprop and Adagrad)
Winit			Weight initilaization (implemented Lecun, Glorot and He for both uniform and normal distributions)
activation		Activation functions 'elu','relu','selu' etc. When you give it a number it will be the slope of leaky relu
Epochs			The Number of Epochs that the NN is trained
Batch			Batch size
Neurons			List of Neurons. First entry = first hidden layer, second entry = second hidden layer .... (Input layer is set by keras and equals number of variables). TMVA uses 2 output neurons
			for binary classification I use 1 both the difference is minor
Dropout			prop. of dropout (only applied to first hidden layer)
Regu			l2 regularization (only first layer)
Bootstrap		tuple first entry on which sample should bootstrap be applied? second the random seed used for bootstrapping if none no bootstrapping is performed
LearnRate		The different learning rate schedules (You can find the used Callbacks in Callbacks.py in folder srcGeneral)

GPU			If false CPUs will be used. You might need to rebuild the anaconda enveriment for the gpu in your system and also for the baf system
Mode			If Slow the dataSets will be loaded from ROOTFiles. If Save datasets loaded from root files and stored as scimmed Numpy files. Fast the scimmed numpy files will be used
NormFlag		Should the plots the normalized too total Yield == one can be helpful when looking at shapes
valSize			The fraction of the whole dataset that is speared for the validation set
Split			use 'EO' even odd splitting
Plots			Should there be plots of the variables?



-----
RNN.py
-----
Very similar to the flags in FNN.py main differences are:
1. Only Keras implementation (on TMVA or BDT)
2. Neurons addtional list if you want to put FFN layers behind LSTM layers
3. Regu == l1 regularization term
4. Dropout can be applied to all layers
5. Sampler.SequenceLength how high is the maximum number of particles of the same type


-----
General
-----

If you want to plot more than the 4-5 plot types that are produced per default have a look at PlotService.py.
The folder Plotting was a project where I started to make everything more readable but I did not finish it during my masters - anyways you might find it helpful.
I did go through the main function they should work correctly. If a function does not let me know




