import Utils

def GuardFNN(ANNSetup):
    """ A Guard that protects the programm from accidentally choosen big values of the NN Parameters """
    if(len(ANNSetup.Neurons) == 1):
        Utils.stderr("The NN has to have a least two layers")
        assert 0 == 1
    if(len(ANNSetup.Neurons) > 20):
        Utils.stderr("The number of layers exceeds 20")
        assert 0 == 1
    if(sum(ANNSetup.Neurons) > 2000):
        Utils.stderr("Unexpected large NN (more than 2000 Neurons)")
        assert 0 == 1
    for N in ANNSetup.Neurons:
        if(N > 2000):
            Utils.stderr("Unexpected large NN (more than 2000 Neurons in one layer)")
            assert 0 == 1
    if(int(ANNSetup.Epochs) > 200):
        Utils.stderr("The number of Epochs exceeds 200")
        assert 0 == 1
    if(int(ANNSetup.Batch) > 300000):
        Utils.stderr("The Batch size exceeds 300000")
        assert 0 == 1
    if(ANNSetup.Architecture not in ['FNN','FNNMulti','TMVA']):
        Utils.stderr("Unknown NN Architecture")
        assert 0 == 1
    if(ANNSetup.LearnRate.factor > 10):
        Utils.stderr("Unexpected high power for polynomial funciton")
        assert 0 == 1


def GuardRNN(ANNSetup):
    """ A Guard that protects the programm from accidentally choosen big values of the NN Parameters """
    if(len(ANNSetup.Neurons[0])+len(ANNSetup.Neurons[1]) == 1):
        Utils.stderr("The NN has to have a least two layers")
        assert 0 == 1
    if(len(ANNSetup.Neurons[0])+len(ANNSetup.Neurons[1]) > 10):
        Utils.stderr("The number of layers exceeds 10")
        assert 0 == 1
    if(sum(ANNSetup.Neurons[0]) > 500):
        Utils.stderr("Unexpected large NN (more than 500 LSTM Neurons)")
        assert 0 == 1
    if(sum(ANNSetup.Neurons[0])+sum(ANNSetup.Neurons[1]) > 2000):
        Utils.stderr("Unexpected large NN (more than 2000 Neurons)")
        assert 0 == 1
    for N in ANNSetup.Neurons[1]:
        if(N > 1000):
            Utils.stderr("Unexpected large NN (more than 1000 Neurons in one layer)")
            assert 0 == 1
    for N in ANNSetup.Neurons[0]:
        if(N > 300):
            Utils.stderr("Unexpected large NN (more than 300 LSTM Neurons in one layer)")
            assert 0 == 1
    if(int(ANNSetup.Epochs) > 200):
        Utils.stderr("The number of Epochs exceeds 200")
        assert 0 == 1
    if(int(ANNSetup.Batch) > 300000):
        Utils.stderr("The Batch size exceeds 300000")
        assert 0 == 1
    if(ANNSetup.Architecture not in ['LSTM','GRU']):
        Utils.stderr("Unknown NN Architecture")
        assert 0 == 1
    if(ANNSetup.LearnRate.factor > 10):
        Utils.stderr("Unexpected high power for polynomial funciton")

