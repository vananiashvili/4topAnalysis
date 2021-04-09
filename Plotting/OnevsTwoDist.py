import os
import Plotting.Basics
import numpy as np

def FeaturePlots(Samples):

    Plotting.SavePath = "./plots/1vs2Had/"
    Plotting.Overflow = True

    # Feature(Samples,"met_met",30,0,600,"E_{miss} [GeV]")
    # Feature(Samples,"met_phi",12,-3.141,3.141,"\phi_{miss}")
    # Feature(Samples,"nJets",8,6,14,"number of Jets")
    # Feature(Samples,"nBJets",8,0,8,"number of b-Jets")
    Feature(Samples,"SumHighestbJets",50,1.5,2.5,"Sum mv2c10 two most b like jets")
    # Feature(Samples,"nBTags_MV2c10_77",4,0,4,"Truth number of had. Tops")

    # Feature(Samples,'dR1blight',10,0,2,"dR1(b,light)")
    # Feature(Samples,'dR2blight',25,0,5,"dR2(b,light)")

    # Feature(Samples,'dR1bl',25,0,5,"dR1(b,l)")
    # Feature(Samples,'dR2bl',35,0,7,"dR2(b,l)")



    # Feature(Samples,"leadingJetpt",40,0,800,"leading Jet p_{T} [GeV]")
    # Feature(Samples,"secondJetpt",40,0,800,"second leading Jet p_{T} [GeV]")
    # Feature(Samples,"theirdJetpt",40,0,800,"theird leading Jet p_{T} [GeV]")

    #Feature(Samples,"nLeps",5,0,5,"number of leptons")
    #Feature(Samples,"nel",5,0,5,"number of electrons")
    #Feature(Samples,"nmu",5,0,5,"number of muons")



def Feature(Samples,Var,bins,xmin,xmax,XTitle):
    YTitle = "Yield"


    index = Samples[0].LVariables.index(Var)
    if(XTitle[-5:] != "[GeV]"):
        NoHad = Plotting.Basics.H1D(Samples[0].Events[:,index],bins,xmin,xmax,Weights=Samples[0].Weights,Norm=True)
        OneHad = Plotting.Basics.H1D(Samples[1].Events[:,index],bins,xmin,xmax,Weights=Samples[1].Weights,Norm=True)
        TwoHad = Plotting.Basics.H1D(Samples[2].Events[:,index],bins,xmin,xmax,Weights=Samples[2].Weights,Norm=True)
        ThreeHad = Plotting.Basics.H1D(Samples[3].Events[:,index],bins,xmin,xmax,Weights=Samples[3].Weights,Norm=True)
    else:
        NoHad = Plotting.Basics.H1D(Samples[0].Events[:,index]/1000.,bins,xmin,xmax,Weights=Samples[0].Weights,Norm=True)
        OneHad = Plotting.Basics.H1D(Samples[1].Events[:,index]/1000.,bins,xmin,xmax,Weights=Samples[1].Weights,Norm=True)
        TwoHad = Plotting.Basics.H1D(Samples[2].Events[:,index]/1000.,bins,xmin,xmax,Weights=Samples[2].Weights,Norm=True)
        ThreeHad = Plotting.Basics.H1D(Samples[3].Events[:,index]/1000.,bins,xmin,xmax,Weights=Samples[3].Weights,Norm=True)
    NoHad.SetLineColor(1)
    OneHad.SetLineColor(4)
    TwoHad.SetLineColor(2)
    ThreeHad.SetLineColor(8)


    leg = Plotting.Basics.Legend([NoHad,OneHad,TwoHad,ThreeHad],["0 had. Tops","1 had. Tops","2 had. Tops","3 had. Tops"])
    Plotting.Basics.Hist1DStack([OneHad,TwoHad],XTitle,YTitle,Var,Leg=leg,Drawopt="nostack")
    os.system("mv "+Plotting.SavePath+"StackHist.png "+Plotting.SavePath+Var+".png")


def MET(Samples):

    Plotting.SavePath = "./plots/Features/"
    Plotting.Overflow = True

    TwoLep, WTwoLep = [], []
    ThreeLep, WThreeLep = [], []
    for i,Sample in enumerate(Samples):
        idx = Sample.LVariables.index("nLeps")
        idx2 = Sample.LVariables.index("met_met")
        for j,Event in enumerate(Sample.Events):
            if(Event[idx] == 2):
                TwoLep.append(Event[idx2])
                WTwoLep.append(Samples[i].Weights[j])
            elif(Event[idx] > 2):
                ThreeLep.append(Event[idx2])
                WThreeLep.append(Samples[i].Weights[j])

    

    TwoLep = Plotting.Basics.H1D(np.array(TwoLep)/1000.,30,0,600,Weights=WTwoLep,Norm=True)
    ThreeLep = Plotting.Basics.H1D(np.array(ThreeLep)/1000.,30,0,600,Weights=WThreeLep,Norm=True)
    ThreeLep.SetLineColor(2)

    leg = Plotting.Basics.Legend([TwoLep,ThreeLep],["2 leptons","3 leptons"])
    Plotting.Basics.Hist1DStack([TwoLep,ThreeLep],"E_{miss} [GeV]","norm. Yield","met_met",Leg=leg,Drawopt="nostack")



