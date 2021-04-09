import os
import Plotting.Basics
import numpy as np

def FeaturePlots(Samples):

    Plotting.SavePath = "./plots/Features/LowLevel/"
    Plotting.Overflow = True

    Feature(Samples,"met_met",30,0,500,"E_{miss} [GeV]")
    Feature(Samples,"met_phi",12,-3.141,3.141,"\phi_{miss}")
    # Feature(Samples,"nJets",8,6,14,"number of Jets")
    # Feature(Samples,"nBTags_MV2c10_77",8,0,8,"number of b-Jets")
    # Feature(Samples,"SumHighestbJets",80,0,4,"Sum mv2c10 two most b like jets")
    # Feature(Samples,"nHad",4,0,4,"Truth number of had. Tops")

    # Feature(Samples,'dR1blight',10,0,2,"dR1(b,light)")
    # Feature(Samples,'dR2blight',25,0,5,"dR2(b,light)")

    # Feature(Samples,'dR1bl',25,0,5,"dR1(b,l)")
    #Feature(Samples,'dR2bl',20,0,5,"dR2(b,l)")

    # Feature(Samples,'MFJeteta',30,-1,5,"most forward Jet eta")
    # Feature(Samples,'MFJetpt',20,0,400,"most forward Jet pt [GeV]")

    # Feature(Samples,"leadingJetpt",30,0,800,"leading Jet p_{T} [GeV]")
    # Feature(Samples,"secondJetpt",40,0,800,"second leading Jet p_{T} [GeV]")
    # Feature(Samples,"theirdJetpt",40,0,800,"theird leading Jet p_{T} [GeV]")

    # Feature(Samples,"nLeps",5,0,5,"number of leptons")
    # Feature(Samples,"nel",5,0,5,"number of electrons")
    # Feature(Samples,"nmu",5,0,5,"number of muons")

    SeqFeature(Samples,"jet_pt",20,0,[800,500,400,300,250,200,150,100,100,100,100,100,100,100,100,100,100,100],"jet p_{T} [GeV]")
    SeqFeature(Samples,"jet_eta",20,-2.7,[2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7,2.7],"jet \eta")
    SeqFeature(Samples,"jet_phi",20,-3.141,[3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141,3.141],"jet \phi")
    SeqFeature(Samples,"jet_e",20,0,[1500,1000,800,600,600,600,400,300,300,300,300,300,300,300,300,300,300,300],"jet E [GeV]")

    SeqFeature(Samples,"el_pt",20,0,[500,250],"e p_{T} [GeV]")
    SeqFeature(Samples,"el_e",20,0,[800,600],"e E [GeV]")
    SeqFeature(Samples,"el_eta",20,-2.7,[2.7,2.7],"e \eta")
    SeqFeature(Samples,"el_phi",20,-3.141,[3.141,3.141],"e \phi")

    SeqFeature(Samples,"mu_pt",20,0,[500,250],"\mu p_{T} [GeV]")
    SeqFeature(Samples,"mu_e",20,0,[800,600],"\mu E [GeV]")
    SeqFeature(Samples,"mu_eta",20,-2.7,[2.7,2.7],"\mu \eta")
    SeqFeature(Samples,"mu_phi",20,-3.141,[3.141,3.141],"\mu \phi")


def Feature(Samples,Var,bins,xmin,xmax,XTitle):
    YTitle = "Yield"

    index = Samples[0].LVariables.index(Var)
    if(XTitle[-5:] != "[GeV]"):
        FourTop = Plotting.Basics.H1D(Samples[0].Events[:,index],bins,xmin,xmax,Weights=Samples[0].Weights.flatten(),Norm=True)
        ThreeTop = Plotting.Basics.H1D(Samples[1].Events[:,index],bins,xmin,xmax,Weights=Samples[1].Weights.flatten(),Norm=True)
    else:
        FourTop  = Plotting.Basics.H1D(Samples[0].Events[:,index]/1000.,bins,xmin,xmax,Weights=Samples[0].Weights.flatten(),Norm=True)
        ThreeTop = Plotting.Basics.H1D(Samples[1].Events[:,index]/1000.,bins,xmin,xmax,Weights=Samples[1].Weights.flatten(),Norm=True)
    FourTop.SetLineColor(4)
    ThreeTop.SetLineColor(2)


    leg = Plotting.Basics.Legend([ThreeTop,FourTop],["Three Top","Four Top"])
    Plotting.Basics.Hist1DStack([ThreeTop,FourTop],XTitle,YTitle,Var,Leg=leg,Drawopt="nostack")
    os.system("mv "+Plotting.SavePath+"StackHist.png "+Plotting.SavePath+Var+".png")

def SeqFeature(Samples,Var,bins,xmin,Lxmax,XTitle):
    YTitle = "norm. Yield"

    maxlen = 0
    index = Samples[0].LVariables.index(Var)
    for i in range(len(Samples)):
        for Event in Samples[i].Events[:,index]:
            if(len(Event) > maxlen):
                maxlen = len(Event)

    for Seq in range(maxlen):
        xmax = Lxmax[Seq]
        FourTop      = np.array([Event[Seq] for Event in Samples[0].Events[:,index] if Seq < len(Event)])
        ThreeTop     = np.array([Event[Seq] for Event in Samples[1].Events[:,index] if Seq < len(Event)])
        FourWeights  = np.array([Samples[0].Weights[i] for i in range(len(Samples[0].Weights)) if Seq < len(Samples[0].Events[:,index][i])])
        ThreeWeights = np.array([Samples[1].Weights[i] for i in range(len(Samples[1].Weights)) if Seq < len(Samples[1].Events[:,index][i])])
        if(XTitle[-5:] != "[GeV]"):
            FourTop  = Plotting.Basics.H1D(FourTop,bins,xmin,xmax,Weights=FourWeights.flatten(),Norm=True)
            ThreeTop = Plotting.Basics.H1D(ThreeTop,bins,xmin,xmax,Weights=ThreeWeights.flatten(),Norm=True)
        else:
            FourTop  = Plotting.Basics.H1D(FourTop/1000.,bins,xmin,xmax,Weights=FourWeights.flatten(),Norm=True)
            ThreeTop = Plotting.Basics.H1D(ThreeTop/1000.,bins,xmin,xmax,Weights=ThreeWeights.flatten(),Norm=True)
        FourTop.SetLineColor(4)
        ThreeTop.SetLineColor(2)

        leg = Plotting.Basics.Legend([ThreeTop,FourTop],["Three Top","Four Top"])
        Plotting.Basics.Hist1DStack([ThreeTop,FourTop],XTitle+" "+str(Seq),YTitle,Var,Leg=leg,Drawopt="nostack")
        os.system("mv "+Plotting.SavePath+"StackHist.png "+Plotting.SavePath+Var+str(Seq)+".png")


# def MET(Samples):

#     Plotting.SavePath = "./plots/Features/"
#     Plotting.Overflow = True

#     TwoLep, WTwoLep = [], []
#     ThreeLep, WThreeLep = [], []
#     for i,Sample in enumerate(Samples):
#         idx = Sample.LVariables.index("nLeps")
#         idx2 = Sample.LVariables.index("met_met")
#         for j,Event in enumerate(Sample.Events):
#             if(Event[idx] == 2):
#                 TwoLep.append(Event[idx2])
#                 WTwoLep.append(Samples[i].Weights[j])
#             elif(Event[idx] > 2):
#                 ThreeLep.append(Event[idx2])
#                 WThreeLep.append(Samples[i].Weights[j])

    

#     TwoLep = Plotting.Basics.H1D(np.array(TwoLep)/1000.,30,0,600,Weights=WTwoLep,Norm=True)
#     ThreeLep = Plotting.Basics.H1D(np.array(ThreeLep)/1000.,30,0,600,Weights=WThreeLep,Norm=True)
#     ThreeLep.SetLineColor(2)

#     leg = Plotting.Basics.Legend([TwoLep,ThreeLep],["2 leptons","3 leptons"])
#     Plotting.Basics.Hist1DStack([TwoLep,ThreeLep],"E_{miss} [GeV]","norm. Yield","met_met",Leg=leg,Drawopt="nostack")


# ---------------------------------------------------------------------------------------------------------------------------------- #
def TrafoPlots(Samples):

    Plotting.SavePath = "./plots/Trafo/"
    Plotting.Overflow = True

    TestTrainPlots(Samples,"met_met",20,-5,5,"norm. E_{miss}")
    TestTrainPlots(Samples,"met_phi",20,-5,5,"norm. \phi_{miss}")

    TestTrainPlots(Samples,"jet_pt",20,-5,5,"norm. jet p_{T}")
    TestTrainPlots(Samples,"jet_eta",20,-5,5,"norm. jet \eta")
    TestTrainPlots(Samples,"jet_phi",20,-5,5,"norm. jet \phi")
    TestTrainPlots(Samples,"jet_e",20,-5,5,"norm. jet E")

    TestTrainPlots(Samples,"el_pt",20,-5,5,"norm. e p_{T}")
    TestTrainPlots(Samples,"el_eta",20,-5,5,"norm. e \eta")
    TestTrainPlots(Samples,"el_phi",20,-5,5,"norm. e \phi")
    TestTrainPlots(Samples,"el_e",20,-5,5,"norm. e E")

    TestTrainPlots(Samples,"mu_pt",20,-5,5,"norm. \mu p_{T}")
    TestTrainPlots(Samples,"mu_eta",20,-5,5,"norm. \mu \eta")
    TestTrainPlots(Samples,"mu_phi",20,-5,5,"norm. \mu \phi")
    TestTrainPlots(Samples,"mu_e",20,-5,5,"norm. \mu E")




def TestTrainPlots(Samples,Var,bins,xmin,xmax,XTitle):
    TrafoFeature(Samples,Var,bins,xmin,xmax,XTitle,"train")
    TrafoFeature(Samples,Var,bins,xmin,xmax,XTitle,"test")    


def TrafoFeature(Samples,Var,bins,xmin,xmax,XTitle,SetName):
    YTitle = "norm. Yield"

    index  = Samples[0].LVariables.index(Var)
    for Sample in Samples:
        if(Sample.Names == SetName):
            Set = Sample

    for iSeq in range(len(Samples[0].Events[:,index][0])):
        FourTop      = np.array([Event[iSeq] for i,Event in enumerate(Set.Events[:,index]) if Set.OutTrue[i] == 1])
        ThreeTop     = np.array([Event[iSeq] for i,Event in enumerate(Set.Events[:,index]) if Set.OutTrue[i] == 0])
        FourWeights  = np.array([Samples[0].Weights[i] for i in range(len(Set.Weights)) if Set.OutTrue[i] == 1])
        ThreeWeights = np.array([Samples[1].Weights[i] for i in range(len(Set.Weights)) if Set.OutTrue[i] == 0])

        if(not (np.all(FourTop == 0) and np.all(ThreeTop == 0))):
            FourTop  = Plotting.Basics.H1D(FourTop,bins,xmin,xmax,Weights=FourWeights.flatten(),Norm=True)
            ThreeTop = Plotting.Basics.H1D(ThreeTop,bins,xmin,xmax,Weights=ThreeWeights.flatten(),Norm=True)
            FourTop.SetLineColor(4)
            ThreeTop.SetLineColor(2)

            leg = Plotting.Basics.Legend([ThreeTop,FourTop],["Three Top","Four Top"])
            Plotting.Basics.Hist1DStack([ThreeTop,FourTop],XTitle+" "+str(iSeq),YTitle,SetName,Leg=leg,Drawopt="nostack")
            os.system("mv "+Plotting.SavePath+"StackHist.png "+Plotting.SavePath+SetName+"/"+Var+str(iSeq)+".png")


    