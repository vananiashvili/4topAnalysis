import os,sys
home = os.path.expanduser('~')
sys.path.insert(1,home+'/Desktop/Master_thesis/WorkStation/studies/RestartHadTop/DIClasses')

import Plotting.Basics
import Plotting.Fits
import numpy as np
from DIHadTop import *
from DITruthTop import *

def Chi2(Matches,Weights,MaxdR,mode="Reco"):
    if(mode == "Reco"):
        num = 2
    elif(mode == "BP"):
        num = 1
    Plotting.SavePath = "./plots/Reco/"
    Plotting.Overflow = True
    YTitle = "norm. Yield"

    Chi, ChiCond = [], []
    for Event in Matches:
        EChi, EChiCon = 0, 0
        for Pair in Event:
            if(Pair[0].E != 0 and Pair[2].E != 0):
                EChi += Pair[num].Chi2()
                if(Pair[0].Discrim(Pair[1]) < MaxdR):            # Only look at tops where we now there is a "good" matching
                    EChiCon += Pair[num].Chi2()
        if(len(Event) == 0):
            assert 0 == 1
        Chi.append(EChi)
        ChiCond.append(EChiCon)

    Plotting.Basics.Hist1D(Chi,r"Chi^{2}",YTitle,16,0,16,Norm=True,Weights=Weights)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"Chi2.png")
    Plotting.Basics.Hist1D(ChiCond,r"Chi^{2}",YTitle,16,0,16,Norm=True,Weights=Weights)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"Chi2Condition.png")    

#---------------------------------------------------------------------------------------------------------------

def Comparision1D(Pairs):

    Plotting.SavePath = "./plots/Reco/"
    Plotting.Overflow = True
    Plotting.Underflow = False

    Pairs1, Pairs2 = [], []
    for Event in Pairs:
        if(Event[0][2].E != 0 and Event[1][2].E != 0 ):
            if(Event[0][0].pT > Event[1][0].pT):
                Pairs1.append(Event[0])
                Pairs2.append(Event[1])
            else:
                Pairs2.append(Event[1])
                Pairs1.append(Event[0])
        elif(Event[1][2].E != 0):
            Pairs1.append(Event[0])

    Kinmetics1D(Pairs1,"M",30,0,300,r"M^{Top} [GeV]","Top1")
    Kinmetics1D(Pairs1,"E",40,0,1200,r"E^{Top} [GeV]","Top1")
    Kinmetics1D(Pairs1,"eta",18,-4.5,4.5,r"\eta^{Top}","Top1")
    Kinmetics1D(Pairs1,"phi",12,-3.141,3.141,r"\phi^{Top}","Top1")
    Kinmetics1D(Pairs1,"pT",12,0,600,r"p_{T}^{Top} [GeV]","Top1")
    Kinmetics1D(Pairs1,"WM",20,0,200,r"M^{W} [GeV]","Top1")
    Kinmetics1D(Pairs1,"WpT",10,0,500,r"p_{T}^{W} [GeV]","Top1")
    Kinmetics1D(Pairs1,"Weta",18,-4.5,4.5,r"\eta^{W}","Top1")
    Kinmetics1D(Pairs1,"Wphi",12,-3.141,3.141,r"\phi^{W}","Top1")
    Kinmetics1D(Pairs1,"bM",12,0,60,r"M^{b} [GeV]","Top1")
    Kinmetics1D(Pairs1,"bpT",10,0,500,r"p_{T}^{b} [GeV]","Top1")
    Kinmetics1D(Pairs1,"beta",18,-4.5,4.5,r"\eta^{b}","Top1")
    Kinmetics1D(Pairs1,"bphi",12,-3.141,3.141,r"\phi^{b}","Top1")
    Kinmetics1D(Pairs1,"bscore",10,0,1,"mv2c10","Top1")

    Kinmetics1D(Pairs2,"M",30,0,300,r"M^{Top} [GeV]","Top2")
    Kinmetics1D(Pairs2,"E",40,0,1200,r"E^{Top} [GeV]","Top2")
    Kinmetics1D(Pairs2,"eta",18,-4.5,4.5,r"\eta^{Top}","Top2")
    Kinmetics1D(Pairs2,"phi",12,-3.141,3.141,r"\phi^{Top}","Top2")
    Kinmetics1D(Pairs2,"pT",12,0,600,r"p_{T}^{Top} [GeV]","Top2")
    Kinmetics1D(Pairs2,"WM",20,0,200,r"M^{W} [GeV]","Top2")
    Kinmetics1D(Pairs2,"WpT",10,0,500,r"p_{T}^{W} [GeV]","Top2")
    Kinmetics1D(Pairs2,"Weta",18,-4.5,4.5,r"\eta^{W}","Top2")
    Kinmetics1D(Pairs2,"Wphi",12,-3.141,3.141,r"\phi^{W}","Top2")
    Kinmetics1D(Pairs2,"bM",12,0,60,r"M^{b} [GeV]","Top2")
    Kinmetics1D(Pairs2,"bpT",10,0,500,r"p_{T}^{b} [GeV]","Top2")
    Kinmetics1D(Pairs2,"beta",18,-4.5,4.5,r"\eta^{b}","Top2")
    Kinmetics1D(Pairs2,"bphi",12,-3.141,3.141,r"\phi^{b}","Top2")
    Kinmetics1D(Pairs2,"bscore",10,0,1,"mv2c10","Top2")



def Kinmetics1D(Pairs,Var,bins,xmin,xmax,XTitle,Title):

    YTitle = "norm. #Events"
    Vars = {"E": lambda x:x.E,
        "pT": lambda x:x.pT,
        "eta": lambda x:x.eta,
        "phi": lambda x:x.phi,
        "M": lambda x:x.M,
        "WM": lambda x:x.W.M,
        "WpT": lambda x:x.W.pT,
        "Weta": lambda x:x.W.eta,
        "Wphi": lambda x:x.W.phi,
        "bM": lambda x:x.b.M,
        "bpT": lambda x:x.b.pT,
        "beta": lambda x:x.b.eta,
        "bphi": lambda x:x.b.phi}

    VarsHad = {"bM": lambda x:x.BJet.M,
            "bpT": lambda x:x.BJet.pT,
            "beta": lambda x:x.BJet.eta,
            "bphi": lambda x:x.BJet.phi}

    if(Var == "bscore"):
        Truth = np.ones(len(Pairs))
        Best  = [Pair[1].BJet.mv2c10 for Pair in Pairs if Pair[1].BJet.mv2c10 != 0]
        Reco  = [Pair[2].BJet.mv2c10 for Pair in Pairs if Pair[2].BJet.mv2c10 != 0]
    elif(Var == "M"):
        Truth = []
        for Pair in Pairs:
            if(Pair[0].E != 0):
                LV = Pair[0].W.LV + Pair[0].b.LV
                Truth.append(LV.mass)
        Best  = [Vars[Var](Pair[1]) for Pair in Pairs if Vars[Var](Pair[1]) != 0]
        Reco  = [Vars[Var](Pair[2]) for Pair in Pairs if Vars[Var](Pair[2]) != 0] 

    elif("b" in Var):
        Truth = [Vars[Var](Pair[0]) for Pair in Pairs if Vars[Var](Pair[0]) != 0]
        Best  = [VarsHad[Var](Pair[1]) for Pair in Pairs if VarsHad[Var](Pair[1]) != 0]
        Reco  = [VarsHad[Var](Pair[2]) for Pair in Pairs if VarsHad[Var](Pair[2]) != 0]
    else:
        Truth = [Vars[Var](Pair[0]) for Pair in Pairs if Vars[Var](Pair[0]) != 0]
        Best  = [Vars[Var](Pair[1]) for Pair in Pairs if Vars[Var](Pair[1]) != 0]
        Reco  = [Vars[Var](Pair[2]) for Pair in Pairs if Vars[Var](Pair[2]) != 0]

    Truth = Plotting.Basics.H1D(Truth,bins,xmin,xmax,Norm=True)
    Best  = Plotting.Basics.H1D(Best,bins,xmin,xmax,Norm=True) 
    Reco  = Plotting.Basics.H1D(Reco,bins,xmin,xmax,Norm=True)
    Reco.SetLineColor(2)
    Truth.SetLineColor(1)

    leg = Plotting.Basics.Legend([Truth,Best,Reco],["truth","best-possible",r"\chi^2"])
    Plotting.Basics.HistCombined([Truth,Best,Reco],XTitle,YTitle,Leg=leg,Title=Title)
    os.system("mv "+Plotting.SavePath+"CombinedHist.png "+Plotting.SavePath+Title+Var+".png")

#----------------------------------------------------------------------------------------------------------

def CompdR(Matches,SampleName,MaxdR):

    Plotting.SavePath = "./plots/Reco/"
    Plotting.Overflow = True
    YTitle = "frac. of Events"

    dRBest, dRReco = [], []
    dRCond = []                                             # dR of Reco where dR of Best is smaller than given cut
    for Event in Matches:
        for Pair in Event:
            if(Pair[0].E != 0 and Pair[1].E != 0):
                dRBest.append(Pair[0].Discrim(Pair[1]))
            if(Pair[0].E != 0 and Pair[2].E != 0):
                dRReco.append(Pair[0].Discrim(Pair[2]))
                if(dRBest.append(Pair[0].Discrim(Pair[1])) < MaxdR):
                    dRCond.append(Pair[0].Discrim(Pair[2]))


    Best = Plotting.Basics.H1D(dRBest,30,0,6,Norm=True)
    Reco = Plotting.Basics.H1D(dRReco,30,0,6,Norm=True)
    Best.SetLineColor(2)
    Leg = Plotting.Basics.Legend([Best,Reco],["best-possible",r"\chi^2"],Pos=(0.7,0.8,0.9,0.9))
    Plotting.Basics.HistCombined([Best,Reco],r"Discriminator(Truth,Reco)",YTitle,Leg=Leg,Title=SampleName+" (mc16e)")
    os.system("mv "+Plotting.SavePath+"CombinedHist.png "+Plotting.SavePath+"CompdR.png")

    #Condition Plot
    dRBest = np.array(dRBest)
    dRBest = dRBest[dRBest < MaxdR]
    Best = Plotting.Basics.H1D(dRBest,50,0,6,Norm=True)
    Reco = Plotting.Basics.H1D(dRCond,50,0,6,Norm=True)
    Best.SetLineColor(2)
    Leg = Plotting.Basics.Legend([Best,Reco],["best-possible",r"\chi^2"],Pos=(0.7,0.8,0.9,0.9))
    Plotting.Basics.HistCombined([Best,Reco],r"\sum \Delta R(Truth,Reco)",YTitle,Leg=Leg,Title=SampleName+" (mc16e)")
    os.system("mv "+Plotting.SavePath+"CombinedHist.png "+Plotting.SavePath+"CompdR_Condition.png")

#---------------------------------------------------------------------------------------------------------------

def CompdREvent(Matches,SampleName):
    """ A special plot to compare to Oguls results in addtion dR was changed to dRSum = dR(Top) + dR(W) (in DIHadTop)"""

    Plotting.SavePath = "./plots/Reco/"
    Plotting.Overflow = True
    YTitle = "frac. of Events"

    dRBest, dRReco = [], []
    for Event in Matches:
        dRBP, dRRe = 0, 0
        if(Event[0][0].E != 0 and Event[1][0].E != 0 and Event[0][2].E != 0 and Event[1][2].E != 0):
            for Pair in Event:
                dRBP += Pair[0].Discrim(Pair[1])
                dRRe += Pair[0].Discrim(Pair[2])
            dRBest.append(dRBP)
            dRReco.append(dRRe)

    Best = Plotting.Basics.H1D(dRBest,50,0,20,Norm=True)
    Reco = Plotting.Basics.H1D(dRReco,50,0,20,Norm=True)
    Best.SetLineColor(2)
    Leg = Plotting.Basics.Legend([Best,Reco],["best-possible",r"\chi^2"],Pos=(0.7,0.8,0.9,0.9))
    Plotting.Basics.HistCombined([Best,Reco],r"\sum \Delta R(Truth,Reco)",YTitle,Leg=Leg,Title=SampleName+" (mc16e)")
    os.system("mv "+Plotting.SavePath+"CombinedHist.png "+Plotting.SavePath+"CompdR.png")


#---------------------------------------------------------------------------------------------------------------------------------------
    
def nHadPlots(Matches,LepCharge,SampleName,MaxdR):

    # NoHad, OneHad, TwoHad = 0, 0, 0
    # for Event in Matches:
    #         if(Event[0][2].E == 0 and Event[1][2].E == 0):
    #             NoHad += 1
    #         elif(Event[0][2].E == 0 or Event[1][2].E == 0):
    #             OneHad += 1
    #         else:
    #             TwoHad += 1

    # print(NoHad,OneHad,TwoHad)
    # assert 0 == 1


    Reco = np.array([[Pair[2] for Pair in Event] for Event in Matches])
    Truth = np.array([[Pair[0] for Pair in Event] for Event in Matches])
    # Mask = []
    # for Events in Matches:
    #     flag = 0
    #     for Pair in Events:
    #         if(Pair[0].Discrim(Pair[1]) > MaxdR):
    #             flag = 1
    #     if(flag == 0):
    #         Mask.append(True)
    #     else:
    #         Mask.append(False)

    # Mask = np.array(Mask)
    # RecoCon = Reco[Mask]
    # TruthCon = Truth[Mask]
    # LepChargeCon = LepCharge[Mask]
    
    # nHad(RecoCon,TruthCon,LepChargeCon,SampleName)
    # os.system("mv "+Plotting.SavePath+"nHad.png "+Plotting.SavePath+"nHad_Con.png")
    # os.system("mv "+Plotting.SavePath+"Heat.png "+Plotting.SavePath+"Heat_Con.png")

    nHad(Reco,Truth,SampleName)



def nHad(Reco,Truth,SampleName):

    Plotting.SavePath = "./plots/Reco/"
    nTruthHad = []
    nRecoHad  = []
    bins = {2:0, 1:1, 0:2}            #{'2top':0, '2top':1, 'Notop':2}

    for i,Event in enumerate(Reco):

        nTruth, nReco = 0, 0
        for j in range(len(Reco[i])):
            if(Reco[i][j].E != 0):
                nReco += 1
        for j in range(len(Truth[i])):
            if(Truth[i][j].Had):
                nTruth += 1
        
        nTruthHad.append(bins[nTruth])
        nRecoHad.append(bins[nReco])

    TwoHad, OneHad, NonHad = 0,0,0
    for Event in nRecoHad:
        if(Event == 0):
            TwoHad += 1
        elif(Event == 2):
            NonHad += 1
        elif(Event == 1):
            OneHad += 1
    print(TwoHad, OneHad, NonHad)

    Labels = ["2top","1top","no tops"]

    #nHad
    nTruth = Plotting.Basics.H1D(nTruthHad,3,0,3,Norm=True)
    Plotting.Basics.AlphanumericLabels(nTruth,Labels)
    nTruth.SetLineColor(2)
    nReco  = Plotting.Basics.H1D(nRecoHad,3,0,3,Norm=True)
    Plotting.Basics.AlphanumericLabels(nReco,Labels)
    Leg = Plotting.Basics.Legend([nTruth,nReco],["Truth","Reco"],Pos=(0.4,0.8,0.6,0.9))
    Plotting.Basics.HistCombined([nTruth,nReco],"","frac. of Events",Leg=Leg,Title=SampleName+" (mc16e)")
    os.system("mv "+Plotting.SavePath+"CombinedHist.png "+Plotting.SavePath+"nHad.png")

    #HadMap
    Heat = Plotting.Basics.Hist2D(nTruthHad,nRecoHad,"Number of truth-tops","Number of reco-tops",3,0,3,3,0,3,Norm="x")
    os.system("rm "+Plotting.SavePath+"Hist2D.png")
    Plotting.Basics.AlphanumericLabels(Heat,Labels)
    Plotting.Basics.AlphanumericLabels(Heat,Labels,"y")
    Plotting.Basics.FromHist(Heat,DrawOpt="colz")
    os.system("mv "+Plotting.SavePath+"Redrawn.png "+Plotting.SavePath+"Heat.png")






def CompVars(Matches,dRBest,dRReco,SampleName):

    Difference = np.subtract(dRReco,dRBest)

    CompKin(Matches,Difference,SampleName,"pT",50,0,500,"p_{T}^{RecoTop}")
    CompKin(Matches,Difference,SampleName,"E",50,0,1500,"E^{RecoTop}")
    CompKin(Matches,Difference,SampleName,"eta",50,-2.7,2.7,r"\eta^{RecoTop}")
    CompKin(Matches,Difference,SampleName,"phi",50,-3.141,3.141,r"\phi^{RecoTop}")
    CompKin(Matches,Difference,SampleName,"M",50,0,300,"M^{RecoTop}")
    CompKin(Matches,Difference,SampleName,"WpT",50,0,500,"p_{T}^{RecoW}")
    CompKin(Matches,Difference,SampleName,"WE",50,0,1500,"E^{RecoW}")
    CompKin(Matches,Difference,SampleName,"Weta",50,-2.7,2.7,r"\eta^{RecoW}")
    CompKin(Matches,Difference,SampleName,"Wphi",50,-3.141,3.141,r"\phi^{RecoW}")
    CompKin(Matches,Difference,SampleName,"WM",50,0,150,"M^{RecoW}")



def CompKin(Matches,Difference,SampleName,Var,xbins,xmin,xmax,XTitle):

    Plotting.SavePath = "./plots/Reco/"
    Vars = {"E": lambda x:x.E,
        "pT": lambda x:x.pT,
        "eta": lambda x:x.eta,
        "phi": lambda x:x.phi,
        "M": lambda x:x.M,
        "WM": lambda x:x.W.M,
        "WpT": lambda x:x.W.pT,
        "WE": lambda x:x.W.E,
        "Weta": lambda x:x.W.eta,
        "Wphi": lambda x:x.W.phi}

    Arr    = [Vars[Var](Pair[2]) for Pair in Matches]

    Plotting.Basics.Hist2D(Arr,Difference,XTitle,r"\Delta R_{\chi^2} - \Delta R_{best-possible}",xbins,xmin,xmax,20,0,20)
    os.system("mv "+Plotting.SavePath+"Hist2D.png "+Plotting.SavePath+Var+"vsdR.png")
    #Plotting.Basics.Hist1D(Difference,"\Delta R_{\chi^2} - \Delta R_{best-possible}","frac. of Events",20,0,20,Norm=True)







