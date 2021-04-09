import os
import Plotting.Basics
import numpy as np


def dR(Pairs):

    Plotting.SavePath = "./plots/Matching/"
    Plotting.Overflow = True
    YTitle = "norm. #Events"
    
    ListPairs = []
    for Event in Pairs:
        for Pair in Event:
            ListPairs.append(Pair)

    dR      = [Pair[0].Discrim(Pair[1]) for Pair in ListPairs]
    Plotting.Basics.Hist1D(dR,r"Discrimnator (RecoTop,TruthTop)",YTitle,50,0,6,Stats=False,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"deltaR.png")


#----------------------------------------------------------------------------------------------#

def Comparision1D(Pairs):

    Plotting.SavePath = "./plots/Matching/"
    Plotting.Overflow = True

    Kinmetics1D(Pairs,"M",40,100,250,r"M^{Top} [GeV]")
    Kinmetics1D(Pairs,"E",40,180,1200,r"E^{Top} [GeV]")
    Kinmetics1D(Pairs,"eta",40,-2.7,2.7,r"\eta^{Top}")
    Kinmetics1D(Pairs,"phi",40,-3.141,3.141,r"\phi^{Top}")
    Kinmetics1D(Pairs,"pT",40,0,800,r"p_{T}^{Top} [GeV]")
    Kinmetics1D(Pairs,"WM",40,30,300,r"M^{W} [GeV]")
    Kinmetics1D(Pairs,"WpT",40,0,800,r"p_{T}^{W} [GeV]")
    Kinmetics1D(Pairs,"Weta",40,-2.7,2.7,r"\eta^{W}")
    Kinmetics1D(Pairs,"Wphi",40,-3.141,3.141,r"\phi^{W}")



def Kinmetics1D(Pairs,Var,bins,xmin,xmax,XTitle):

    YTitle = "norm. #Events"
    Vars = {"E": lambda x:x.E,
        "pT": lambda x:x.pT,
        "eta": lambda x:x.eta,
        "phi": lambda x:x.phi,
        "M": lambda x:x.M,
        "WM": lambda x:x.W.M,
        "WpT": lambda x:x.W.pT,
        "Weta": lambda x:x.W.eta,
        "Wphi": lambda x:x.W.phi}

    Truth = [Vars[Var](Pair[0]) for Pair in Pairs]
    Reco = [Vars[Var](Pair[1]) for Pair in Pairs]

    Truth = Plotting.Basics.H1D(Truth,bins,xmin,xmax,Norm=True)
    Truth.SetLineColor(2)
    Reco  = Plotting.Basics.H1D(Reco,bins,xmin,xmax,Norm=True)

    leg = Plotting.Basics.Legend([Truth,Reco],["truth","best possible"])
    Plotting.Basics.HistCombined([Truth,Reco],XTitle,YTitle,Leg=leg)
    os.system("mv "+Plotting.SavePath+"CombinedHist.png "+Plotting.SavePath+Var+".png")

# ------------------------------------------------------------------------------------------- #

def Comparision2D(Pairs):

    Plotting.SavePath = "./plots/Matching/"
    Plotting.Overflow = True

    #Kinmetics2D(Pairs,"M",120,230,120,230,r"M^{Top} [GeV]")
    Kinmetics2D(Pairs,"E",40,40,180,50,40,180,r"E^{Top} [GeV]")
    Kinmetics2D(Pairs,"eta",40,-2.7,2.7,40,-2.7,2.7,r"\eta^{Top}")
    Kinmetics2D(Pairs,"phi",40,-3.141,3.141,40,-3.141,3.141,r"\phi^{Top}")
    Kinmetics2D(Pairs,"pT",40,0,800,40,0,800,r"p_{T}^{Top} [GeV]")
    Kinmetics2D(Pairs,"WM",40,30,130,40,30,130,r"M^{W} [GeV]")
    Kinmetics2D(Pairs,"WpT",40,0,800,40,0,800,r"p_{T}^{W} [GeV]")
    Kinmetics2D(Pairs,"Weta",40,-2.7,2.7,40,-2.7,2.7,r"\eta^{W}")
    Kinmetics2D(Pairs,"Wphi",40,-3.141,3.141,40,-3.141,3.141,r"\phi^{W}")






def Kinmetics2D(Pairs,Var,xbins,xmin,xmax,ybins,ymin,ymax,XTitle,log=False):

    YTitle = "Truth "+XTitle
    XTitle = "Reco "+XTitle
    Vars = {"E": lambda x:x.E,
        "pT": lambda x:x.pT,
        "eta": lambda x:x.eta,
        "phi": lambda x:x.phi,
        "M": lambda x:x.M,
        "WM": lambda x:x.W.M,
        "WpT": lambda x:x.W.pT,
        "Weta": lambda x:x.W.eta,
        "Wphi": lambda x:x.W.phi}

    Truth = [Vars[Var](Pair[0]) for Pair in Pairs]
    Reco = [Vars[Var](Pair[1]) for Pair in Pairs] 

    Plotting.Basics.Hist2D(Truth,Reco,XTitle,YTitle,xbins,xmin,xmax,ybins,ymin,ymin)
    os.system("mv "+Plotting.SavePath+"Hist2D.png "+Plotting.SavePath+Var+"_2D.png")





# ------------------------------------------------------------------------------ #
#Jet Based Matching

def GetPTvsR(Matches):
    firstWdR, secondWdR, bdR = [], [], []
    for Event in Matches:
        if(len(Event) not in [0,3,6]):
            print('incorrect length')
            assert 0 == 1
        if(len(Event) == 0):
            pass
        if(len(Event) >= 3):
            if(Event[0][1].pT > Event[1][1].pT):
                firstWdR.append(Event[0])
                secondWdR.append(Event[1])
            else:
                firstWdR.append(Event[1])
                secondWdR.append(Event[0])
            bdR.append(Event[2])
        if(len(Event) == 6):
            if(Event[3][1].pT > Event[4][1].pT):
                firstWdR.append(Event[3])
                secondWdR.append(Event[4])
            else:
                firstWdR.append(Event[4])
                secondWdR.append(Event[3])
            bdR.append(Event[5])

    return firstWdR, secondWdR, bdR


def MakeEffPlots(Matches,TopPairs,Object,TopCut=0.9):
    if(Object == "Jet"):
        num = 1
        Title = "Reco "
    elif(Object == "Quark"):
        num = 0
        Title = "Truth "

    #Tops
    # dRvspT = []
    # for Event in TopPairs:
    #     for Pair in Event:
    #         if(Pair[1].E != 0):
    #             dRvspT.append([Pair[num].phi,Pair[0].SumdR(Pair[1])])
    # pTeff(dRvspT,62,-3.141,3.141,tag=Title+"Top",Cut=TopCut)
    # os.system("mv "+Plotting.SavePath+"Redrawn.png "+Plotting.SavePath+"eff_Tops.png")
    # Plotting.MatchingPlots.dR(TopPairs)

    #Sub structure
    # firstWdR, secondWdR, bdR = GetPTvsR(Matches)
    # dRvspT = [[Pair[num].pT,Pair[0].DeltaR(Pair[1])] for Pair in firstWdR]
    # pTeff(dRvspT,20,0,600,tag="leading W "+Title+Object)
    # os.system("mv "+Plotting.SavePath+"Redrawn.png "+Plotting.SavePath+"eff_firstWJet.png")
    # dRvspT = [[Pair[num].pT,Pair[0].DeltaR(Pair[1])] for Pair in secondWdR]
    # pTeff(dRvspT,20,0,600,tag="sub-leading W "+Title+Object)
    # os.system("mv "+Plotting.SavePath+"Redrawn.png "+Plotting.SavePath+"eff_secondWJet.png")
    # dRvspT = [[Pair[num].pT,Pair[0].DeltaR(Pair[1])] for Pair in bdR]
    # pTeff(dRvspT,20,0,600,tag="b "+Title+Object)
    # os.system("mv "+Plotting.SavePath+"Redrawn.png "+Plotting.SavePath+"eff_bJet.png")


def pTeff(dRvspT,bins,xmin,xmax,tag="",Cut=0.4):
    Plotting.SavePath = "./plots/Matching/"
    dRvspT = np.array(dRvspT)

    In = [kin[0] for kin in dRvspT if kin[1] < Cut]
    In  = Plotting.Basics.H1D(In,bins,xmin,xmax)
    All = Plotting.Basics.H1D(dRvspT[:,0],bins,xmin,xmax)
    for i in range(In.GetNbinsX() + 1):
        if(All.GetBinContent(i) == 0):
            In.SetBinContent(i,0)
        else:
            In.SetBinContent(i, In.GetBinContent(i) / All.GetBinContent(i))

    In.GetXaxis().SetTitle(tag+" \phi")
    In.GetYaxis().SetTitle("truth-matching eff.")
    In.SetMarkerStyle(5)
    Plotting.Basics.FromHist(In,DrawOpt="P")