import Plotting.Basics
import numpy as np
import os
from DIJet import *
from DIHadTop import *

def nW(BasicTruth):
    #Find the Events that have a W
    nW = []
    for i in range(len(BasicTruth)):
        Ws = 0
        for Id in BasicTruth[i]:
            if(abs(Id) == 24):
                Ws += 1
        nW.append(Ws)

    nW = np.array(nW)

    Plotting.SavePath = './plots/TruthInv/'
    Plotting.Basics.Hist1D(nW,'number of Ws','#Events',10,0,10,Norm=True)


def nVectorBosons(BasicTruth):

    Plotting.SavePath = './plots/TruthInv/'
    Plotting.Basics.Hist1D(BasicTruth.ravel(),'number of Vector Bosons','#Events',3,0,3,Norm=True)

def nHadTops(Tops,Weights,Sample):
    nTops = []
    twohad = 0.
    for i in range(len(Tops)):
        Ts = 0
        for Top in Tops[i]:
            if(Top.Had):
                Ts += 1
        nTops.append(Ts)
        if(Ts == 2):
            twohad += 1
    Plotting.SavePath = './plots/Truth/'+Sample+"/"
    Plotting.Basics.Hist1D(nTops,'number of had. Tops','norm. Yield',4,0,4,Title=Sample+" (mc16e)",Weights=Weights,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"nHadTop.png")

    print("{0:.3f} percent of the events have two had. Tops".format(twohad/len(Tops)))


def HadOnly(Tops,Childs):

    Tops   = Tops.ravel()
    Childs = Childs.reshape(len(Tops),Childs.shape[2])
    # Only take hadronic tops
    Childs = np.array([Childs[i] for i in range(len(Childs)) if Tops[i].Had])
    Tops   = np.array([Tops[i] for i in range(len(Tops)) if Tops[i].Had])

    return Tops, Childs


def dRWb(Childs,Tops):
   
    dR = []
    for i in range(len(Childs)):
        for j in range(4):
            for k in range(4):
                if(Tops[i][j].Had and k != j):
                    dR.append(Childs[i][j][0].LV.deltar(Childs[i][k][1].LV))

    Plotting.SavePath = './plots/TruthInv/'
    Plotting.Overflow = True     
    Plotting.Basics.Hist1D(dR,r'\Delta R(W,b)','#Events',12,0,6,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"dRWb_Random.png")

    wanted = len(dR) - int(round(len(dR) * 0.68))
    dR.sort()
    dR.reverse()
    print(dR[wanted])

def HadWMass(Childs,Tops):
   
    WM = []
    for i in range(len(Childs)):
        for j in range(4):
            if(Tops[i][j].Had):
                LVW = Childs[i][j][0].Jets[0].LV + Childs[i][j][0].Jets[1].LV

                WM.append(Childs[i][j][0].M)

    Plotting.SavePath = './plots/TruthInv/'
    Plotting.Overflow = True     
    Plotting.Basics.Hist1D(WM,r'M_{W}','#Events',80,0,120,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"WMass.png")


def HadTopMass(Tops):

    MTop = []
    for i in range(len(Tops)):
        for j in range(4):
            if(Tops[i][j].Had):
                MTop.append(Tops[i][j].M)
        
    Plotting.SavePath = './plots/TruthInv/'
    Plotting.Overflow = True     
    Plotting.Basics.Hist1D(MTop,r'M_{Top}','#Events',40,150,190,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"HadTopdMass.png")    


def HadTopMassChild(Childs,Tops):

    assert len(Childs) == len(Tops)

    MTop = []
    for i in range(len(Childs)):
        for j in range(4):
            if(Tops[i][j].Had):
                TopLV = Childs[i][j][0].Jets[0].LV + Childs[i][j][0].Jets[1].LV + Childs[i][j][1].LV
                MTop.append(TopLV.mass)
        
    Plotting.SavePath = './plots/TruthInv/'
    Plotting.Overflow = True     
    Plotting.Basics.Hist1D(MTop,r'M_{Top}','#Events',200,0,200,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"HadTopChildMass.png") 


def CompareChildTop(Tops,Childs,weights,index,Sample):

    assert len(Tops) == len(Childs)
    assert len(Childs) == len(weights)

    Name = {0:"top1", 1:"top2", 2:"tbar1", 3:"tbar2"}
    MChilds, MTops, Weights = [], [], []
    for i in range(len(Tops)):
        if(Tops[i][index].Had):
            TopLV = Childs[i][index][0].Jets[0].LV + Childs[i][index][0].Jets[1].LV + Childs[i][index][1].LV
            MChilds.append(TopLV.mass)
            MTops.append(Tops[i][0].M)
            Weights.append(weights[i])


    Plotting.SavePath = './plots/TruthInv/'
    Plotting.Overflow = True
    Weights = np.array(Weights).ravel()
    MTops   = Plotting.Basics.H1D(MTops,50,0,200,Weights=Weights,Norm=True)
    MChilds = Plotting.Basics.H1D(MChilds,50,0,200,Weights=Weights,Norm=True)
    MTops.SetLineColor(4)
    MChilds.SetLineColor(2)
    leg = Plotting.Basics.Legend([MTops,MChilds],["truth_"+Name[index],"truth_"+Name[index]+"_childs"],Pos=(0.1,0.8,0.4,0.9))
    Plotting.Basics.HistCombined([MTops,MChilds],r'M_{'+Name[index]+'}','norm. Yield',Title=Sample+' (mc16e)',Leg=leg,Logy=True)
    os.system("mv "+Plotting.SavePath+"CombinedHist.png "+Plotting.SavePath+"HadMassComp"+Name[index]+"_"+Sample+".png")

# ------------------------------------------------------------------------------------------------------- #

def Kinematic(Tops, Weights,Sample,Kind='HadOnly'):

    Types = {'HadOnly': HadOnlyVars,
            'LepHad': LepHadVars}

    Plotting.SavePath = "./plots/Truth/"+Sample+"/"
    Plotting.Overflow = True
    YTitle = "norm. #Events"

    Types[Kind](Tops,"tbar1", 0,Weights)
    Types[Kind](Tops,"tbar2", 1,Weights)
    Types[Kind](Tops,"top1", 2,Weights)
    Types[Kind](Tops,"top2", 3,Weights)

# ------------------------------------------------------------------------------------------------------ #
""" LepHad Plots"""

def LepHadVars(Tops,Name,index,Weights):
    
    LepHadStacked(Tops,Name,index,"E",40,180,1200,r"E^{"+Name+"} [GeV]",Weights)
    LepHadStacked(Tops,Name,index,"M",40,150,190,r"M^{"+Name+"} [GeV]",Weights)
    LepHadStacked(Tops,Name,index,"eta",40,-2.7,2.7,r"\eta^{"+Name+"}",Weights)
    LepHadStacked(Tops,Name,index,"phi",40,-3.141,3.141,r"\phi^{"+Name+"}",Weights)
    LepHadStacked(Tops,Name,index,"pT",40,0,800,r"p_{T}^{"+Name+"}",Weights)


def LepHadStacked(Tops,Name,index,Var,bins,xmin,xmax,XTitle,Weights):

    Vars = {"E": lambda x:x.E,
            "pT": lambda x:x.pT,
            "eta": lambda x:x.eta,
            "phi": lambda x:x.phi,
            "M": lambda x:x.M}

    E = np.array([Top.E for Top in Tops[:,index]])
    Weights = Weights[E != 0]
    Tops = [Top for Top in Tops[:,index] if Top.E != 0]

    Arr = [Vars[Var](Top) for Top in Tops if Top.Had == True]
    WHad = [Weight for i,Weight in enumerate(Weights) if Tops[i].Had == True]
    Had = Plotting.Basics.H1D(Arr,bins,xmin,xmax,Weights=WHad,Norm=True)
    Had.SetFillColor(2)

    Arr = [Vars[Var](Top) for Top in Tops if Top.Had == False]
    WLep = [Weight for i,Weight in enumerate(Weights) if Tops[i].Had == False]
    Lep = Plotting.Basics.H1D(Arr,bins,xmin,xmax,Weights=WLep,Norm=True)
    Lep.SetFillColor(8)
    
    Leg = Plotting.Basics.Legend([Lep,Had],["Leptonic","Hadronic"])
    Plotting.Basics.Hist1DStack([Lep,Had],XTitle,"Norm. Yield","",Leg)
    os.system("mv "+Plotting.SavePath+"StackHist.png "+Plotting.SavePath+Name+Var+".png")


# ---------------------------------------------------------------------------------------- #
""" Had Only Plots """

def HadOnlyVars(Tops,Name,index,Weights):
    
    HadOnlyKin(Tops,Name,index,"E",40,180,1200,r"E^{"+Name+"} [GeV]",Weights)
    HadOnlyKin(Tops,Name,index,"M",40,150,190,r"M^{"+Name+"} [GeV]",Weights)
    HadOnlyKin(Tops,Name,index,"eta",40,-2.7,2.7,r"\eta^{"+Name+"}",Weights)
    HadOnlyKin(Tops,Name,index,"phi",40,-3.141,3.141,r"\phi^{"+Name+"}",Weights)
    HadOnlyKin(Tops,Name,index,"pT",40,0,800,r"p_{T}^{"+Name+"}",Weights)


def HadOnlyKin(Tops,Name,index,Var,bins,xmin,xmax,XTitle,Weights):

    Vars = {"E": lambda x:x.E,
            "pT": lambda x:x.pT,
            "eta": lambda x:x.eta,
            "phi": lambda x:x.phi,
            "M": lambda x:x.M}

    E = np.array([Top.E for Top in Tops[:,index]])
    Weights = Weights[E != 0]
    Tops = [Top for Top in Tops[:,index] if Top.E != 0]

    if(Var == "M"):
        Arr = []
        for Top in Tops:
            LV = Top.W.LV + Top.b.LV
            Arr.append(LV.mass)
    else:
        Arr = [Vars[Var](Top) for Top in Tops if Top.Had == True]
    WHad = [Weight for i,Weight in enumerate(Weights) if Tops[i].Had == True]
    Had = Plotting.Basics.Hist1D(Arr,XTitle,'norm. Yield',bins,xmin,xmax,Weights=WHad,Norm=True,Title=Plotting.SavePath[-6:-1]+'(mc16e) '+Name)

    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+Name+Var+".png")
