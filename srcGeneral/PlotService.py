import ROOT
import PlotSetup
import numpy as np
import scipy
import ScorePlots
import re, sys, os
from DIClasses import DISample
from sklearn.metrics import roc_curve
from Utils import roc_auc_score
from Utils import stderr, stdwar, stdinfo, dlen, ContainsList
from root_numpy import fill_hist

def Score(path, OutPreOther, OutPreSame, DataSet, Name):
    return ScorePlots.Score(path, OutPreOther, OutPreSame, DataSet, Name)

def MultiScore(path, OutPreOther, OutPreSame, DataSet, Name, ClassNum):
    ScorePlots.MultiScore(path, OutPreOther, OutPreSame, DataSet, Name, ClassNum)



def VarHist(Data, Sig,path,key,name,XTitle,xmin,xmax,bins,Norm=False,pos=''):
    """ Plots the variable as a Stackplot and adds a scaled Sig (as red Line) """
    if(not isinstance(Data,DISample)):
        LVars = Data.LVariables
        Data = PlotSetup.SetInputArr(Data,key,Sig)
    else:
        LVars = Data.LVariables

    Color = {13:ROOT.kGray,12:ROOT.kBlack,11:ROOT.kGray+1,10:ROOT.kGray+2,             #Colorcoding
    2:ROOT.kAzure+1,3:ROOT.kYellow-7,4:ROOT.kGreen-5,1:ROOT.kGreen-1,       
    9:ROOT.kGreen-10,8:ROOT.kPink-9,7:ROOT.kViolet-4,
    6:ROOT.kBlue-7,5:ROOT.kBlue-10,0:ROOT.kRed+1}
    Names = {0:r't\bar{t}t\bar{t}',13:r't\bar{t}W',12:r't\bar{t}WW',11:r't\bar{t}Z',10:r't\bar{t}H',2:'V+jets'
        ,3:'VV',4:r't(\bar{t})X',1:'others',9:r't\bar{t} QmisID',8:r't\bar{t} CO',
        7:r't\bar{t} HF',6:r't\bar{t} light',5:r't\bar{t} other'}                                                                                   # Use for scaling Sig to Bkg (line)
    ListOfHists = {}                                                                                # Work around  (Pyroot GetHists has a bug)
    Yield       = {}                                                                   
    if(name in LVars):                                                            # Only do the Plot if it is used later on
        hs = ROOT.THStack("hs","")
        
        #Loop over all Sig and Bkgs
        Order = [1,2,3,4,5,6,7,8,9,10,11,12,13,0]
        for Class in Order:                          #[::-1] ttt should be on top of the stack plot and therefore the last in the list
            Events = PlotSetup.GetBrachArr(Data,LVars.index(name),Class,XTitle)

            #Create hist for Events and add to hstack
            hist = ROOT.TH1F("hist"+str(Class)+XTitle,"hstack",bins,xmin,xmax)
            fill_hist(hist,Events,weights=Data.Weights[Data.MultiClass == Class])
            hist.SetFillColor(Color[Class])
            hist.SetLineWidth(1)
            hist.SetLineColor(1)
            SetOverflow(hist)
            Yield.update({Class:hist.Integral()})
            ListOfHists.update({Class:hist})
            hs.Add(hist,"Hist")

            #Create Linehist for Sig scaled to Bkg    
            if(Class == 0):                                                                 #if Class == Signal Tupe
                h1 = ROOT.TH1F("h1","Sig",bins,xmin,xmax)
                fill_hist(h1,Events,weights=Data.Weights[Data.MultiClass == Class])
                h1.Scale(sum(Yield.values())/h1.Integral())       
                h1.SetLineColor(2)
                h1.SetLineWidth(2)
                SetOverflow(h1)

        #Set Y axis norm       
        if(Norm == True):
            TotalNorm = 0
            for key in ListOfHists:
                TotalNorm += ListOfHists[key].Integral()
            for key in ListOfHists:
                ListOfHists[key].Scale(1/TotalNorm)
            h1.Scale(1/TotalNorm)

        leg = ROOT.TLegend(0.0,0.1,0.99,0.9)                    #creating the legend in the correct order
        for Class in np.unique(Data.MultiClass):
            idx = (Order[::-1])[int(Class)]
            leg.AddEntry(ListOfHists[idx],Names[idx]+" (Yield: "+"{:04.2f}".format(Yield[idx])+")")
        leg.AddEntry(h1,"tttt normalized to total bkg")

        #Create Canvas and draw everything
        cs = ROOT.TCanvas("cs","cs",1000,600)
        pad1 = ROOT.TPad("pad1","HistPad",0.02,0.02,0.80,0.97)
        pad2 = ROOT.TPad("pad2","LegPad",0.80,0.02,1.0,0.97)
        pad1.Draw()
        pad2.Draw()
        pad1.cd()
        pad1.SetRightMargin(0.02)
        hs.Draw()
        h1.Draw("SameHist")
        if(h1.GetMaximum() > hs.GetMaximum()):
            hs.SetMaximum(h1.GetMaximum()*1.1)
        
        if('Seq' in XTitle):
            XTitle = str(XTitle[4:])
        
        hs.GetXaxis().SetTitle(XTitle)
        if(Norm == False):
            hs.GetYaxis().SetTitle("Yield")
        pad2.cd()
        leg.SetTextSize(0.07)
        leg.Draw()
        if(pos == ''):
            cs.SaveAs(path+name+".png")
        else:
            cs.SaveAs(path+name+"_"+str(pos)+'.png')   


def TotalWeightPlot(Weights, name, max):
    c1 = ROOT.TCanvas("c1","Canvas",700,500)
    ROOT.gStyle.SetOptStat(0)

    h1 = ROOT.TH1F("hist", name, 100, 0, max)
    fill_hist(h1, Weights)
    h1.GetXaxis().SetTitle("Weights")
    h1.GetYaxis().SetTitle("Number of events")
    h1.Draw('Hist')

    c1.Update()
    c1.SaveAs("./plots/"+name+".png")


def LossFunction(path,history,n_epoch,tag=''):
    history_dict = history.history
    loss_values = history_dict['loss']
    test_loss_values = history_dict['val_loss']

    epochs = []
    for i in range(n_epoch):
        epochs.append(float(i)) 
        
    ROOT.gStyle.SetOptFit()
    c1 = ROOT.TCanvas("c1","Neural Network output",700,500)
    c1.SetGrid()

    mg = ROOT.TMultiGraph()

    gr1 = ROOT.TGraph(n_epoch,scipy.array(epochs),scipy.array(loss_values))
    gr1.SetMarkerColor(4)
    gr1.SetMarkerStyle(21)
    mg.Add(gr1,"P")

    gr2 = ROOT.TGraph(n_epoch,scipy.array(epochs),scipy.array(test_loss_values))
    gr2.SetMarkerColor(2)
    gr2.SetMarkerStyle(21)
    mg.Add(gr2,"P")

    mg.Draw("A")
    mg.GetXaxis().SetTitle("Epoch")
    mg.GetYaxis().SetTitle("Loss")

    leg = ROOT.TLegend(0.7,0.7,0.9,0.9)
    leg.AddEntry(gr1,"Loss on train sample")
    leg.AddEntry(gr2,"Loss on test sample")
    leg.Draw()

    Cl = ROOT.gROOT.GetListOfCanvases()
    Cl.Draw()

    c1.SaveAs("plots/LossHist"+tag+".png")


def RocCurve(path,OutPre,DataSet,ModelNames):

    c1 = ROOT.TCanvas("c1","Canvas",800,600)
    leg = ROOT.TLegend(0.1,0.1,0.3,0.3)
    mg = ROOT.TMultiGraph()
    c1.SetGrid()
    if(not isinstance(OutPre,list)):
        OutPre = [OutPre]
    for i,Name in enumerate(ModelNames):
        train, test = DataSet.GetInput(Name)
        OutTrue = test.OutTrue
        weights = test.Weights       
        fpr, tpr, threshold = roc_curve(OutTrue, OutPre[i], sample_weight=weights)
        modelAuc = roc_auc_score(OutTrue, OutPre[i], sample_weight=weights)

        gr1 = ROOT.TGraph(len(tpr),1-fpr,tpr)
        gr1.SetLineColor(1+i)
        gr1.SetLineWidth(2)
        mg.Add(gr1,"L")
        leg.AddEntry(gr1,ModelNames[i]+" (Auc={:05.3f})".format(modelAuc))

    mg.Draw("A")
    leg.Draw()
    mg.GetXaxis().SetTitle("Signal efficiency")
    mg.GetYaxis().SetTitle("Background rejection")
    mg.GetHistogram().GetXaxis().SetRangeUser(0.2,1)
    mg.GetHistogram().GetYaxis().SetRangeUser(0.2,1)

    c1.SaveAs(path+"RocCurve.png")

def VarCrossCheck(Sig,Bkg,SigW,BkgW,name,xmin,xmax,bins):

    hSig = ROOT.TH1F("hSig",name,bins,xmin,xmax)
    fill_hist(hSig,Sig,weights=SigW)
    hSig.SetLineColor(2)
    hSig.SetLineWidth(3)
    SetOverflow(hSig)
    hSig.Scale(np.sum(BkgW)/hSig.Integral())
    hBkg = ROOT.TH1F("hBkg",name,bins,xmin,xmax)
    fill_hist(hBkg,Bkg,weights=BkgW)
    hBkg.SetLineWidth(3)
    SetOverflow(hBkg)

    c1 = ROOT.TCanvas("c1","c1",800,600)
    ROOT.gStyle.SetOptStat(0)
    hBkg.Draw("Hist")
    hSig.Draw("SameHist")

    if(hSig.GetMaximum() > hBkg.GetMaximum()):
        hBkg.SetMaximum(int(round(hSig.GetMaximum()*1.1)))
    hBkg.GetXaxis().SetTitle("Jet multiplicity")
    hBkg.GetYaxis().SetTitle("Yield")

    leg = ROOT.TLegend(0.7,0.7,0.9,0.9)
    leg.AddEntry(hSig,"Sig (Yield: {:04.2f})".format(np.sum(SigW)))
    leg.AddEntry(hBkg,"Bkg (Yield: {:04.2f})".format(np.sum(BkgW)))
    leg.Draw()

    c1.Update()
    c1.SaveAs("./plots/VarCrossCeck.png")

def SBVar(Savepath,Var,DISample,XTitle,bins,xmin,xmax,tag='',Norm=False):
    """ Plots Background and Signal as a StackPlot """
    if(Norm == True):
        stdwar("This needs to be implemented!!!")   
        assert 0 == 1

    hs = ROOT.THStack("hs",Var+tag)
    hSig = ROOT.TH1F("hSig","hstack",bins,xmin,xmax)
    hBkg = ROOT.TH1F("hBkg","hstack",bins,xmin,xmax)
    Sig  = DISample.Events[DISample.OutTrue == 0].ravel()
    SigW = DISample.Weights[DISample.OutTrue == 0] 
    Bkg  = DISample.Events[DISample.OutTrue == 1].ravel()
    BkgW = DISample.Weights[DISample.OutTrue == 1]

    if(XTitle[-5:] == "[GeV]"):                                                              # Convert MeV to GeV
        Sig = Sig / 1000
        Bkg = Bkg / 1000

    fill_hist(hSig,Sig,weights=SigW)
    fill_hist(hBkg,Bkg,weights=BkgW)
    hSig.SetFillColor(4)
    hBkg.SetFillColor(2)
    hSig.SetLineWidth(1)
    hBkg.SetLineWidth(1)
    SetOverflow(hSig)
    SetOverflow(hBkg)
    hs.Add(hBkg,"Hist")
    hs.Add(hSig,"Hist")

    leg = ROOT.TLegend(0.75,0.75,0.9,0.9)
    leg.AddEntry(hSig,"Signal")
    leg.AddEntry(hBkg,"Background")

    cs = ROOT.TCanvas("cs","cs",800,600)
    hs.Draw()
    hs.SetMaximum(hs.GetMaximum()*1.1)
    hs.GetXaxis().SetTitle(XTitle)
    leg.Draw()
    cs.SaveAs(Savepath+Var+tag+'.png')

def NegativeWeightsBkg(col,train,test,vali,XTitle,bins,xmin,xmax,tag=''):

    Label = {'others':2,'vjets':3,'vv':4,'singletop':5,'ttother':6,'ttlight':7,'ttHF':8,
             'ttCO':9,'ttQmisID':10,'ttH':11,'ttZ':12,'ttWW':13,'ttW':14}

    if(tag != 'All'):
        lw    = [train.Weights[train.MultiClass == Label[tag]], test.Weights[test.MultiClass == Label[tag]], vali.Weights[vali.MultiClass == Label[tag]]]
        w     = np.concatenate(lw,axis=0)
        lx    = [train.Events[train.MultiClass == Label[tag]], test.Events[test.MultiClass == Label[tag]], vali.Events[vali.MultiClass == Label[tag]]]
        x     = np.concatenate(lx, axis=0)[:,col]
    else:
        lw    = [train.Weights[train.OutTrue == 0], test.Weights[test.OutTrue == 0], vali.Weights[vali.OutTrue == 0]]
        w     = np.concatenate(lw,axis=0)
        lx    = [train.Events[train.OutTrue == 0], test.Events[test.OutTrue == 0], vali.Events[vali.OutTrue == 0]]
        x     = np.concatenate(lx, axis=0)[:,col]
    
    wabs  = np.where(w > 0, w, abs(w))
    wzero = np.where(w > 0, w, 0)
    #wone  = np.where(w > 0, w, 1)

    c1 = ROOT.TCanvas("c1","Canvas",800,600)
    ROOT.gStyle.SetOptStat(0)

    hw     =  ROOT.TH1F("h1", tag,bins,xmin,xmax)
    hwabs  =  ROOT.TH1F("h2", "",bins,xmin,xmax)
    hwzero =  ROOT.TH1F("h3", "",bins,xmin,xmax)

    HistNW(hw,x,w,1)
    HistNW(hwabs,x,wabs,3)
    HistNW(hwzero,x,wzero,7)

    hw.GetYaxis().SetTitle('Yield')
    hw.GetXaxis().SetTitle(XTitle)
    hw.GetYaxis().SetRangeUser(0,hw.GetMaximum()*1.2)
    hw.SetLineWidth(3)

    hw.Draw("Hist")
    hwabs.Draw("HistSame")
    hwzero.Draw("HistSame")
    #hwone.Draw("HistSame")

    leg = ROOT.TLegend(0.6,0.75,0.9,0.9)
    leg.AddEntry(hw,"with negative weights")
    leg.AddEntry(hwabs,"|weight|")
    leg.AddEntry(hwzero,"negative weights to 0")
    #leg.AddEntry(hwone,"negative weights to 1")
    leg.Draw()

    c1.SaveAs("./plots/nWeightsBkg"+tag+".png")

def NegativeWeightsSig(LO,LOW,NLO,NLOW,XTitle,bins,xmin,xmax):

    wabs  = np.where(NLOW > 0, NLOW, abs(NLOW))
    wzero = np.where(NLOW > 0, NLOW, 0)

    c1 = ROOT.TCanvas("c1","Canvas",800,600)
    ROOT.gStyle.SetOptStat(0)

    hLO      = ROOT.TH1F("h1", "",bins,xmin,xmax)
    hNLO     = ROOT.TH1F("h2", "",bins,xmin,xmax)
    hNLOabs  = ROOT.TH1F("h3", "",bins,xmin,xmax)
    hNLOzero = ROOT.TH1F("h4", "",bins,xmin,xmax)

    HistNW(hLO,LO,LOW,1)
    HistNW(hNLO,NLO,NLOW,2)
    HistNW(hNLOabs,NLO,wabs,3)
    HistNW(hNLOzero,NLO,wzero,7)

    hLO.GetYaxis().SetTitle('Yield')
    hLO.GetXaxis().SetTitle(XTitle)
    hLO.GetYaxis().SetRangeUser(0,hLO.GetMaximum()*1.2)
    #hLO.SetLineWidth(3)

    hLO.Draw("Hist")
    hNLO.Draw("HistSame")
    hNLOabs.Draw("HistSame")
    hNLOzero.Draw("HistSame")

    leg = ROOT.TLegend(0.6,0.75,0.9,0.9)
    leg.AddEntry(hLO,"LO")
    leg.AddEntry(hNLO,"NLO")
    leg.AddEntry(hNLOabs,"NLO negative weights to 0")
    leg.AddEntry(hNLOzero,"NLO negative weights to 1")
    leg.Draw()

    c1.SaveAs("./plots/nWeightsSig.png")

def HistNW(hist, Events, Weights, Color):
    fill_hist(hist, Events, weights=Weights)
    hist.SetLineColor(Color)
    hist.SetLineWidth(1)
    SetOverflow(hist)
    SetUnderflow(hist)
    if(hist.Integral() != 0):
        hist.Scale(1/hist.Integral())

def SigBkgHist(Sample,XTitle,bins,xmin,xmax,tag=''):
    x = Sample.Events
    Sig = x[Sample.OutTrue == 1]                # Signal values of each event
    wSig = Sample.Weights[Sample.OutTrue == 1]  # Signal weights of each event
    Bkg =x[Sample.OutTrue == 0]                 # Background values of each event
    wBkg = Sample.Weights[Sample.OutTrue == 0]  # Background weights of each event

    c1 = ROOT.TCanvas("c1","Canvas",800,600)
    ROOT.gStyle.SetOptStat(0)

    hSig = ROOT.TH1F("hSig", "",bins,xmin,xmax)
    fill_hist(hSig,Sig,weights=wSig)
    SetOverflow(hSig)
    SetUnderflow(hSig)
    hBkg = ROOT.TH1F("hBkg", "",bins,xmin,xmax)
    fill_hist(hBkg,Bkg,weights=wBkg)
    SetOverflow(hBkg)
    SetUnderflow(hBkg)
    hBkg.SetLineColor(2)
    if(hSig.GetMaximum() > hBkg.GetMaximum()):
        hSig.GetYaxis().SetRangeUser(0,hSig.GetMaximum()*1.4)
    else:
        hSig.GetYaxis().SetRangeUser(0,hBkg.GetMaximum()*1.4)
    hSig.GetXaxis().SetTitle(XTitle)
    hSig.Draw("Hist")
    hBkg.Draw("SameHist")

    c1.SaveAs("./plots/SigBkg"+tag+".png")

def SigBkg2D(SampleX,SampleY,XTitle,YTitle,XBins,YBins,xmin,xmax,ymin,ymax):

    combined = np.vstack((SampleX.Events,SampleY.Events)).T

    c1 = ROOT.TCanvas("c1","Canvas",800,600)
    ROOT.gStyle.SetOptStat(0)

    h1 = ROOT.TH2F("h1", "",XBins,xmin,xmax,YBins,ymin,ymax)
    fill_hist(h1,combined,weights=SampleX.Weights)
    h1.GetXaxis().SetTitle(XTitle)
    h1.GetYaxis().SetTitle(YTitle)
    h1.SetMarkerSize(1.3)
    h1.Draw("COLZ")

    c1.SaveAs("./plots/MutualInfo.png")

    return h1



def ConfusionMatrix(cm, Names, bins, tag=''):

    c = ROOT.TCanvas("c","Canvas",1100,800)
    ROOT.gStyle.SetOptStat(0)
    hconf = ROOT.TH2D("hconf","",bins,0,bins,bins,0,bins)
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            hconf.SetBinContent(row+1,col+1,cm[cm.shape[1]-1-col][row])
    hconf.GetXaxis().SetLabelSize(0)
    hconf.GetYaxis().SetLabelSize(0)
    hconf.Draw("colz")
    T = ROOT.TText()
    if(bins == 4):
        T.SetTextSize(0.04)
        TLabel = ROOT.TText()
        TLabel.SetTextSize(0.03)
        TLabel.DrawTextNDC(0.04,0.91,"True label")
        TLabel.DrawTextNDC(0.45,0.03,"Predicted label")
        for row in range(cm.shape[0]):
            T.DrawTextNDC(0.02,0.18+row*0.2,Names[cm.shape[0]-1-row])
            T.DrawTextNDC(0.18+row*0.2,0.05,Names[row])
            for col in range(cm.shape[1]):
                T.DrawTextNDC(0.18+col*0.2,0.18+row*0.2,str(round(hconf.GetBinContent(col+1,row+1),2)))
    elif(bins == 3):
        T.SetTextSize(0.04)
        TLabel = ROOT.TText()
        TLabel.SetTextSize(0.03)
        TLabel.DrawTextNDC(0.04,0.91,"True label")
        TLabel.DrawTextNDC(0.45,0.03,"Predicted label")
        for row in range(cm.shape[0]):
            T.DrawTextNDC(0.04,0.215+row*0.265,Names[cm.shape[0]-1-row])
            T.DrawTextNDC(0.215+row*0.265,0.06,Names[row])
            for col in range(cm.shape[1]):
                T.DrawTextNDC(0.215+col*0.265,0.215+row*0.265,str(round(hconf.GetBinContent(col+1,row+1),2)))
    elif(bins == 14):
        T.SetTextSize(0.02)
        TLabel = ROOT.TText()
        TLabel.SetTextSize(0.03)
        TLabel.DrawTextNDC(0.04,0.91,"True label")
        TLabel.DrawTextNDC(0.45,0.03,"Predicted label")
        for row in range(cm.shape[0]):
            T.DrawTextNDC(0.04,0.12+row*0.057,Names[cm.shape[0]-1-row])
            T.DrawTextNDC(0.12+row*0.057,0.08,Names[row])
            for col in range(cm.shape[1]):
                T.DrawTextNDC(0.12+col*0.057,0.12+row*0.057,str(round(hconf.GetBinContent(col+1,row+1),2)))
    elif(bins == 2):
        T.SetTextSize(0.04)
        TLabel = ROOT.TText()
        TLabel.SetTextSize(0.03)
        TLabel.DrawTextNDC(0.04,0.91,"True label")
        TLabel.DrawTextNDC(0.45,0.03,"Predicted label")
        for row in range(cm.shape[0]):
            T.DrawTextNDC(0.05,0.29+row*0.4,Names[cm.shape[0]-1-row])
            T.DrawTextNDC(0.29+row*0.4,0.07,Names[row])
            for col in range(cm.shape[1]):
                T.DrawTextNDC(0.29+col*0.4,0.29+row*0.4,str(round(hconf.GetBinContent(col+1,row+1),2)))

    c.SaveAs("./plots/Cm"+tag+".png")


def SetOverflow(hist):
    hist.SetBinContent(hist.GetNbinsX(), hist.GetBinContent(hist.GetNbinsX()) + hist.GetBinContent(hist.GetNbinsX() + 1))


def SetUnderflow(hist):
    hist.SetBinContent(0, hist.GetBinContent(0) + hist.GetBinContent(-1))


def VarHists(DataSet, key="train", Norm=False ,Sig="LO"):
    PlotSetup.VarHists(DataSet, key=key, Norm=False, Sig=Sig)
    
    











    
