import ROOT
import numpy as np
import math
from root_numpy import fill_hist, fill_graph
import Plotting
import math

def SetOverflow(hist):
    if(isinstance(hist,ROOT.TH1F)):
        hist.SetBinContent(hist.GetNbinsX(), hist.GetBinContent(hist.GetNbinsX()) + hist.GetBinContent(hist.GetNbinsX() + 1))
    elif(isinstance(hist,ROOT.TH2F)):
        for i in range(hist.GetNbinsY()):
            hist.SetBinContent(hist.GetNbinsX(), i, hist.GetBinContent(hist.GetNbinsX(),i) + hist.GetBinContent(hist.GetNbinsX() + 1,i))
    else:
        print("no Overflow known for this class")
        assert 0 == 1

def SetUnderflow(hist):
    if(isinstance(hist,ROOT.TH1F)):
        hist.SetBinContent(0, hist.GetBinContent(0) + hist.GetBinContent(-1))
    else:
        print("no Overflow known for this class")
        assert 0 == 1


def H1D(Array,bins,xmin,xmax,Weights=None,Title='',Norm=None):

    h1 = ROOT.TH1F("h1",Title,bins,xmin,xmax)
    fill_hist(h1,Array,weights=Weights)

    if(Plotting.Underflow == True):
        SetUnderflow(h1)

    if(Plotting.Overflow == True):
        SetOverflow(h1)

    if(Norm == True and h1.Integral() != 0):
        h1.Scale(1/h1.Integral())

    h1.SetLineWidth(2)

    return h1


def Legend(LHist,Content,Pos=(0.6,0.75,0.9,0.9)):
    """
    Creates a TLegend
    Content = Tuple (Hist,Name)
    Pos     = Position of the TLegend
    """

    leg = ROOT.TLegend(Pos[0],Pos[1],Pos[2],Pos[3])
    for i in range(len(Content)):
        leg.AddEntry(LHist[i],Content[i])

    return leg


def GetMinMax(Arr1,Arr2):
    Arr = np.append(Arr1,Arr2)

    return np.min(Arr), np.max(Arr)


def Hist1D(Arr,XTitle,YTitle,bins,xmin,xmax,Weights=None,Stats=None,Title="",Norm=None,drawOpt="HIST"):
    """ Simple TH1F Histogramm """
    
    Hist =  H1D(Arr,bins,xmin,xmax,Weights=Weights,Title=Title,Norm=Norm)
    
    c1 = ROOT.TCanvas("c1","c1",800,600)
    ROOT.gStyle.SetOptStat(0)
    Hist.Draw(drawOpt)

    Hist.GetXaxis().SetTitle(XTitle)
    Hist.GetYaxis().SetTitle(YTitle)

    if(Stats == True):
        MaxBin = Hist.GetMaximumBin()
        MaxBinPos = round(Hist.GetXaxis().GetBinCenter(MaxBin))
        mean = round(np.mean(Arr))
        sigma = round(math.sqrt(np.var(Arr)))
        T1 = ROOT.TLatex()
        T1.SetTextSize(0.03)
        T1.SetTextAlign(13)
        T2 = ROOT.TLatex()
        T2.SetTextSize(0.03)
        T2.SetTextAlign(13)
        T2.DrawLatexNDC(0.73,0.89,"Max Bin at "+str(MaxBinPos))
        T1.DrawLatexNDC(0.73,0.86,"#splitline{#mu = "+str(mean)+"}{#sigma = "+str(sigma)+"}")


    c1.SaveAs(Plotting.SavePath+"Hist1D.png")

    return Hist



def Hist1DStack(LHist,XTitle,YTitle,Title='',Leg=None,Drawopt=""):

    c1 = ROOT.TCanvas("c1","c1",800,600)
    ROOT.gStyle.SetOptStat(0)

    hs = ROOT.THStack("hs","")

    for Hist in LHist:
        hs.Add(Hist,"Hist")

    hs.Draw(Drawopt)
    hs.GetXaxis().SetTitle(XTitle)
    hs.GetYaxis().SetTitle(YTitle)
    hs.SetTitle(Title)

    if(Leg != None):
        Leg.Draw()

    c1.SaveAs(Plotting.SavePath+"StackHist.png")

    return Hist


def HistCombined(LHist,XTitle,YTitle,Title='',Leg=None,Logy=False):

    c1 = ROOT.TCanvas("c1","c1",800,600)
    ROOT.gStyle.SetOptStat(0)
    if(Logy):
        c1.SetLogy()
    else:
        LHist[0].GetYaxis().SetRangeUser(0,FindMax(LHist))
        #LHist[0].GetYaxis().SetRangeUser(0,0.6)
    LHist[0].SetTitle(Title)

    for i in range(len(LHist)):
        if(i == 0):
            LHist[i].Draw("Hist")
        else:
            LHist[i].Draw("SAMEHIST")

    for i in range(len(LHist)):
        LHist[i].SetLineWidth(2)

    LHist[0].GetXaxis().SetTitle(XTitle)
    LHist[0].GetYaxis().SetTitle(YTitle)

    if(Leg != None):
        Leg.Draw()

    c1.SaveAs(Plotting.SavePath+"CombinedHist.png")


def FindMax(LHist):
    Max = 0
    for Hist in LHist:
        if(Hist.GetMaximum() > Max):
            Max = Hist.GetMaximum()
    
    if(Max < 1):
        return Max*1.3
    else:
        return int(math.ceil(Max*1.3))


def H2D(x,y,xbins,xmin,xmax,ybins,ymin,ymax,Weights=None,Title='',Norm=None):

    Array = np.transpose(np.vstack((x,y)))

    h2 = ROOT.TH2F("h2",Title,xbins,xmin,xmax,ybins,ymin,ymax)
    fill_hist(h2,Array,weights=Weights)

    if(Plotting.Overflow == True):
        SetOverflow(h2)

    if(Norm == "x"):
        for i in range(h2.GetNbinsX()):
            Sum = 0
            for j in range(h2.GetNbinsY()):
                Sum += h2.GetBinContent(i+1,j+1)
            if(Sum != 0):
                for j in range(h2.GetNbinsY()):
                    h2.SetBinContent(i+1,j+1,h2.GetBinContent(i+1,j+1)/Sum)
    elif(Norm == "y"):
        for i in range(h2.GetNbinsY()):
            Sum = 0
            for j in range(h2.GetNbinsX()):
                Sum += h2.GetBinContent(j+1,i+1)
            if(Sum != 0):
                for j in range(h2.GetNbinsX()):
                    h2.SetBinContent(j+1,i+1,h2.GetBinContent(j+1,i+1)/Sum)



    return h2


def Hist2D(x,y,XTitle,YTitle,xbins,xmin,xmax,ybins,ymin,ymax,Title='',Norm=False,log=False):

    Hist =  H2D(x,y,xbins,xmin,xmax,ybins,ymin,ymax,Title=Title,Norm=Norm)
    
    c1 = ROOT.TCanvas("c1","c1",800,600)
    ROOT.gStyle.SetOptStat(0)
    Hist.Draw("colz")

    Hist.GetXaxis().SetTitle(XTitle)
    Hist.GetYaxis().SetTitle(YTitle)


    c1.SaveAs(Plotting.SavePath+"Hist2D.png")

    return Hist

def FromHist(Hist,DrawOpt=""):

    c1 = ROOT.TCanvas("c1","c1",800,600)
    ROOT.gStyle.SetOptStat(0)
    Hist.Draw(DrawOpt)

    c1.SaveAs(Plotting.SavePath+"Redrawn.png")


def AlphanumericLabels(Hist,Labels,Axis="x"):

    if(Axis == "x"):
        Axis = Hist.GetXaxis()
    elif(Axis == "y"):
        Axis = Hist.GetYaxis()

    for i,Label in enumerate(Labels):
        Axis.SetBinLabel(i+1,Label)

def ImportFromtxt(Path,delimiter=";"):

    with open(Path,"r") as f:
        lines = f.readlines()
        col = [float(s) for s in lines[1].split(delimiter)]
        TotalArr = np.array([col])
        for i in range(2,len(lines)):
            col = [float(s) for s in lines[i].split(delimiter)]
            Arr = np.array([col])
            TotalArr = np.vstack((TotalArr,Arr))

    return TotalArr
