import ROOT
import Plotting

def FitBreitWigner(Hist,xmin,xmax,Init=None):
    """ Fits a Breit Wigner to a given Histogramm"""

    ROOT.gStyle.SetOptFit(1)

    Breit = ROOT.TF1("BreitFit","[0]*TMath::BreitWigner(x,[1],[2])",xmin,xmax)
    Breit.SetParName(0,"C")
    Breit.SetParName(1,"mean")
    Breit.SetParName(2,"gamma")

    if Init != None:
        Breit.SetParameter(0,Init[0])
        Breit.SetParameter(1,Init[1])
        Breit.SetParameter(2,Init[2])

    Hist.Fit(Breit,"R")

    c1 = ROOT.TCanvas("c1","c1",800,600)
    Hist.Draw()
    Hist.SetStats(1)

    c1.SaveAs(Plotting.SavePath+"BreitWignerFit.png")

    return Hist


def FitLandau(Hist,xmin,xmax,Init=None):
    """ Fits a Landau to a given Histogramm"""

    ROOT.gStyle.SetOptFit(1)

    Landau = ROOT.TF1("LandauFit","[0]*[1]*TMath::Landau(x,[2],[3])",xmin,xmax)
    Landau.SetParName(0,r"N_{events}")
    Landau.SetParName(1,"C")
    Landau.SetParName(2,"mpv")
    Landau.SetParName(3,"simga")

    if Init != None:
        Landau.SetParameter(0,Init[0])
        Landau.SetParameter(1,Init[1])
        Landau.SetParameter(2,Init[2])
        Landau.SetParameter(3,Init[3])

    Hist.Fit(Landau,"R")

    c1 = ROOT.TCanvas("c1","c1",800,600)
    Hist.SetStats(1)
    Hist.Draw()

    c1.SaveAs(Plotting.SavePath+"LandauFit.png")

    return Hist


def FitGauss(Hist,xmin,xmax,Init=None):
    """ Fits a Gaussian to a given Histogramm"""

    ROOT.gStyle.SetOptFit(1)

    Gauss = ROOT.TF1("Gauss","[0]*TMath::Gaus(x,[1],[2])",xmin,xmax)
    Gauss.SetParName(0,"C")
    Gauss.SetParName(1,"mean")
    Gauss.SetParName(2,"sigma")

    if Init != None:
        Gauss.SetParameter(0,Init[0])
        Gauss.SetParameter(1,Init[1])
        Gauss.SetParameter(2,Init[2])

    Hist.Fit(Gauss,"R")

    c1 = ROOT.TCanvas("c1","c1",800,600)
    Hist.SetStats(1)
    Hist.Draw()

    c1.SaveAs(Plotting.SavePath+"GaussianFit.png")

    return Hist