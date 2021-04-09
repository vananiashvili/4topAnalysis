import os
import Plotting.Basics
import Plotting.Fits
import numpy as np

def Chi2Calibration(dR,Tops,Weights):
    Plotting.SavePath = "./plots/Calibration/"
    Plotting.Overflow = True
    YTitle = "norm. #Events"
    Weights = None


    #DeltaR
    Plotting.Basics.Hist1D(dR,r"\Delta R (RecoTop,TruthTop)",YTitle,40,0,4,Stats=False,Weights=Weights,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"deltaR.png")

    #Plotting.Overflow = False

    #Top Mass
    TopMass = np.array([RecoTop.M for RecoTop in Tops])
    Hist = Plotting.Basics.Hist1D(TopMass,r"M^{RecoTop} [GeV]",YTitle,200,100,300,Stats=True,Weights=Weights,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"TopMass.png")
    #Hist.GetYaxis().SetRangeUser(0,0.3)
    Plotting.Fits.FitGauss(Hist,145,200,[0,172,30])
    os.system("mv "+Plotting.SavePath+"GaussianFit.png "+Plotting.SavePath+"TopMass.png")

    #W Mass
    WMass = np.array([RecoTop.W.M for RecoTop in Tops])
    Hist = Plotting.Basics.Hist1D(WMass,r"M^{RecoW} [GeV]",YTitle,200,40,140,Stats=True,Weights=Weights,Norm=True)
    os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"WMass.png")
    Plotting.Fits.FitGauss(Hist,70,94,[0,80.4,20])
    os.system("mv "+Plotting.SavePath+"GaussianFit.png "+Plotting.SavePath+"WMass.png")

def CaliComparison(DeltaR1,DeltaR2):
    Plotting.SavePath = "./plots/Calibration/"
    Plotting.Overflow = True
    YTitle = "norm. #Events"
    Weights = None

    H1 = Plotting.Basics.Hist1D(DeltaR1[:,0],r"\Delta R (RecoTop,TruthTop)",YTitle,50,0,0.5,Stats=False,Weights=Weights)
    H2 = Plotting.Basics.Hist1D(DeltaR2[:,0],r"\Delta R (RecoTop,TruthTop)",YTitle,50,0,0.5,Stats=False,Weights=Weights)

    c1 = ROOT.TCanvas("ct","ct",800,600)
    ROOT.gStyle.SetOptStat(0)
    H1.Draw("Hist")
    H2.Draw("SAME")

    c1.SavePath(Plotting.SavePath+"dRComparision.png")




# def LikeliCalibration(BestDeltaR):
#     Plotting.SavePath = "./plots/Calibration/"
#     Plotting.Overflow = True
#     YTitle = "norm. #Events"
#     Weights = None

#     #DeltaR
#     Plotting.Basics.Hist1D(BestDeltaR[:,0],r"\Delta R (RecoTop,TruthTop)",YTitle,50,0,0.5,Stats=False,Weights=Weights,Norm=True)
#     os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"deltaR.png")

#     #Top Mass
#     TopMass = np.array([RecoTop.M for RecoTop in BestDeltaR[:,1]])
#     Hist = Plotting.Basics.Hist1D(TopMass,r"M^{RecoTop} [GeV]",YTitle,100,100,250,Stats=True,Weights=Weights)
#     os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"TopMass.png")
#     Plotting.Fits.FitGauss(Hist,140,210,[0,172,50])
#     os.system("mv "+Plotting.SavePath+"GaussianFit.png "+Plotting.SavePath+"TopMass.png")

#     #W Mass
#     WMass = np.array([RecoTop.W.M for RecoTop in BestDeltaR[:,1]])
#     Hist = Plotting.Basics.Hist1D(WMass,r"M^{RecoW} [GeV]",YTitle,100,0,180,Stats=True,Weights=Weights)
#     os.system("mv "+Plotting.SavePath+"Hist1D.png "+Plotting.SavePath+"WMass.png")
#     Plotting.Fits.FitGauss(Hist,50,110,[0,80,30])
#     os.system("mv "+Plotting.SavePath+"GaussianFit.png "+Plotting.SavePath+"WMass.png")