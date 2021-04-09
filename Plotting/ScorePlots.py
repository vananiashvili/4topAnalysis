import ROOT
from Basics import *
from sklearn.metrics import roc_auc_score

class ScorePlots:

    def Score(path, OutPreOther, OutPreSame, DataSet, Name):

        train, test = DataSet.GetInput(Name)
        AucOther    = roc_auc_score(test.OutTrue, OutPreOther, sample_weight=test.Weights)
        AucSame     = roc_auc_score(train.OutTrue, OutPreSame, sample_weight=train.Weights)

        maxValue, minValue = GetMinMax(OutPreOther,OutPreSame)
        PreSigOther = OutPreOther[test.OutTrue == 1]
        SigWOther   = test.Weights[test.OutTrue == 1]
        PreBkgOther = OutPreOther[test.OutTrue == 0]
        BkgWOther   = test.Weights[test.OutTrue == 0]
        PreSigSame  = OutPreSame[train.OutTrue == 1]
        SigWSame    = train.Weights[train.OutTrue == 1]
        PreBkgSame  = OutPreSame[train.OutTrue == 0]
        BkgWSame    = train.Weights[train.OutTrue == 0]

        c1 = ROOT.TCanvas("c1","Canvas",800,600)
        ROOT.gStyle.SetOptStat(0)

        hSigOther = H1D(PreSigOther,40,minValue,maxValue,Weights=SigWOther)
        hBkgOther = H1D(PreBkgOther,40,minValue,maxValue,Weights=BkgWOther)
        hSigSame  = H1D(PreSigSame,40,minValue,maxValue,Weights=SigWSame)
        hBkgSame  = H1D(PreBkgSame,40,minValue,maxValue,Weights=BkgWSame)

        hSigOther.Scale(1/hSigOther.GetSumOfWeights()/Getdx(hSigOther))
        hBkgOther.Scale(1/hBkgOther.GetSumOfWeights()/Getdx(hBkgOther))
        hSigSame.Scale(1/hSigSame.GetSumOfWeights()/Getdx(hSigSame))
        hBkgSame.Scale(1/hBkgSame.GetSumOfWeights()/Getdx(hBkgSame))

        hBkgSame.SetMarkerColor(2)
        hBkgSame.SetLineColor(2)    
        hBkgSame.SetMarkerSize(0.7)
        hBkgSame.SetMarkerStyle(21)
        hSigSame.SetMarkerColor(4)
        hSigSame.SetMarkerSize(0.7)
        hSigSame.SetMarkerStyle(21)

        hSigOther.GetXaxis().SetTitle('Score')
        hBkgOther.SetLineColor(ROOT.kRed)
        if(hSigOther.GetMaximum() > hBkgSame.GetMaximum()):
            hSigOther.GetYaxis().SetRangeUser(0,hSigOther.GetMaximum()*1.4)
        else:
            hSigOther.GetYaxis().SetRangeUser(0,hBkgOther.GetMaximum()*1.4)
        hSigOther.Draw("Hist")
        hBkgOther.Draw("HistSame")
        hSigSame.Draw("E1 Same")
        hBkgSame.Draw("E1 Same")

        T1 = ROOT.TText()
        T2 = ROOT.TText()
        T1.SetTextSize(0.03)
        T2.SetTextSize(0.03)
        T1.SetTextAlign(21)
        T2.SetTextAlign(21)
        T1.DrawTextNDC(.23,.87,r"AUC = "+str(round(AucOther,3))+"  (on test)")
        T2.DrawTextNDC(.23,.84,r"AUC = "+str(round(AucSame,3))+" (on train)")

    Content = [(hSigSame,"Signal (on train sample)"), (hBkgSame,"Background (on train sample)")
                (hSigOther,"Signal (on test sample)"), (hBkgOther,"Background (on test sample)")]
    leg = Legend(Content)
    leg.Draw()

    c1.SaveAs(path+"Score"+Name+".png")

    return AucOther, AucSame

def Getdx(hist):
    dx = (hist.GetXaxis().GetXmax() - hist.GetXaxis().GetXmin()) / hist.GetNbinsX()
    return dx