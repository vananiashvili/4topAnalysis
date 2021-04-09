import ROOT
import numpy as np
from Utils import roc_auc_score
from root_numpy import fill_hist


def Score(path, OutPreOther, OutPreSame, DataSet, Name):

    train, test = DataSet.GetInput(Name)
    AucOther    = roc_auc_score(test.OutTrue, OutPreOther, sample_weight=test.Weights)
    AucSame     = roc_auc_score(train.OutTrue, OutPreSame, sample_weight=train.Weights)

    maxValue, minValue = np.max(OutPreOther), np.min(OutPreOther)
    if(np.max(OutPreSame) > maxValue):
        maxValue = np.max(OutPreSame)
    if(np.min(OutPreSame) < minValue):
        minValue = np.min(OutPreSame)
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

    hSigOther = ROOT.TH1F("h1", "",40,minValue,maxValue)
    hBkgOther = ROOT.TH1F("h2", "",40,minValue,maxValue)
    hSigSame  = ROOT.TH1F("h3", "",40,minValue,maxValue)
    hBkgSame  = ROOT.TH1F("h4", "",40,minValue,maxValue)

    fill_hist(hSigOther, PreSigOther, weights=SigWOther)
    fill_hist(hBkgOther, PreBkgOther, weights=BkgWOther)
    fill_hist(hSigSame, PreSigSame, weights=SigWSame)
    fill_hist(hBkgSame, PreBkgSame, weights=BkgWSame)

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


    leg = ROOT.TLegend(0.6,0.75,0.9,0.9)
    leg.AddEntry(hSigSame,"Signal (on train sample)")
    leg.AddEntry(hBkgSame,"Background (on train sample)")
    leg.AddEntry(hSigOther,"Signal (on test sample)")
    leg.AddEntry(hBkgOther,"Background (on test sample)")
    leg.Draw()

    c1.SaveAs(path+"Score"+Name+".png")

    return AucOther, AucSame

def MultiScore(path, OutPreOther, OutPreSame, DataSet, Name, ClassNum):

    train, test = DataSet.GetInput(Name)
    AucOther    = roc_auc_score(GetTrueLabels(test, ClassNum), OutPreOther, sample_weight=test.Weights)
    AucSame     = roc_auc_score(GetTrueLabels(train, ClassNum), OutPreSame, sample_weight=train.Weights)
    maxValue, minValue = np.max(OutPreOther), np.min(OutPreOther)
    if(np.max(OutPreSame) > maxValue):
        maxValue = np.max(OutPreSame)
    if(np.min(OutPreSame) < minValue):
        minValue = np.min(OutPreSame)

    c1 = ROOT.TCanvas("c1","Canvas",800,600)
    ROOT.gStyle.SetOptStat(0)

    LClass = [0,10,11,13]
    # Names = {0:r't\bar{t}t\bar{t}',13:r't\bar{t}W',12:r't\bar{t}WW',11:r't\bar{t}Z',10:r't\bar{t}H',2:'V+jets'
    #     ,3:'VV',4:r't(\bar{t})X',1:'others',9:r't\bar{t} QmisID',8:r't\bar{t} CO',
    #     7:r't\bar{t} HF',6:r't\bar{t} light',5:r't\bar{t} other'}
    # Color = {13:3,12:ROOT.kBlack,11:6,10:4,                                  #Colorcoding
    #         2:ROOT.kAzure+1,3:ROOT.kYellow-7,4:ROOT.kGreen-5,1:ROOT.kGreen-1,       
    #         9:ROOT.kGreen-10,8:ROOT.kPink-9,7:ROOT.kViolet-4,
    #         6:ROOT.kBlue-7,5:ROOT.kBlue-10,0:ROOT.kRed+1}
    Names = {0:r't\bar{t}t\bar{t}',1:r't\bar{t}V',2:'rest'}
    Color = {0:ROOT.kRed+1,1:ROOT.kBlack,2:ROOT.kGreen}
    LHists = []
    for Class in LClass:
        hist  = GetScoreTH(Class,OutPreOther,test.Weights,test.MultiClass,minValue,maxValue)
        hist.SetLineColor(Color[Class])
        hist.SetLineWidth(2)
        LHists.append(hist)

    Pre     = OutPreOther[GetTrueLabels(test, ClassNum) == 0]
    Weight  = test.Weights[GetTrueLabels(test, ClassNum) == 0]
    histAll    = ROOT.TH1F("hAll", "",20,minValue,maxValue)
    fill_hist(histAll, Pre, weights=Weight)
    histAll.Scale(1/histAll.GetSumOfWeights()/Getdx(histAll))
    histAll.SetLineColor(1)
    histAll.SetLineWidth(2)

    LHists[0].GetXaxis().SetTitle('Score')
    Lmax = [hist.GetMaximum() for hist in LHists]
    LHists[0].GetYaxis().SetRangeUser(0,max(Lmax)*1.4)
    LHists[0].Draw("Hist")
    histAll.Draw("HistSame")
    for i in range(1,len(LHists)):
        LHists[i].Draw("HistSame")


    T1 = ROOT.TText()
    T2 = ROOT.TText()
    T1.SetTextSize(0.03)
    T2.SetTextSize(0.03)
    T1.SetTextAlign(21)
    T2.SetTextAlign(21)
    T1.DrawTextNDC(.23,.87,r"AUC = "+str(round(AucOther,3))+"  (on test)")
    T2.DrawTextNDC(.23,.84,r"AUC = "+str(round(AucSame,3))+" (on train)")


    leg = ROOT.TLegend(0.6,0.75,0.9,0.9)
    for i,hist in  enumerate(LHists):
        leg.AddEntry(hist,Names[LClass[i]])
    leg.AddEntry(histAll,"all backgrounds")
    leg.Draw()

    c1.SaveAs(path+"Score"+Name+'_'+str(ClassNum)+".png")

    return AucOther, AucSame

# def MultiScore(path, OutPreOther, OutPreSame, DataSet, Name, ClassNum):

#     train, test = DataSet.GetInput(Name)
#     AucOther    = roc_auc_score(GetTrueLabels(test, ClassNum), OutPreOther, sample_weight=test.Weights)
#     AucSame     = roc_auc_score(GetTrueLabels(train, ClassNum), OutPreSame, sample_weight=train.Weights)
#     maxValue, minValue = np.max(OutPreOther), np.min(OutPreOther)
#     if(np.max(OutPreSame) > maxValue):
#         maxValue = np.max(OutPreSame)
#     if(np.min(OutPreSame) < minValue):
#         minValue = np.min(OutPreSame)

#     c1 = ROOT.TCanvas("c1","Canvas",800,600)
#     ROOT.gStyle.SetOptStat(0)

#     LClass = [0,1,100]
#     # Names = {0:r't\bar{t}t\bar{t}',13:r't\bar{t}W',12:r't\bar{t}WW',11:r't\bar{t}Z',10:r't\bar{t}H',2:'V+jets'
#     #     ,3:'VV',4:r't(\bar{t})X',1:'others',9:r't\bar{t} QmisID',8:r't\bar{t} CO',
#     #     7:r't\bar{t} HF',6:r't\bar{t} light',5:r't\bar{t} other'}
#     Names = [r't\bar{t}t\bar{t}','others',r't\bar{t}tV']
#     # Color = {13:3,12:ROOT.kBlack,11:6,10:4,                                  #Colorcoding
#     #         2:ROOT.kAzure+1,3:ROOT.kYellow-7,4:ROOT.kGreen-5,1:ROOT.kGreen-1,       
#     #         9:ROOT.kGreen-10,8:ROOT.kPink-9,7:ROOT.kViolet-4,
#     #         6:ROOT.kBlue-7,5:ROOT.kBlue-10,0:ROOT.kRed+1}
#     Color = [ROOT.kRed+1,3,9]
#     LHists = []
#     for i,Class in enumerate(LClass):
#         if(Class == 100):
#                 TrueLabel = test.MultiClass
#                 Pre     = OutPreOther[TrueLabel == 10]
#                 Pre     = np.append(Pre,OutPreOther[TrueLabel == 11])
#                 Pre     = np.append(Pre,OutPreOther[TrueLabel == 13])

#                 Weight  = test.Weights[TrueLabel == 10]
#                 Weight  = np.append(Weight,test.Weights[TrueLabel == 11])
#                 Weight  = np.append(Weight,test.Weights[TrueLabel == 13])

#                 histttV    = ROOT.TH1F("httV", "",20,minValue,maxValue)
#                 fill_hist(histttV, Pre, weights=Weight)
#                 histttV.Scale(1/histttV.GetSumOfWeights()/Getdx(histttV))
#                 histttV.SetLineColor(9)
#                 histttV.SetLineWidth(2)
#         else:
#             hist  = GetScoreTH(Class,OutPreOther,test.Weights,test.MultiClass,minValue,maxValue)
#             hist.SetLineColor(Color[i])
#             hist.SetLineWidth(2)
#             LHists.append(hist)

#     Pre     = OutPreOther[GetTrueLabels(test, ClassNum) == 0]
#     Weight  = test.Weights[GetTrueLabels(test, ClassNum) == 0]
#     histAll    = ROOT.TH1F("hAll", "",20,minValue,maxValue)
#     fill_hist(histAll, Pre, weights=Weight)
#     histAll.Scale(1/histAll.GetSumOfWeights()/Getdx(histAll))
#     histAll.SetLineColor(1)
#     histAll.SetLineWidth(2)

#     LHists[0].GetXaxis().SetTitle('Score')
#     Lmax = [hist.GetMaximum() for hist in LHists]
#     LHists[0].GetYaxis().SetRangeUser(0,max(Lmax)*1.4)
#     LHists[0].Draw("Hist")
#     histAll.Draw("HistSame")
#     histttV.Draw("HistSame")
#     for i in range(1,len(LHists)):
#         LHists[i].Draw("HistSame")


#     T1 = ROOT.TText()
#     T2 = ROOT.TText()
#     T1.SetTextSize(0.03)
#     T2.SetTextSize(0.03)
#     T1.SetTextAlign(21)
#     T2.SetTextAlign(21)
#     T1.DrawTextNDC(.23,.87,r"AUC = "+str(round(AucOther,3))+"  (on test)")
#     T2.DrawTextNDC(.23,.84,r"AUC = "+str(round(AucSame,3))+" (on train)")


#     leg = ROOT.TLegend(0.6,0.75,0.9,0.9)
#     for i,hist in  enumerate(LHists):
#         leg.AddEntry(hist,Names[LClass[i]])
#     leg.AddEntry(histttV, r"t\bar{t}V")
#     leg.AddEntry(histAll,"all backgorunds")
#     leg.Draw()

#     c1.SaveAs(path+"Score"+Name+'_'+str(ClassNum)+".png")

#     return AucOther, AucSame

def GetTrueLabels(Sample,ClassNum):
    """ Returns the binary truth lable of a Class (Class vs all) """
    OutTrue = np.zeros(len(Sample.Weights))
    OutTrue = np.where(Sample.MultiClass != ClassNum, OutTrue, 1)
    return OutTrue

def GetScoreTH(ClassNum,OutPreOther,Weights,MultiClass,xmin,xmax):

    Pre     = OutPreOther[MultiClass == ClassNum]
    Weight  = Weights[MultiClass == ClassNum]
    hist    = ROOT.TH1F("h"+str(ClassNum), "",20,xmin,xmax)
    fill_hist(hist, Pre, weights=Weight)
    hist.Scale(1/hist.GetSumOfWeights()/Getdx(hist))

    return hist

def Getdx(hist):
    dx = (hist.GetXaxis().GetXmax() - hist.GetXaxis().GetXmin()) / hist.GetNbinsX()
    return dx