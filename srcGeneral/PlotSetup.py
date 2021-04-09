import PlotService
import numpy as np
from DIClasses import DISample
import re

def SetInputArr(DataSet,key,Sig):
    """ Set hist input Arrays """
    train, test = DataSet.GetInput("Even")
    vali = DataSet.vali

    if(key == 'All'):
        if(Sig == "NLO"):                                                                   #Excluding LO sig events
            l = [train.Events[train.OutTrue != 1], test.Events, vali.Events]
            lw = [train.Weights[train.OutTrue != 1], test.Weights, vali.Weights]
            lm = [train.MultiClass[train.OutTrue != 1], test.MultiClass, vali.MultiClass]
        elif(Sig == "LO"):                                                                  #Excluding NLO sig events
            l = [train.Events,test.Events[test.OutTrue != 1], vali.Events[vali.OutTrue != 1]]
            lw = [train.Weights,test.Weights[test.OutTrue != 1], vali.Weights[vali.OutTrue != 1]]
            lm = [train.MultiClass,test.MultiClass[test.OutTrue != 1], vali.MultiClass[vali.OutTrue != 1]]
        x = np.concatenate(l,axis=0)
        w = np.concatenate(lw,axis=0)
        m = np.concatenate(lm,axis=0)
        return DISample(x,w,None,m,train.LVariables,train.Names)
    elif(key == 'train'):
        return train
    elif(key == 'test'):
        return test
    elif(key == 'vali'):
        return vali

def GetBrachArr(Data,col,Class,XTitle):
    Events = Data.Events[Data.MultiClass == Class]                  #Only Events from the process
    if(Events.ndim == 2):
        Events = Events[:,col]                                          #Only Events from the Variable
    elif(Events.ndim == 3):
        if(XTitle[:3] == 'Seq'):                                                    # If the Variable contains values for different Variables...
            XTitle = XTitle[4:]                                                     # remove Seq_ from title
            SeqPos = XTitle.split()[0].split('^')[1]
            SeqPos = int(re.findall(r'\d+', SeqPos)[0])                             # Get Sequence Position from the title
            print(Events.shape,col,SeqPos)
            Events = Events[:,col][:,SeqPos]                                        
        else:                                                                       # If there is only one object
            Events = Events[:,col][:,0]                                             

    if(XTitle[-5:] == "[GeV]"):                                                              # Convert MeV to GeV
        Events = Events / 1000

    return Events


def VarHists(DataSet,key="train",Norm=False,Sig="LO"):

    path = "./plots/Variables/"
    bins = 30

    PlotService.VarHist(DataSet, Sig,path,key, "leading_jet_pT", r"p_{T}(j)^{\text{leading}} [GeV]",0,800,bins,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "lep_0_pt", r"p_{T}(\ell)^{\text{leading}} [GeV]",0,500,bins,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "nJets", "Jet multiplicity",4,14,10,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "met_met",r" E_{T}^{\text{miss}} [GeV]",0,500,bins,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "deltaR_ll_min", r"\Delta R(\ell,\ell)_{min}",0,5,bins,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "deltaR_ll_max", r"\Delta R(\ell,\ell)_{max}",0,5,bins,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "deltaR_ll_sum", r"\sum \Delta R(\ell,\ell)",0,10,bins,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "deltaR_bb_min", r"\Delta R(b,b)_{min}",0,5,bins,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "jet_sum_mv2c10_Continuous", r"\sum w_{Mv2c10}",10,25,15,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'deltaPhi_l0j0', r'\Delta \Phi (l_{0}j_{0})', 0, 3.141, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'deltaR_bj_min', r'\Delta R(b,j)_{min}', 0, 2.5, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'deltaR_lb_max', r'\Delta R(\ell,b)_{max}',  1, 5, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'deltaR_lb_min', r'\Delta R(\ell,b)_{min}', 0, 3, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'deltaR_lj_min', r'\Delta R(\ell,j)_{min}', 0, 2, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'HT_jets_noleadjet', 'H_{T}^{no leading jet} [GeV]', 100, 1200, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'jet_1_pt', 'jet1 p_{T} [GeV]', 0, 500, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'jet_mass_dividedby_pt_max', 'm_{jet}/max(jet p_{T})', 0.1, 0.3, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'leading_bjet_pT', 'Leading b-jet p_{T} [GeV]', 0, 500, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'lep_1_pt', 'lep1 p_{T} [GeV]', 20, 200, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'lep_2_pt', 'lep2 p_{T} [GeV]', 0, 150, bins ,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'lep_0_phi', r'\phi (\ell)^{\text{leading}}', 0, 3.2, bins ,Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'lowest_jet_pT', 'lowest Jet p_{T} [GeV]', 0, 80, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'met_phi', r'\phi^{\text{miss}}', -3.5, 3.5, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'nleps', r'Number of leptons', 0, 10, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'lep_0_eta', r'\eta_{lep,0}', -2.7, 2.7, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, 'el_eta[0]', r'el eta', -3.141, 3.141, bins, Norm=Norm)
    PlotService.VarHist(DataSet, Sig,path,key, "jet_5_pt", r"lep5 p_{T} [GeV]",20,100,bins,Norm=Norm)
    # PlotService.VarHist(DataSet, Sig,path,key, "Evt_Channel", r"Event Channel",1,7,7,Norm=Norm)


    #Jet
    PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{0} [GeV]', 0, 1000, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{1} [GeV]', 0, 650, bins, Norm=Norm, pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{2} [GeV]', 0, 600, bins, Norm=Norm, pos=2)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{3} [GeV]', 0, 600, bins, Norm=Norm, pos=3)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{4} [GeV]', 0, 600, bins, Norm=Norm, pos=4)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{5} [GeV]', 0, 600, bins, Norm=Norm, pos=5)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{6} [GeV]', 0, 600, bins, Norm=Norm, pos=6)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{7} [GeV]', 0, 150, bins, Norm=Norm, pos=7)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_pt', r'Seq_pt_{j}^{8} [GeV]', 0, 150, bins, Norm=Norm, pos=8)
    PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{0}', -2.5, 2.5, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{1}', -2.5, 2.5, bins, Norm=Norm,pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{2}', -2.5, 2.5, bins, Norm=Norm,pos=2)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{3}', -2.5, 2.5, bins, Norm=Norm,pos=3)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{4}', -2.5, 2.5, bins, Norm=Norm,pos=4)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{5}', -2.5, 2.5, bins, Norm=Norm, pos=5)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{6}', -2.5, 2.5, bins, Norm=Norm, pos=6)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{7}', -2.5, 2.5, bins, Norm=Norm, pos=7)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_eta', r'Seq_\eta_{j}^{8}', -2.5, 2.5, bins, Norm=Norm, pos=8)
    PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{0}', -3.14, 3.14, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{1}', -3.14, 3.14, bins, Norm=Norm,pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{2}', -3.14, 3.14, bins, Norm=Norm,pos=2)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{3}', -3.14, 3.14, bins, Norm=Norm,pos=3)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{4}', -3.14, 3.14, bins, Norm=Norm,pos=4)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{5}', -3.14, 3.14, bins, Norm=Norm, pos=5)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{6}', -3.14, 3.14, bins, Norm=Norm, pos=6)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{7}', -3.14, 3.14, bins, Norm=Norm, pos=7)
    # PlotService.VarHist(DataSet, Sig,path,key, 'jet_phi', r'Seq_\phi_{j}^{8}', -3.14, 3.14, bins, Norm=Norm, pos=8)
    #electron
    PlotService.VarHist(DataSet, Sig,path,key, 'el_pt', r'Seq_pt_{el}^{0} [GeV]', 20, 400, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'el_pt', r'Seq_pt_{el}^{1} [GeV]', 20, 400, bins, Norm=Norm, pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'el_pt', r'Seq_pt_{el}^{2} [GeV]', 20, 400, bins, Norm=Norm, pos=2)
    PlotService.VarHist(DataSet, Sig,path,key, 'el_eta', r'Seq_\eta_{el}^{0}', -2.5, 2.5, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'el_eta', r'Seq_\eta_{el}^{1}', -2.5, 2.5, bins, Norm=Norm, pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'el_eta', r'Seq_\eta_{el}^{2}', -2.5, 2.5, bins, Norm=Norm, pos=2)
    PlotService.VarHist(DataSet, Sig,path,key, 'el_phi', r'Seq_\phi_{el}^{0}', -3.14, 3.14, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'el_phi', r'Seq_\phi_{el}^{1}', -3.14, 3.14, bins, Norm=Norm, pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'el_phi', r'Seq_\phi_{el}^{2}', -3.14, 3.14, bins, Norm=Norm, pos=2)
    #muon
    PlotService.VarHist(DataSet, Sig,path,key, 'mu_pt', r'Seq_pt_{\mu}^{0} [GeV]', 0, 400, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'mu_pt', r'Seq_pt_{\mu}^{1} [GeV]', 0, 350, bins, Norm=Norm, pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'mu_pt', r'Seq_pt_{\mu}^{2} [GeV]', 0, 350, bins, Norm=Norm, pos=2)
    PlotService.VarHist(DataSet, Sig,path,key, 'mu_eta', r'Seq_\eta_{\mu}^{0}', -2.5, 2.5, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'mu_eta', r'Seq_\eta_{\mu}^{1}', -2.5, 2.5, bins, Norm=Norm, pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'mu_eta', r'Seq_\eta_{\mu}^{2}', -2.5, 2.5, bins, Norm=Norm, pos=2)
    PlotService.VarHist(DataSet, Sig,path,key, 'mu_phi', r'Seq_\phi_{\mu}`^{0}', -3.14, 3.14, bins, Norm=Norm,pos=0)
    # PlotService.VarHist(DataSet, Sig,path,key, 'mu_phi', r'Seq_\phi_{\mu}^{1}', -3.14, 3.14, bins, Norm=Norm, pos=1)
    # PlotService.VarHist(DataSet, Sig,path,key, 'mu_phi', r'Seq_\phi_{\mu}^{2}', -3.14, 3.14, bins, Norm=Norm, pos=2)