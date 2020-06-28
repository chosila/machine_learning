

inputPathNTuples='../data/'
treeDirName='hh_3l_OS_forBDTtraining/sel/evtntuple' # name of the tree directory inside ntuple;
                                                    #Format: "%s/%s/evtTree" % (treeDirName,sampleName)


keys=[
    'WZ', ## WZ inclusive sample + WZZ
    'signal_ggf_spin0_400_hh_4v', # signal 
]
'''
trainVarsAll = [
    "lep1_pt", "lep1_conePt", "lep1_eta", "lep1_tth_mva", "mindr_lep1_jet", "mT_lep1", 
    "lep2_pt", "lep2_conePt", "lep2_eta", "lep2_tth_mva", "mindr_lep2_jet", "mT_lep2",
    "lep3_pt", "lep3_conePt", "lep3_eta", "lep3_tth_mva", "mindr_lep3_jet", "mT_lep3", 
    "avg_dr_jet", "ptmiss",  "htmiss", "dr_leps",
    #"lumiScale", 
    "genWeight", "evtWeight",
    "lep1_genLepPt", "lep2_genLepPt", "lep3_genLepPt",
    "lep1_fake_prob", "lep2_fake_prob", "lep3_fake_prob",
    "lep1_frWeight", "lep2_frWeight", "lep3_frWeight",  
#"mvaOutput_3l_ttV", "mvaOutput_3l_ttbar", "mvaDiscr_3l",
    "mbb_loose", "mbb_medium",
    "dr_lss", "dr_los1", "dr_los2",
    "met", "mht", "met_LD", "HT", "STMET",
    #"mSFOS2l", 
    "m_jj", "diHiggsVisMass", "diHiggsMass",
    "mTMetLepton1", "mTMetLepton2",
    "vbf_m_jj", "vbf_dEta_jj", "numSelJets_nonVBF",
    #
    "nJet", "nBJetLoose", "nBJetMedium", "nElectron", "nMuon",
    "lep1_isTight", "lep2_isTight", "lep3_isTight",
    "sumLeptonCharge", "numSameFlavor_OS", "isVBF",
    "event",        
    "gen_mHH"        
]
'''

trainVarsAll = [
    "dr_lss",
    "m_jj", "diHiggsVisMass",
    "numSameFlavor_OS_3l",
    "met_LD",#"mT_MEtLep1", "mT_MEtLep1",
    "event", "genWeight", "evtWeight",
]


trainVars = [
    "dr_lss",
    "m_jj", "diHiggsVisMass",
    "numSameFlavor_OS_3l",
    "met_LD",#"mT_MEtLep1", "mT_MEtLep1",
]
