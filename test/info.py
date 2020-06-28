

inputPathNTuples='../data/'
treeDirName='hh_3l_OS_forBDTtraining/sel/evtntuple' # name of the tree directory inside ntuple;
                                                    #Format for tree inside ROOT file: "%s/%s/evtTree" % (treeDirName,sampleName)


keys=[
    'WZ', ## WZ inclusive sample + WZZ
    'signal_ggf_spin0_400_hh_4v', # signal 
]


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
