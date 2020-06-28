import os
import numpy as np
import ROOT
import math , array
from root_numpy import tree2array
import glob

def load_data(
        inputPathNTuples, 
        treeDirName,
        variables) :
    print "In data_manager::load_data()::\n inputPathNTuples: ",inputPathNTuples, "\n treeDirName: ",treeDirName
    print " variables: ",variables
    
    my_cols_list=variables+['proces', 'key', 'target', "totalWeight"]
    data = pandas.DataFrame(columns=my_cols_list) ## right now an empty dataframe with columns = my_cols_list
    print "data: ",data
    
    target = None
    for process in keys :
        print 'process %s ' % (process)

        if 'WZ' in process:
            sampleName = "WZ"
            target = 0
        if 'signal' in process:
            sampleName = "signal_ggf_spin0_400_hh_wwww"
            target = 1

        inputNTuples = glob.glob("%s/%s*_forBDTtraining.root" % (inputPathNTuples, process))
        inputTree = "%s/%s/evtTree" % (treeDirName,sampleName)
        print "inputTree",inputTree,",  len(inputNTuples):",len(inputNTuples), "  inputNTuples: ",inputNTuples

        for intuple in range(0, len(inputNTuples)):
            try: tfile = ROOT.TFile(inputNTuples[intuple])
            except :
                print "%s   FAIL load root file" % inputNTuples[intuple]
                continue
            try: tree = tfile.Get(inputTree)
            except :
                print (inputTree, "FAIL read inputTree", tfile)
                continue
            if tree is not None :
                print "sampleName: ",sampleName,",  process: ",process, ", inputNTuples[intuple]: ",inputNTuples[intuple], ", nEvents: ",tree.GetEntries()
                try: chunk_arr = tree2array(tree)
                except :
                    print (inputTree, "FAIL tree2array ", tfile)
                    tfile.Close()
                    continue
                else :
                    chunk_df = pandas.DataFrame(chunk_arr, columns=variables)
                    tfile.Close()
                    chunk_df['proces']=sampleName
                    chunk_df['key']=process
                    chunk_df['target']=target
                    chunk_df["totalWeight"] = chunk_df["evtWeight"]
                    #print "chunk_df: ",chunk_df
                    data=data.append(chunk_df, ignore_index=True)
            else : print ("file "+list[ii]+"was empty")
        nS = len(data.ix[(data.target.values == 1) & (data.key.values==process) ])
        nB = len(data.ix[(data.target.values == 0) & (data.key.values==process) ])
        print "%s  signal size %g,  bk size %g,   evtWeight %g,  totalWeight %g" % (process, nS,nB, data.ix[ (data.key.values==process)]["evtWeight"].sum(), data.ix[(data.key.values==process)]["totalWeight"].sum())
        nNW = len(data.ix[(data["totalWeight"].values < 0) & (data.key.values==process) ])
        print process, " no. of events with -ve weights", nNW
        
    #print 'data to list = ', (data.columns.values.tolist())
    n = len(data)
    nS = len(data.ix[data.target.values == 1])
    nB = len(data.ix[data.target.values == 0])
    print treeDirName," size of sig, bkg: ", nS, nB
    return data
