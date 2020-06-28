
import os
from datetime import datetime
import sys , time

import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import sklearn
from sklearn import datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import psutil



## Command to run:
# python sklearn_xgboost_training.py  --ntrees 800 --treeDeph 2 --lr 0.01  --mcw 1



from optparse import OptionParser
parser = OptionParser()
parser.add_option("--ntrees", type="int", dest="ntrees", help="hyp", default=2000) #1500
parser.add_option("--treeDeph", type="int", dest="treeDeph", help="hyp", default=2) #3
parser.add_option("--lr", type="float", dest="lr", help="hyp", default=0.01)
parser.add_option("--mcw", type="float", dest="mcw", help="hyp", default=1)
#parser.add_option("--gridSearchCV", action="store_true", dest="gridSearchCV", help="Search optimal values for XGB training parameters", default=False)
parser.add_option("--doXML", action="store_true", dest="doXML", help="Do save not write the xml file", default=False)
parser.add_option("--HypOpt", action="store_true", dest="HypOpt", help="If you call this will not do plots with repport", default=False)
(options, args) = parser.parse_args()




## to GridSearchCV (options.HypOpt==True) the test_size should not be smaller than 0.4 == it is used for cross validation!
## to final BDT fit test_size can go down to 0.1 without sign of overtraining
test_size = 0.4



print "--ntrees ",options.ntrees,"  --treeDeph ",options.treeDeph,"  --lr ",options.lr,"    --mcw ",options.mcw
hyppar="ntrees_"+str(options.ntrees)+"_deph_"+str(options.treeDeph)+"_mcw_"+str(options.mcw)+"_lr_"+str(int(options.lr))
startTime = datetime.now()
execfile("info.py")
execfile("../python/data_manager.py")

weights="totalWeight"
target='target'


fileLog_ = open('roc.log','w+')

print "sklearn_xgboost_training:: "
data = load_data(inputPathNTuples, treeDirName, trainVarsAll)


dataOriginal = data.copy()

print "\n\nEvent table before normalizing:: \n %30s  %10s   %s " % ("Process", "nEvents", "nEventsWeighted")
for key in keys:
    nEvents = int(len(data.loc[(data['key']==key)]))
    nEventsWtg = float(data.loc[(data['key']==key), [weights]].sum())
    #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
    print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)




# normalize total signal w.r.t. total background
data.loc[data['target']==0, [weights]] *= 100000/data.loc[data['target']==0][weights].sum()
data.loc[data['target']==1, [weights]] *= 100000/data.loc[data['target']==1][weights].sum()


print "\n\nEvent table after normalizing:: \n %30s  %10s   %s " % ("Process", "nEvents", "nEventsWeighted")
for key in keys:
    nEvents = int(len(data.loc[(data['key']==key)]))
    nEventsWtg = float(data.loc[(data['key']==key), [weights]].sum())
    #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
    print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)


# drop events with NaN weights - for safety
data.dropna(subset=[weights],inplace = True) # data
data.fillna(0)

print "length of sig, bkg without NaN: bk:", len(data.loc[data.target.values == 0]),", sig:", len(data.loc[data.target.values == 1])
#################################################################################



# make plots for variables---------------------------------------------------------
dataSig = data.ix[data.target.values == 1]
dataBk  = data.ix[data.target.values == 0]


for feature in trainVars:
    print "make plot: ", feature
    hist_params = {'normed': True, 'histtype': 'bar', 'fill': True , 'lw':3, 'alpha' : 0.4}
    nbins = 8
    min_valueS, max_valueS = np.percentile(dataSig[feature], [0.0, 99])
    min_valueB, max_valueB = np.percentile(dataBk[feature], [0.0, 99])
    print('feather: {} \t min_value {}, max_value {},min_value2 {}, max_value2 {}'.format(feature, min_valueS,max_valueS,min_valueB, max_valueB))
    range_local = (min(min_valueS,min_valueB),  max(max_valueS,max_valueB))

    valuesS, binsS, _ = plt.hist(
        dataSig[feature].values,
        weights = dataSig[weights].values.astype(np.float64),
        range = range_local,
        bins = nbins, edgecolor='b', color='b',
        label = "Signal", **hist_params
    )
    to_ymax = max(valuesS)
    to_ymin = min(valuesS)

    valuesB, binsB, _ = plt.hist(
        dataBk[feature].values,
        weights = dataBk[weights].values.astype(np.float64),
        range = range_local,
        bins = nbins, edgecolor='g', color='g',
        label = "Background", **hist_params
    )
    to_ymax2 = max(valuesB)
    to_ymax  = max([to_ymax2, to_ymax])
    to_ymin2 = min(valuesB)
    to_ymin  = max([to_ymin2, to_ymin])
    plt.ylim(ymin=to_ymin*0.1, ymax=to_ymax*1.2)
    #plt.yscale('log')
    #plt.yscale('linear')
    plt.legend(loc='best')
    plt.xlabel(feature)
    plt.savefig("plots/plot_%s.png" % feature)
    plt.clf()


#############################################################################################  
# split data for train and test 
print("\n\nTraining dataset: {},   Validation dataest: {}".format((1-test_size), test_size))
traindataset, valdataset  = train_test_split(data[trainVars + ["target","totalWeight","key"]], test_size=test_size, random_state=7)


print 'nEvents for BDT in full       data: %i,  bk: %i,  signal: %i' % (len(data), len(data.loc[data.target.values == 0]), len(data.loc[data.target.values == 1]) )
print 'nEvents for BDT in training   data: %i,  bk: %i,  signal: %i' % (len(traindataset), len(traindataset.loc[traindataset.target.values == 0]), len(traindataset.loc[traindataset.target.values == 1]) )
print 'nEvents for BDT in validation data: %i,  bk: %i,  signal: %i' % (len(valdataset), len(valdataset.loc[valdataset.target.values == 0]), len(valdataset.loc[valdataset.target.values == 1]) )

print 'Tot weight of train and validation for signal= ', traindataset.loc[traindataset[target]==1]["totalWeight"].sum(), valdataset.loc[valdataset[target]==1]["totalWeight"].sum()
print 'Tot weight of train and validation for bkg= ', traindataset.loc[traindataset[target]==0]['totalWeight'].sum(),valdataset.loc[valdataset[target]==0]['totalWeight'].sum()


#############################################################################################  
NormalizeTrainTestData = True
if NormalizeTrainTestData:
    print("\n\nCheck how train_test_split data background and signals are normalized::: *****")
    order_train1 = [traindataset, valdataset]
    order_train1_name = ["train", "test"]
    for idx, data_do in enumerate(order_train1) :
        print "\n\ndata %i %s \nEvent table :: \n %30s  %10s   %s " % (idx,order_train1_name[idx],"Process", "nEvents", "nEventsWeighted")
        for key in keys:
            nEvents = int(len(data_do.loc[(data_do['key']==key)]))
            nEventsWtg = float(data_do.loc[(data_do['key']==key), [weights]].sum())
            #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
            print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)
        
        data_do.loc[data_do['target']==0, [weights]] *= 100000/data_do.loc[data_do['target']==0][weights].sum()
        data_do.loc[data_do['target']==1, [weights]] *= 100000/data_do.loc[data_do['target']==1][weights].sum()
        
        print "\n\ndata %i %s \nEvent table after normalizing and scaling up by 1e5:: \n %30s  %10s   %s " % (idx,order_train1_name[idx],"Process", "nEvents", "nEventsWeighted")
        for key in keys:
            nEvents = int(len(data_do.loc[(data_do['key']==key)]))
            nEventsWtg = float(data_do.loc[(data_do['key']==key), [weights]].sum())
            #print " \n\n",key," sum: ",len(data.loc[(data['key']==key)]),", \t ",data.loc[(data['key']==key), [weights]].sum()
            print " %30s  %10i  %10.3f" % (key,nEvents,nEventsWtg)
        
        print("After normalizing and scaling::")        
        print("nEvents for BDT: %i,  bk: %i,  signal: %i" % (len(data_do), len(data_do.loc[data_do['target']==0]), len(data_do.loc[data_do['target']==1])))
        print("nEventsweight  : %f,  bk: %f,  signal: %f" % (data_do[weights].sum(), data_do.loc[data_do['target']==0][weights].sum(), data_do.loc[data_do['target']==1][weights].sum()))




## to GridSearchCV the test_size should not be smaller than 0.4 == it is used for cross validation!
## to final BDT fit test_size can go down to 0.1 without sign of overtraining
#############################################################################################
## Training parameters
if options.HypOpt==True :
    # http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    param_grid = {
    	'n_estimators': [500, 800],#[200,500,800, 1000,2500],
    	'min_child_weight': [1,100],
    	'max_depth': [2,3], #[1,2,3,4],
    	'learning_rate': [0.1, 0.01], #[0.01,0.02,0.03]
    }
    scoring = "roc_auc" # BDT performance evaluation parameter.
                        # roc_aur: area under ROC curve
                        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    early_stopping_rounds = 200 # Will train until validation_0-auc hasn't improved in 100 rounds.
    cv=3
    cls = xgb.XGBClassifier()
    fit_params = { "eval_set" : [(valdataset[trainVars].values,valdataset[target])],
                   "eval_metric" : "auc", 
                   "early_stopping_rounds" : early_stopping_rounds,
		   'sample_weight': valdataset[weights].values }
    gs = GridSearchCV(cls, param_grid, scoring, fit_params, cv = cv, verbose = 0)
    gs.fit(traindataset[trainVars].values,
           traindataset.target.astype(np.bool)
    )
    for i, param in enumerate(gs.cv_results_["params"]):
        print("params : {} \n    cv auc = {}  +- {} ".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]))
    print("best parameters",gs.best_params_)
    print("best score",gs.best_score_)
    #print("best iteration",gs.best_iteration_)
    #print("best ntree limit",gs.best_ntree_limit_)
    file = open("plots/XGB_HyperParameterGridSearch_GSCV.log","w")
    file.write(
	str(trainVars)+"\n"+
	"best parameters"+str(gs.best_params_) + "\n"+
	"best score"+str(gs.best_score_)+ "\n"
	#"best iteration"+str(gs.best_iteration_)+ "\n"+
	#"best ntree limit"+str(gs.best_ntree_limit_)
    )
    for i, param in enumerate(gs.cv_results_["params"]):
	file.write("params : {} \n    cv auc = {}  +- {} {}".format(param,gs.cv_results_["mean_test_score"][i],gs.cv_results_["std_test_score"][i]," \n"))
    file.close()



#############################################################################################        
cls = xgb.XGBClassifier(
			n_estimators = options.ntrees,
			max_depth = options.treeDeph,
			min_child_weight = options.mcw, # min_samples_leaf
			learning_rate = options.lr,
			#max_features = 'sqrt',
			#min_samples_leaf = 100
			#objective='binary:logistic', #booster='gbtree',
			#gamma=0, #min_child_weight=1,
			#max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, #random_state=0
			)
cls.fit(
	traindataset[trainVars].values,
	traindataset.target.astype(np.bool),
	sample_weight=(traindataset[weights].astype(np.float64))
	# more diagnosis, in case
	#eval_set=[(traindataset[trainVars(False)].values,  traindataset.target.astype(np.bool),traindataset[weights].astype(np.float64)),
	#(valdataset[trainVars(False)].values,  valdataset.target.astype(np.bool), valdataset[weights].astype(np.float64))] ,
	#verbose=True,eval_metric="auc"
	)

#print trainVars(False)
print 'traindataset[trainVars].columns.values.tolist() : ', traindataset[trainVars].columns.values.tolist()



print ("XGBoost trained")
proba = cls.predict_proba(traindataset[trainVars].values )
fpr, tpr, thresholds = roc_curve(traindataset[target].astype(int), proba[:,1], sample_weight=(traindataset[weights].astype(np.float64)) )
train_auc = auc(fpr, tpr, reorder = True)
print("XGBoost train set auc - {}".format(train_auc))
proba = cls.predict_proba(valdataset[trainVars].values )
fprt, tprt, thresholds = roc_curve(valdataset[target].astype(int), proba[:,1], sample_weight=(valdataset[weights].astype(np.float64))  )
test_auct = auc(fprt, tprt, reorder = True)
print("XGBoost test set auc - {}".format(test_auct))
fileLog_.write("XGBoost_train = %0.8f\n" %train_auc)
fileLog_.write("XGBoost_test = %0.8f\n" %test_auct)
fig, ax = plt.subplots()
#f_score_dict =cls.booster().get_fscore()


pklpath="XGB_classifier_"+str(len(trainVars))+"Var"
print ("Done  ",pklpath,hyppar)
if options.doXML==True :
    print ("Date: ", time.asctime( time.localtime(time.time()) ))
    pickle.dump(cls, open(pklpath+".pkl", 'wb'))
    file = open(pklpath+"pkl.log","w")
    file.write(str(trainVars(False))+"\n")
    file.close()
    print ("saved ",pklpath+".pkl")
    print ("variables are: ",pklpath+"_pkl.log")




##################################################
## Draw ROC curve
fig, ax = plt.subplots(figsize=(8, 8))
train_auc = auc(fpr, tpr, reorder = True)
#ax.plot(fpr, tpr, lw=1, color='g',label='XGB train all mass excluding 350GeV(area = %0.5f)'%(train_auc))
#ax.plot(fprt, tprt, lw=1, ls='--',color='g',label='XGB test excluding 350GeV(area = %0.5f)'%(test_auct))
#ax.plot(fprtightF, tprtightF, lw=1, label='XGB test - Fullsim All (area = %0.3f)'%(test_auctightF))
ax.plot(fpr, tpr, lw=1, color='g',label='XGB train (area = %0.5f)'%(train_auc))
ax.plot(fprt, tprt, lw=1, ls='--',color='g',label='XGB test (area = %0.5f)'%(test_auct) )
ax.set_ylim([0.0,1.0])
ax.set_xlim([0.0,1.0])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
ax.grid()
fig.savefig("plots/%s_roc.png" % hyppar)


###########################################################################
## feature importance plot
fig, ax = plt.subplots()
#f_score_dict =cls.booster().get_fscore()
f_score_dict =cls.get_booster().get_fscore()
print("f_score_dict: {}".format(f_score_dict))
f_score_dict = {trainVars[int(k[1:])] : v for k,v in f_score_dict.items()}
feat_imp = pandas.Series(f_score_dict).sort_values(ascending=True)
feat_imp.plot(kind='barh', title='Feature Importances')
fig.tight_layout()
fig.savefig("plots/%s_XGB_importance.png" % hyppar)


###########################################################################
## classifier o/p plot
hist_params = {'normed': True, 'bins': 10 , 'histtype':'step'}
plt.clf()
y_pred  = cls.predict_proba(valdataset.ix[valdataset.target.values == 0, trainVars].values)[:, 1] #
y_predS = cls.predict_proba(valdataset.ix[valdataset.target.values == 1, trainVars].values)[:, 1] #
plt.figure('XGB',figsize=(6, 6))
values, bins, _ = plt.hist(y_pred ,  label="Background", **hist_params)
values, bins, _ = plt.hist(y_predS , label="Signal", **hist_params )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig("plots/%s_XGB_classifier.png" % hyppar)


###########################################################################
# plot correlation matrix
if options.HypOpt==False :
    for ii in [1,2] :
        if ii == 1 :
            datad=traindataset.loc[traindataset[target].values == 1]
            label="Signal"
        else:
            datad=traindataset.loc[traindataset[target].values == 0]
            label="Background"
        datacorr = datad[trainVars].astype(float)  #.loc[:,trainVars(False)] #dataHToNobbCSV[[trainVars(True)]]
        correlations = datacorr.corr()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlations, vmin=-1, vmax=1)
        ticks = np.arange(0,len(trainVars),1)
        plt.rc('axes', labelsize=8)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(trainVars,rotation=45)
        ax.set_yticklabels(trainVars)
        ax.tick_params(top=False, bottom=True,
                   labeltop=False, labelbottom=True)
        fig.colorbar(cax)
        fig.tight_layout()
        #plt.subplots_adjust(left=0.9, right=0.9, top=0.9, bottom=0.1)
        plt.savefig("plots/%s_corr_%s.png" % (hyppar,label))
        ax.clear()
process = psutil.Process(os.getpid())
print("process.memory_info().rss: ",process.memory_info().rss)
print("Execution time: ",datetime.now() - startTime)


print "sklearn_xgboost_training:: Done "
