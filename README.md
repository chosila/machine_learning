# machine_learning

# 1. BDT training with xgboost:
BDT training example training on one signal and one background process.

test/info.py:
File containing details of
i) path of input ROOT trees of signal and background for BDT-training
ii) list of variables (named 'keys') to use for BDT training (trainVars);  trainVarsAll: list of all variables which can be considered for BDT training. trainVars is subset of trainVarsAll

data/:
Signal and background trees for BDT training are stored in data/ directory

python/data_manager.py:
It reads variables from signal and background trees and load them into pandas.DataFrame

test/sklearn_xgboost_training.py:
It perform BDT training. It can be run in the follwoing two modes:

I] Do BDT training for a choosen/trial set of hyper-parameters of training:
For e.g.
```
python sklearn_xgboost_training.py  --ntrees 800 --treeDeph 2 --lr 0.01  --mcw 1
```

II] Optimum hyper-parameter (grid) search:
Perform BDT training for different values for different hyperparameters (grid-search). Evaluate BDT performance by choosen parameter (for e.g. area under ROD curve).
Return best combination for hyperparameters. Use this best combination of hyperparameter for BDT training with mode I]
```
python sklearn_xgboost_training.py  --HypOpt True
```

BDT performance plots are stored in test/plots directory


Note: Many other BDT hyperparameters, which have not been used in the example code, can also be used for BDT optimization.

Additional material:
Link explaining XGBoost hyperparameters: [https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/]