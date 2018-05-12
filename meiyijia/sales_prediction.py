import re
import time
import ipdb
import xlrd
import argparse
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import make_scorer

from data_processor import DateProcessing


#definitions of some transformation rules  
def ToWeight(y):  
    w = np.zeros(y.shape, dtype=float)  
    ind = y != 0  
    w[ind] = 1./(y[ind]**2)  
    return w  
  
def rmspe(yhat, y):  
    w = ToWeight(y)  
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))  
    return rmspe

  
def rmspe_xg(yhat, y):  
    # y = y.values  
    y = y.get_label()  
    #y = np.exp(y) - 1  
    #yhat = np.exp(yhat) - 1  
    w = ToWeight(y)  
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))  
    return "rmspe", rmspe 


def my_custom_loss_func(ground_truth, predictions):
    return np.sqrt(np.mean( (predictions/ground_truth - 1)**2 ))
    #return rmspe(predictions, ground_truth)

def get_data(args):
    data = DateProcessing()
    print('Preparing data ...')
    start = time.time()
    data.DAT2Matrix(args)
    end = time.time()
    print('Training data processed, time: {:.2f}s\n'.format(end - start))
    ipdb.set_trace()
    test_num = int(len(data.x_all) * 0.2)  
    if test_num > 100000:
        test_num = 100000
    num = len(data.x_all) - test_num
    x_train = data.x_all[: num]
    x_valid = data.x_all[num: ]
    y_train = data.y_all[: num]
    y_valid = data.y_all[num: ]
    return x_train, y_train, x_valid, y_valid


def main(args, x_train, y_train, x_valid, y_valid):
    if args.mode == 'xgb':
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_valid, label=y_valid)
        params = {"objective": "reg:linear", # linear regression 
                  "booster": "gbtree", # gradient boosting tree
                  "eta": args.learning_rate,  # learning rate
                  "max_depth": args.max_depth,  # max depth of gbtree
                  "subsample": args.subsample,  # subsampling rate
                  "colsample_bytree": 1.0,  #
                  "min_child_weight": args.min_child_weight,
                  "silent": 1,  
                  "seed": 1301,
                  'lambda': 2, # regularization  
                  }  
        watchlist = [(dtrain, 'train'), (dvalid, 'val')]
        model = xgb.train(params, dtrain, args.num_boost_round, evals=watchlist,
                    early_stopping_rounds=args.early_stop, feval=rmspe_xg, verbose_eval=True)
        
        print("\nValidating")  
        #ipdb.set_trace()
        yhat = model.predict(xgb.DMatrix(x_valid), ntree_limit=model.best_ntree_limit)

    else:
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
        params = {  'task': 'train',
                    'application': 'regression',
                    'boosting_type': 'gbdt',
                    'metric': {'l2_root', 'mape'},
                    'num_leaves': args.num_leaves,
                    'learning_rate': args.learning_rate,
                    'feature_fraction': 0.9,
                    'bagging_fraction': args.subsample,
                    #"min_child_weight": args.min_child_weight,
                    'bagging_freq': 5,
                    'verbose': 0,
                    'max_depth': -1,
                    'min_data_in_leaf': 20,
                    'lambda_l1': 0,
                    'lambda_l2': 1
                }
        model = lgb.train(params, lgb_train, args.num_boost_round, valid_sets=lgb_eval,
                    early_stopping_rounds=args.early_stop, verbose_eval=True)
        yhat = model.predict(x_valid, num_iteration=model.best_iteration)

    error = rmspe(yhat, y_valid)  
    print('RMSPE: {:.6f}\n'.format(error))
    for k,v in params.items():
        args.outfiler.write('{}: {}\n'.format(k, v))
    args.outfiler.write('best iteration: {} RMSPE: {:.6f}\n\n'.format(
        model.best_iteration, error))


def searching(args, x_train, y_train, x_valid, y_valid):
    param_test = {
        'learning_rate': [0.05, 0.1, 0.25, 0.5],
        #'n_estimators': list(range(50, 500, 50)),
        #'max_depth': list(range(3,10,2)),
        #'min_child_weight': list(range(1,6,2)),
    }
    if self.grid_search == 1:
        loss = make_scorer(my_custom_loss_func, greater_is_better=False)
        gs = GridSearchCV(estimator=XGBRegressor(max_depth=10, 
                                                learning_rate=0.05, 
                                                n_estimators=20, 
                                                silent=True, 
                                                objective='reg:linear', 
                                                nthread=-1, 
                                                gamma=0,
                                                min_child_weight=1, 
                                                max_delta_step=0, 
                                                subsample=0.9, 
                                                colsample_bytree=1.0, 
                                                colsample_bylevel=1, 
                                                reg_alpha=0, 
                                                reg_lambda=1, 
                                                scale_pos_weight=1, 
                                                seed=1301, 
                                                missing=None),
                        param_grid=param_test, n_jobs=-1, verbose=32, 
                        error_score='raise', scoring='neg_mean_squared_error')
        gs.fit(data.x_all, data.y_all)
        #ipdb.set_trace()
        for s in gs.grid_scores_:
            print('scores: {}'.format(s))
            args.outfiler.write('scores: {}\n'.format(s))
        print('best params: {}\nbest scores: {}\n'.format(gs.best_params_, gs.best_score_))
        args.outfiler.write('best params: {}\nbest scores: {}\n'.format(gs.best_params_, gs.best_score_))
    elif self.grid_search == 2:
        xlf = LGBMRegressor(num_leaves=50, 
                            max_depth=13, 
                            learning_rate=0.1,   
                            n_estimators=100, 
                            objective='regression', 
                            min_child_weight=1,   
                            subsample=0.8, 
                            colsample_bytree=0.8, 
                            nthread= -1,  
                            subsample_for_bin=200000, 
                            class_weight=None, 
                            min_split_gain=0.0, 
                            min_child_samples=20, 
                            subsample_freq=1, 
                            reg_alpha=0.0, 
                            reg_lambda=0.0, 
                            random_state=None, 
                            n_jobs=-1, 
                            silent=True)  
        xlf.fit(x_train, y_train, eval_metric='l2', verbose=True, 
            eval_set = [(x_valid, y_valid)])#, early_stopping_rounds=100)
        yhat = xlf.predict(x_valid)
        error = rmspe(yhat, y_valid)
        print('rmspe: {}\n'.format(error))
        args.outfiler.write('rmspe: {}\n'.format(error))
    else:
        for ne in range(10, 300, 20):
            xlf = XGBRegressor( max_depth=10, 
                                learning_rate=0.05, 
                                n_estimators=ne,
                                silent=True, 
                                objective='reg:linear', 
                                nthread=-1,
                                gamma=0, 
                                min_child_weight=1, 
                                max_delta_step=0,
                                subsample=0.9, 
                                colsample_bytree=1.0, 
                                colsample_bylevel=1,
                                reg_alpha=0, 
                                reg_lambda=1, 
                                scale_pos_weight=1,
                                seed=1301, 
                                missing=None)
            xlf.fit(x_train, y_train, eval_metric=rmspe_xg, verbose=True, 
                eval_set = [(x_valid, y_valid)])#, early_stopping_rounds=100)
            yhat = xlf.predict(x_valid)
            error = rmspe(yhat, y_valid)
            print('n_estimators: {}  rmspe: {}\n'.format(ne, error))
            args.outfiler.write('n_estimators: {}  rmspe: {}\n'.format(ne, error))


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default='main', help='')
    parser.add_argument("--mode", type=str, default='lgb', help='')
    parser.add_argument("--save_data", type=bool, default=False, help='')
    parser.add_argument("--avg_hist", type=bool, default=False, help='')
    parser.add_argument("--grid_search", type=int, default=0, help='')
    parser.add_argument("--num_boost_round", type=int, default=500, help='')
    parser.add_argument("--early_stop", type=int, default=20, help='')
    parser.add_argument("--max_depth", type=int, default=10, help='')
    parser.add_argument("--min_child_weight", type=int, default=5, help='')
    parser.add_argument("--num_leaves", type=int, default=31, help='')
    parser.add_argument("--max_data", type=int, default=1000000, help='')
    parser.add_argument("--learning_rate", type=float, default=0.05, help='')
    parser.add_argument("--subsample", type=float, default=0.9, help='')
    parser.add_argument("--result_dir", type=str, default='reg_md_mcw_ft30_lgb_100w_ss', help='')
    args = parser.parse_args()
    x_train, y_train, x_valid, y_valid = get_data(args)
    with open('results/%s.txt'%args.result_dir,'w') as args.outfiler:
        for k,v in args.__dict__.items():
            args.outfiler.write('{}: {}\n'.format(k, v))
        args.outfiler.write('\n')
        if args.func == 'main':
            if args.mode == 'lgb':
                for i in range(5, 9):
                    for j in range(5, 11):
                        args.subsample = j*0.1
                        args.num_leaves = 2**i - 1
                        #args.max_depth = i
                        #args.min_child_weight = j
                        #args.learning_rate = 0.05*(i+1)
                        main(args, x_train, y_train, x_valid, y_valid)
            else:
                for i in range(3, 11):
                    for j in range(1, 6):
                        #args.subsample = j*0.1
                        args.max_depth = i
                        args.min_child_weight = j
                        #args.learning_rate = 0.05*(i+1)
                        main(args, x_train, y_train, x_valid, y_valid)
        else:
            searching(args, x_train, y_train, x_valid, y_valid)
    end = time.time()
    print('\nTotal time cost: %.2fs\n' % (end - start))
