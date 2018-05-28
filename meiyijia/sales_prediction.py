import re
import time
import ipdb
import xlrd
import json
import argparse
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm 
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import make_scorer

from data_processor import DateProcessing, timeit
from environment import Environment
        

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


def mape(yhat, y):
    w = np.zeros(y.shape, dtype=float)  
    ind = y != 0  
    w[ind] = 1./ y[ind]
    mape = np.mean( np.abs(w * (y - yhat)) )
    return mape


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

@timeit
def get_data(args):
    data = DateProcessing()
    if args.predict_model == 'store':
        values = data.data2matrix_stores(args)
    else:
        values = data.data2matrix(args)
    values = [np.array(v, dtype=np.int32) for v in values]
    print([v.shape for v in values])
    [x_train, y_train, x_valid, y_valid, x_test, y_test] = values
    if args.save_data:
        print('Saving data/training_data.json ...')
        with open('data/training_data.json', 'w') as f:
            json.dump(values, f)
        print('Successfully saved file.')
    # print('Preparing data ...')
    # start = time.time()
    # data.DAT2Matrix(args)
    # end = time.time()
    # print('Training data processed, time: {:.2f}s\n'.format(end - start))
    # ipdb.set_trace()
    # test_num = int(len(data.x_all) * 0.2)  
    # if test_num > 100000:
    #     test_num = 100000
    # num = len(data.x_all) - test_num
    # x_train = [] #data.x_all[: num]
    # x_valid = [] #data.x_all[num: ]
    # x_test  = []
    # y_train = [] #data.y_all[: num]
    # y_valid = [] #data.y_all[num: ]
    # y_test  = []
    # ipdb.set_trace()
    # env = Environment(args, data)
    # env.restart(is_train=True)
    # for i in tqdm(xrange(args.max_data)):
    #     state = env.get_state()
    #     target = env.get_target()
    #     if i%10 == 0:
    #         x_valid.append(state)
    #         y_valid.append(target)
    #     else:
    #         x_train.append(state)
    #         y_train.append(target)
    #     terminal = env.boosting_step(is_train=True)
    #     if terminal:
    #         args.count_train_items += 1

    # env.restart(is_train=False)
    # for i in tqdm(xrange(100000)):
    #     state = env.get_state()
    #     target = env.get_target()
    #     x_test.append(state)
    #     y_test.append(target)
    #     terminal = env.boosting_step(is_train=False)
    #     if terminal:
    #         args.count_test_items += 1

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def main(args, x_train, y_train, x_valid, y_valid, x_test, y_test):
    log_results = {}
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
        
        print("\nTesting")  
        #ipdb.set_trace()
        yhat = model.predict(xgb.DMatrix(x_test), ntree_limit=model.best_ntree_limit)

    else:
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
        params = {  'task': 'train',
                    'application': 'regression',
                    'boosting_type': 'gbdt',
                    'metric': {'l2_root', 'mape'},
                    'num_leaves': args.num_leaves, # 255
                    'learning_rate': args.learning_rate,
                    'feature_fraction': 0.9,
                    'bagging_fraction': args.subsample, # 0.5, 0.6
                    #"min_child_weight": args.min_child_weight,
                    'bagging_freq': 5,
                    'verbose': 0,
                    'max_depth': -1,
                    'min_data_in_leaf': 20,
                    'lambda_l1': 0,
                    'lambda_l2': 1
                }
        model = lgb.train(params, lgb_train, args.num_boost_round, valid_sets=lgb_eval,
                    early_stopping_rounds=args.early_stop, verbose_eval=True, evals_result=log_results)
        yhat = model.predict(x_test, num_iteration=model.best_iteration)

    #ipdb.set_trace()
    yhat_clip = (0 < yhat) * yhat # 0 < yhat returns bool array  
    with open('results/%s.predict.txt'%args.result_dir, 'a') as f:
        f.write('\ny_real \t y_clip_predict \t y_raw_predict\n')
        for i in xrange(len(yhat)):
            f.write('{:<10} {:<10} {:<10}\n'.format(int(y_test[i]), int(yhat_clip[i]), int(yhat[i])))
    result_logger(model, args.log_step, args, x_valid, y_valid, x_test, y_test)
    error = rmspe(yhat_clip, y_test)  
    error_mape = mape(yhat_clip, y_test)
    print('RMSPE: {:<10.6f} MAPE: {:<10.6f}\n'.format(error, error_mape))
    for k,v in params.items():
        args.outfiler.write('{}: {}\n'.format(k, v))
    args.outfiler.write('best iteration: {} RMSPE: {:<10.6f} MAPE: {:<10.6f}\n\n'.format(
        model.best_iteration, error, error_mape))

    if args.save_model != '':
        model_json = model.dump_model()
        #model.save_model('models/%s.model')
        #model.dump_model('models/%s.raw.txt') # dump model
        json.dump(model_json, open('models/%s.raw.txt'%args.save_model, 'w'), indent=4)#, 
                        #'models/%s_featmap.txt'%args.save_model)# dump model with feature map
    return error, error_mape


def result_logger(model, step, args, x_valid, y_valid, x_test, y_test):
    print('Logging results ...')
    with open('results/%s.log'%args.result_dir, 'a') as f:
        f.write('log_mape results:\n')
        for i in xrange(step, model.best_iteration+args.early_stop+1, step):
            yhat_v = model.predict(x_valid, num_iteration=i)
            yhat_vclip = (0 < yhat_v) * yhat_v
            error_v = mape(yhat_vclip, y_valid)

            yhat_t = model.predict(x_test, num_iteration=i)
            yhat_tclip = (0 < yhat_t) * yhat_t
            error_t = mape(yhat_tclip, y_test)
            f.write('{:<10.6f}  {:<10.6f}\n'.format(error_v, error_t))


def searching(args, x_train, y_train, x_valid, y_valid, x_test, y_test):
    param_test = {
        'learning_rate': [0.05, 0.1, 0.25, 0.5],
        #'n_estimators': list(xrange(50, 500, 50)),
        #'max_depth': list(xrange(3,10,2)),
        #'min_child_weight': list(xrange(1,6,2)),
    }
    if args.grid_search == 1:
        x_all = np.concatenate([x_train, x_valid], axis=0)
        y_all = np.concatenate([y_train, y_valid], axis=0)
        ipdb.set_trace()
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
        gs.fit(x_all, y_all)
        #ipdb.set_trace()
        for s in gs.grid_scores_:
            print('scores: {}'.format(s))
            args.outfiler.write('scores: {}\n'.format(s))
        print('best params: {}\nbest scores: {}\n'.format(gs.best_params_, gs.best_score_))
        args.outfiler.write('best params: {}\nbest scores: {}\n'.format(gs.best_params_, gs.best_score_))
    elif args.grid_search == 2:
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
        yhat = xlf.predict(x_test)
        error = rmspe(yhat, y_test)
        print('rmspe: {}\n'.format(error))
        args.outfiler.write('rmspe: {}\n'.format(error))
    else:
        for ne in xrange(10, 300, 20):
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
            yhat = xlf.predict(x_test)
            error = rmspe(yhat, y_test)
            print('n_estimators: {}  rmspe: {}\n'.format(ne, error))
            args.outfiler.write('n_estimators: {}  rmspe: {}\n'.format(ne, error))


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('-train_start_date',            type=str, default='2017-01-01')
    parser.add_argument('-train_end_date',              type=str, default='2017-12-31')
    parser.add_argument('-test_start_date',             type=str, default='2018-01-01')
    parser.add_argument('-test_end_date',               type=str, default='2018-03-31')
    parser.add_argument('-count_items_of_missing_day',  type=int, default=0)
    parser.add_argument('-count_train_items',           type=int, default=0)
    parser.add_argument('-count_test_items',            type=int, default=0)
    parser.add_argument('-min_stores',                  type=int, default=0)
    parser.add_argument('-use_padding',                 type=int, default=0)
    parser.add_argument('-random',                      type=int, default=0)
    parser.add_argument('-onehot',                      type=int, default=1)
    # model
    parser.add_argument("--grid_search",        type=int, default=2, help='')
    parser.add_argument("--num_boost_round",    type=int, default=500, help='')
    parser.add_argument("--early_stop",         type=int, default=50, help='')
    parser.add_argument("--max_depth",          type=int, default=9, help='')
    parser.add_argument("--min_child_weight",   type=int, default=3, help='')
    parser.add_argument("--num_leaves",         type=int, default=63, help='')
    parser.add_argument("--max_data",           type=int, default=2000000, help='')
    parser.add_argument("--learning_rate",      type=float, default=0.05, help='')
    parser.add_argument("--subsample",          type=float, default=0.8, help='')
    # main
    parser.add_argument("--func",               type=str, default='main', help='')
    parser.add_argument("--mode",               type=str, default='lgb', help='')
    parser.add_argument("--predict_model",      type=str, default='store', help='')
    parser.add_argument("--log_step",           type=int, default=1, help='')
    parser.add_argument("--save_data",          type=int, default=0, help='')
    parser.add_argument("--avg_hist",           type=bool, default=False, help='')
    parser.add_argument("--valid_days",         type=list, default=[31, 90], help='')
    parser.add_argument("--valid_split",        type=int, default=8, help='')
    parser.add_argument("--save_model",         type=str, default='best_lgb_model', help='')
    parser.add_argument("--result_dir",         type=str, default='leaves63_train500_50_1_valid31_90_item_id_onehot1', help='')
    args = parser.parse_args()
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_data(args)
    with open('results/%s.txt'%args.result_dir,'w') as args.outfiler:
        for k,v in args.__dict__.items():
            args.outfiler.write('{}: {}\n'.format(k, v))
        args.outfiler.write('\n')
        if args.func == 'main':
            if args.mode == 'lgb':
                # lest_error = None
                # best_args = {}
                # for i in xrange(5, 9):
                #     for j in xrange(5, 11):
                #         args.subsample = j*0.1
                #         args.num_leaves = 2**i - 1
                # #         # args.max_depth = i
                # #         # args.min_child_weight = j
                # #         # args.learning_rate = 0.005*(i+1)
                _, e = main(args, x_train, y_train, x_valid, y_valid, x_test, y_test)
                #         if lest_error == None or lest_error > e:
                #             lest_error = e
                #             best_args['subsample'] = args.subsample
                #             best_args['num_leaves'] = args.num_leaves
                # print(best_args)
            else:
                # for i in xrange(3, 11):
                #     for j in xrange(1, 6):
                #         #args.subsample = j*0.1
                #         args.max_depth = i
                #         args.min_child_weight = j
                #         #args.learning_rate = 0.05*(i+1)
                main(args, x_train, y_train, x_valid, y_valid, x_test, y_test)
        else:
            for args.grid_search in xrange(1, 3):
                args.outfiler.write('\n\n args.grid_search: %d\n'%args.grid_search)
                searching(args, x_train, y_train, x_valid, y_valid, x_test, y_test)
    end = time.time()
    print('\nTotal time cost: %.2fs\n' % (end - start))
