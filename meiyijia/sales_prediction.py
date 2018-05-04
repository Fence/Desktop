import re
import time
import ipdb
import xlrd
import argparse
import numpy as np
import datetime as dt
import lightgbm as lgb
import xgboost as xgb
from datetime import datetime as dtdt
from lightgbm.sklearn import LGBMRegressor
from xgboost.sklearn import XGBRegressor
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import make_scorer

class DateProcessing(object):
    """docstring for DateProcessing"""
    def __init__(self, max_data):
        self.max_data = max_data
        workbook = xlrd.open_workbook('basic_data.xlsx')
        #names = workbook.sheet_names()
        #['门店', '商品', '供应商', '天气', '半月档促销商品', '门店仓位']
        self.stores_wb = workbook.sheet_by_name('门店')
        self.goods_wb = workbook.sheet_by_name('商品')
        self.weather_wb = workbook.sheet_by_name('天气')
        self.suppliers_wb = workbook.sheet_by_name('供应商')
        self.promotions_wb = workbook.sheet_by_name('半月档促销商品')
        self.warehouses_wb = workbook.sheet_by_name('门店仓位')

        # self.get_sales_volume_by_day()
        # self.get_stock_and_return_by_day()
        #ipdb.set_trace()
        self.get_stores()
        self.get_goods()
        self.get_suppliers()
        self.get_weather()
        self.get_promotions()
        self.get_warehouses()
        #self.DAT2Matrix()

    def DAT2Matrix(self):
        x_all = []
        y_all = []
        count_error = 0
        #ipdb.set_trace()
        for line in open('sales_volume_by_day.DAT'):
            if len(y_all) >= self.max_data:
                break
            try:
                date, store_id, goods_id, sales_volume = line.split('\t')
                year, month, day = date.split('-')
                tmp_x = list(self.date_transformation(year, month, day))
                city = self.city2int[self.stores[store_id][0]]
                weather = self.weather[date.replace('-','')][city]
                date = dtdt.strptime(date, '%Y-%m-%d')
                promotion = [0]*8
                if goods_id in self.promotions:
                    for p in self.promotions[goods_id]:
                        # if this item is promoted at this day
                        if p[0] <= date <= p[1]: 
                            promotion = [p[0].year, p[0].month, p[0].day] 
                            promotion = promotion + [p[1].year, p[1].month, p[1].day] + p[2:]
                            break
                tmp_x.extend([int(store_id), int(goods_id), city])
                tmp_x.extend(weather)
                tmp_x.extend(promotion)
                x_all.append(np.array(tmp_x, dtype=np.float32))
                y_all.append(float(sales_volume))
            except Exception as e: 
                # some store_ids are missing, so store_id might be ''
                count_error += 1
                #print(e, count_error)
        self.x_all = np.array(x_all)
        self.y_all = np.array(y_all)
        print('x shape:{}  y shape:{} errors:{}\n'.format(self.x_all.shape, self.y_all.shape, count_error))


    def date_transformation(self, year, month, day):
        # 1~7 denote Monday~Sunday
        # 1~4 denote Spring~Winter
        year, month, day = int(year), int(month), int(day)
        weekday = dt.date.weekday(dt.date(year, month, day)) + 1 
        if 3 <= month <= 5:
            season = 1
        elif 6 <= month <= 8:
            season = 2
        elif 9 <= month <= 11:
            season = 3
        else:
            season = 4
        return year, month, day, season, weekday


    def get_sales_volume_by_day(self):
        data = {}
        dates = []
        count = 0
        for line in open('门店商品销售流水数据_2017.DAT'):
            count += 1
            if count %1000000 == 0:
                print(count)
            items = line.split()
            date = items[0]
            _, store_id, goods_id, sales_volume = items[1].split('|')
            if date in data:
                if (store_id, goods_id) in data[date]:
                    data[date][(store_id, goods_id)] += float(sales_volume)
                else:
                    data[date][(store_id, goods_id)] = float(sales_volume)
            else:
                dates.append(date)
                data[date] = {(store_id, goods_id): float(sales_volume)}

        for line in open('门店商品销售流水数据_2018.DAT'):
            count += 1
            if count %1000000 == 0:
                print(count)
            items = line.split()
            date = items[0]
            _, store_id, goods_id, sales_volume = items[1].split('|')
            if date in data:
                if (store_id, goods_id) in data[date]:
                    data[date][(store_id, goods_id)] += float(sales_volume)
                else:
                    data[date][(store_id, goods_id)] = float(sales_volume)
            else:
                data[date] = {(store_id, goods_id): float(sales_volume)}
        print('days of sales volume: %d' % len(data))
        with open('sales_volume_by_day.DAT','w') as f:
            for date in dates:
                for (store_id, goods_id), sales_volume in data[date].items():
                    f.write('{}\t{}\t{}\t{}\n'.format(date, store_id, goods_id, sales_volume))
            print('Successfully save sales_volume_by_day.DAT\n')


    def get_stock_and_return_by_day(self):
        data = {}
        dates = []
        count = 0
        for line in open('门店商品进退货数据.DAT'):
            count += 1
            if count %1000000 == 0:
                print(count)
            items = line.split()
            date = items[0]
            _, store_id, goods_id, stock_volume, return_volume = items[1].split('|')
            if date in data:
                if (store_id, goods_id) in data[date]:
                    data[date][(store_id, goods_id)][0] += float(stock_volume)
                    data[date][(store_id, goods_id)][1] += float(return_volume)
                else:
                    data[date][(store_id, goods_id)] = [float(stock_volume), float(return_volume)]
            else:
                dates.append(date)
                data[date] = {(store_id, goods_id): [float(stock_volume), float(return_volume)]}
        print('days of stock and return volume: %d' % len(data))
        with open('stock_and_return_volume_by_day.DAT','w') as f:
            for date in dates:
                for (store_id, goods_id),[stock_volume, return_volume] in data[date].items():
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(date, store_id, goods_id, stock_volume, return_volume))
            print('Successfully save stock_and_return_volume_by_day.DAT\n')


    def get_stores(self):
        self.stores = {} # {store_id: (city, delivery_cycle) }
        for i in range(1, self.stores_wb.nrows):
            store_id = self.stores_wb.cell(i, 0).value
            if store_id not in self.stores:
                self.stores[store_id] = self.stores_wb.row_values(i, 1, 3)
        print('stores: %d' % len(self.stores))


    def get_goods(self):
        self.goods = {} # {goods_id: [labels, supplier_id] }
        for i in range(1, self.goods_wb.nrows):
            goods_id = self.goods_wb.cell(i, 0).value
            supplier_id = self.goods_wb.cell(i, self.goods_wb.ncols - 1).value
            if goods_id not in self.goods:
                self.goods[goods_id] = self.goods_wb.row_values(i, 1, None)[0::2]
                self.goods[goods_id].append(supplier_id)
        print('goods: %d' % len(self.goods))


    def get_suppliers(self):
        self.suppliers = {} # {supplier_id: {warehouse_id: [data]}}
        for i in range(1, self.suppliers_wb.nrows):
            warehouse_id = self.suppliers_wb.cell(i, 0).value
            supplier_id = self.suppliers_wb.cell(i, 1).value
            data = self.suppliers_wb.row_values(i, 2, None)
            if supplier_id in self.suppliers:
                if warehouse_id not in self.suppliers[supplier_id]:
                    self.suppliers[supplier_id][warehouse_id] = [data]
                else:
                    self.suppliers[supplier_id][warehouse_id].append(data)
            else:
                self.suppliers[supplier_id] = {warehouse_id: [data]}
        print('suppliers: %d' % len(self.suppliers))


    def get_weather(self):
        self.weather = {} # {date: {city: [weather]} }
        self.weather2int = [{},{},{},{},{},{},{}]
        self.city2int = {}
        for i in range(1, self.weather_wb.nrows):
            date = self.weather_wb.cell(i, 0).value
            city = self.weather_wb.cell(i, 1).value
            weather = self.weather_wb.row_values(i, 2, None)
            if city not in self.city2int:
                self.city2int[city] = len(self.city2int) + 1
            for j,w in enumerate(weather):
                if w not in self.weather2int[j]:
                    try:
                        self.weather2int[j][w] = int(w)
                    except:
                        self.weather2int[j][w] = len(self.weather2int[j]) + 1
            city = self.city2int[city]
            weather = [self.weather2int[j][w] for j,w in enumerate(weather)]
            if date in self.weather:
                if city not in self.weather[date]: # a day of a city has one kind of weather
                    self.weather[date][city] = weather
            else:
                self.weather[date] = {city: weather}
        print('day with weather: %d' % len(self.weather))
            

    def get_promotions(self):
        self.promotions = {} # {goods_id: [[date_start, date_end, promote_way, discount] ] }
        self.promo2int = {}
        for i in range(1, self.promotions_wb.nrows):
            date_start = dtdt.strptime(self.promotions_wb.cell(i, 1).value.split()[0], '%Y-%m-%d')
            date_end = dtdt.strptime(self.promotions_wb.cell(i, 2).value.split()[0], '%Y-%m-%d')
            goods_id = self.promotions_wb.cell(i, 4).value
            promote_way = self.promotions_wb.cell(i, 5).value
            discount = self.promotions_wb.cell(i, 6).value
            if promote_way not in self.promo2int:
                self.promo2int[promote_way] = len(self.promo2int) + 1
            if discount == '-': # have bought A and pay more money to buy B
                discount = '0'
            promote_way = self.promo2int[promote_way]
            if goods_id in self.promotions:
                self.promotions[goods_id].append([date_start, date_end, promote_way, discount])
            else:
                self.promotions[goods_id] = [[date_start, date_end, promote_way, discount]]
        print('goods with promotions: %d' % len(self.promotions))


    def get_warehouses(self):
        self.warehouses = {} # {store_id: {class_id: warehouse_id} }
        for i in range(self.warehouses_wb.nrows):
            store_id = self.warehouses_wb.cell(i, 0).value
            if store_id == 'NULL': # skip NULL stores
                continue
            class_id = self.warehouses_wb.cell(i, 1).value
            warehouse_id = self.warehouses_wb.cell(i, 2).value
            if store_id in self.warehouses:
                if class_id not in self.warehouses[store_id]: 
                # a class of a store corresponds to only one warehouse 
                    self.warehouses[store_id][class_id] = warehouse_id
            else:
                self.warehouses[store_id] = {class_id: warehouse_id}
        print('stores with class_id and warehouse_id: %d' % len(self.warehouses))


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

def get_data(max_data):
    data = DateProcessing(max_data=max_data)
    print('Preparing data ...')
    start = time.time()
    data.DAT2Matrix()
    end = time.time()
    print('Training data processed, time: {:.2f}s\n'.format(end - start))
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
                  "silent": 1,  
                  "seed": 1301,
                  'lambda': 2, # regularization  
                  }  
        watchlist = [(dtrain, 'train'), (dvalid, 'val')]
        model = xgb.train(params, dtrain, args.num_boost_round, evals=watchlist,
                    early_stopping_rounds=50, feval=rmspe_xg, verbose_eval=True)
        
        print("\nValidating")  
        #ipdb.set_trace()
        yhat = model.predict(xgb.DMatrix(x_valid), ntree_limit=model.best_ntree_limit)

    else:
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_valid, y_valid, reference=lgb_train)
        params = {'task': 'train',
                'application': 'regression_l1',
                'boosting_type': 'gbdt',
                'metric': {'l1', 'mape'},
                'num_leaves': args.num_leaves,
                'learning_rate': args.learning_rate,
                'feature_fraction': 0.9,
                'bagging_fraction': args.subsample,
                'bagging_freq': 5,
                'verbose': 0,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'lambda_l1': 0,
                'lambda_l2': 1
                }
        model = lgb.train(params, lgb_train, args.num_boost_round, valid_sets=lgb_eval,
                    early_stopping_rounds=50, verbose_eval=True)
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
    if args.gs == 1:
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
                param_grid=param_test, n_jobs=-1, verbose=32, error_score='raise', 
                scoring='neg_mean_squared_error')
        gs.fit(data.x_all, data.y_all)
        #ipdb.set_trace()
        for s in gs.grid_scores_:
            print('scores: {}'.format(s))
            args.outfiler.write('scores: {}\n'.format(s))
        print('best params: {}\nbest scores: {}\n'.format(gs.best_params_, gs.best_score_))
        args.outfiler.write('best params: {}\nbest scores: {}\n'.format(gs.best_params_, gs.best_score_))
    elif args.gs == 2:
        xlf = LGBMRegressor(num_leaves=50, max_depth=13, learning_rate=0.1,   
                            n_estimators=100, objective='regression', min_child_weight=1,   
                            subsample=0.8, colsample_bytree=0.8, nthread= -1,  
                            subsample_for_bin=200000, class_weight=None, min_split_gain=0.0, 
                            min_child_samples=20, subsample_freq=1, reg_alpha=0.0, 
                            reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True)  
        xlf.fit(x_train, y_train, eval_metric='l2', verbose=True, 
            eval_set = [(x_valid, y_valid)])#, early_stopping_rounds=100)
        yhat = xlf.predict(x_valid)
        error = rmspe(yhat, y_valid)
        print('rmspe: {}\n'.format(error))
        args.outfiler.write('rmspe: {}\n'.format(error))
    else:
        for ne in range(10, 300, 20):
            xlf = XGBRegressor(max_depth=10, learning_rate=0.05, n_estimators=ne,
                             silent=True, objective='reg:linear', nthread=-1,
                             gamma=0, min_child_weight=1, max_delta_step=0,
                             subsample=0.9, colsample_bytree=1.0, colsample_bylevel=1,
                             reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                             seed=1301, missing=None)
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
    parser.add_argument("--mode", type=str, default='xgb', help='')
    parser.add_argument("--gs", type=int, default=0, help='')
    parser.add_argument("--num_boost_round", type=int, default=500, help='')
    parser.add_argument("--max_depth", type=int, default=10, help='')
    parser.add_argument("--num_leaves", type=int, default=127, help='')
    parser.add_argument("--max_data", type=int, default=10000000, help='')
    parser.add_argument("--learning_rate", type=float, default=0.05, help='')
    parser.add_argument("--subsample", type=float, default=0.9, help='')
    parser.add_argument("--result_dir", type=str, default='xgb_1000w', help='')
    args = parser.parse_args()
    x_train, y_train, x_valid, y_valid = get_data(args.max_data)
    with open('results/%s.txt'%args.result_dir,'w') as args.outfiler:
        for k,v in args.__dict__.items():
            args.outfiler.write('{}: {}\n'.format(k, v))
        args.outfiler.write('\n')
        if args.func == 'main':
            if args.mode == 'lgb':
                for i in range(1):
                    for j in range(5, 9):
                        args.num_leaves = 2**j - 1
                        args.learning_rate = 0.05*(i+1)
                        main(args, x_train, y_train, x_valid, y_valid)
            else:
                for i in range(1):
                    for j in range(10, 55, 5):
                        args.max_depth = j
                        args.learning_rate = 0.05*(i+1)
                        main(args, x_train, y_train, x_valid, y_valid)
        else:
            searching(args, x_train, y_train, x_valid, y_valid)
    end = time.time()
    print('\nTotal time cost: %.2fs\n' % (end - start))
