import re
import time
import ipdb
import xlrd
import numpy as np
import datetime as dt
import xgboost as xgb
from datetime import datetime as dtdt
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

class DateProcessing(object):
    """docstring for DateProcessing"""
    def __init__(self, max_data):
        self.max_data = max_data
        workbook = xlrd.open_workbook('商品门店供应商天气半月档基础数据.xlsx')
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


#定义一些变换和评判准则  
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
    y = np.exp(y) - 1  
    yhat = np.exp(yhat) - 1  
    w = ToWeight(y)  
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))  
    return "rmspe", rmspe 


def main():
    data = DateProcessing(max_data=1000000)
    print('Preparing data ...')
    data.DAT2Matrix()
    print('Training data processed\n')
    test_num = int(len(data.x_all) * 0.2)  
    if test_num > 100000:
        test_num = 100000
    num = len(data.x_all) - test_num
    x_train = data.x_all[: num]
    x_valid = data.x_all[num: ]
    y_train = data.y_all[: num]
    y_valid = data.y_all[num: ]
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    #ipdb.set_trace()

    params = {"objective": "reg:linear", # linear regression 
              "booster": "gbtree", # gradient boosting tree
              "eta": 0.05,  # learning rate
              "max_depth": 10,  # max depth of gbtree
              "subsample": 0.9,  # subsampling rate
              "colsample_bytree": 1.0,  #
              "silent": 1,  
              "seed": 1301,
              'lambda': 2, # regularization  
              }  
    num_boost_round = 500
    watchlist = [(dtrain, 'train'), (dvalid, 'val')]
    model = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                early_stopping_rounds=100, verbose_eval=True)
    
    print("\nValidating")  
    #ipdb.set_trace()
    yhat = model.predict(xgb.DMatrix(x_valid))  
    error = rmspe(yhat, y_valid)  
    print('RMSPE: {:.6f}\n'.format(error))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('\nTotal time cost: %.2fs\n' % (end - start))