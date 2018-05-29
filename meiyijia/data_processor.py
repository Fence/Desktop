# coding: utf-8
import os
import gc
import json
import xlrd
import ipdb
import time
import numpy as np
import datetime as dt
from tqdm import tqdm
from datetime import datetime as dtdt

def timeit(f):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()

        print("   [-] %s : %2.5f sec" % (f.__name__, end_time - start_time))
        return result
    return timed

class DateProcessing(object):
    """docstring for DateProcessing"""
    def __init__(self):
        import sys
        reload(sys)
        sys.setdefaultencoding('utf-8')
        workbook = xlrd.open_workbook('basic_data.xlsx')
        #names = workbook.sheet_names()
        #['门店', '商品', '供应商', '天气', '半月档促销商品', '门店仓位']
        self.stores_wb = workbook.sheet_by_name('门店')
        self.items_wb = workbook.sheet_by_name('商品')
        self.weather_wb = workbook.sheet_by_name('天气')
        self.suppliers_wb = workbook.sheet_by_name('供应商')
        self.promotions_wb = workbook.sheet_by_name('半月档促销商品')
        self.warehouses_wb = workbook.sheet_by_name('门店仓位')

        self.n_days = 538 # total days with sales, stock and return data
        self.start_date = dtdt.strptime('2016-11-01','%Y-%m-%d')
        self.dc2int = {'A': 1, 'B': 2, 'C': 3, '': 0} # delivery_cycle to int
        self.stores = {} # {store_id: (city, delivery_cycle) }
        self.store2int = {}
        self.items = {} # {item_id: [labels, ..., supplier_id] }
        self.item2int = {}
        self.suppliers = {} # {supplier_id: {warehouse_id: [data]}}
        self.weather = {} # {date: {city: [weather]} }
        self.weather2int = [{},{},{},{},{},{},{}]
        self.city2int = {}
        self.promotions = {} # {item_id: [[date_start, date_end, promote_way, discount] ] }
        self.promo2int = {}
        self.warehouses = {} # {store_id: {class_id: warehouse_id} }
        self.warehouse_item_transtime = {} # {warehouse_id: {item_id: [orderdays, arrivedays, transtime]}}
        self.warehouse_item = {}
        #self.warehouse_item_sales_volume = json.load(open('data/warehouse_sales_inventory_stock_return.json','r'))

        self.get_stores()
        self.get_items()
        self.get_suppliers()
        self.get_weather()
        self.get_promotions()
        self.get_warehouses()
        self.get_transport_time()


    def is_holidays(self, date):
        holidays = [['2016-12-31', '2017-01-02'], # New Year's Day
                    ['2017-01-27', '2017-02-02'], # Spring Festival
                    ['2017-04-02', '2017-04-04'], # Qingming Festival
                    ['2017-04-29', '2017-05-01'], # May Day
                    ['2017-05-28', '2017-05-30'], # the Dragon Boat Festival
                    ['2017-10-01', '2017-10-08'], # National Day
                    ['2017-12-30', '2018-01-01'],
                    ['2018-02-15', '2018-02-21'],
                    ['2018-04-05', '2018-04-07']
                    ]
        for d in holidays:
            if d[0] <= date <= d[1]:
                return 1
        return 0


    def int2onehot(self, data_dict, value=1):
        new_dict = {'UNK': np.zeros(len(data_dict), dtype=np.int32)}
        for k,v in data_dict.iteritems():
            one_hot = np.zeros(len(data_dict), dtype=np.int32)
            one_hot[v] = value
            new_dict[v] = one_hot
        return new_dict


    def onehot_coding(self, num, dim, value=1):
        output = np.zeros(dim)
        output[num] = value
        return output


    def save_file(self, data, name='save_file'):
        with open('data/%s.json' % name, 'w') as f:
            print('Saving file ...')     
            json.dump(data, f, indent=2)
            print('Successfully save file as data/%s.json\n' % name)    


    def date_transformation(self, date, onehot=False):
        # 1~7 denote Monday~Sunday
        # 1~4 denote Spring~Winter
        [year, month, day] = [int(i) for i in date.split('-')]
        weekday = dt.date.weekday(dt.date(year, month, day)) + 1
        weekend = 1 if weekday >= 6 else 0
        holiday = self.is_holidays(date)
        if 3 <= month <= 5:
            season = 1
        elif 6 <= month <= 8:
            season = 2
        elif 9 <= month <= 11:
            season = 3
        else:
            season = 4

        if onehot:
            date_coding = [holiday] # dim of date coding: 1+1+4+12+31+7=56
            date_coding.append(weekend)
            date_coding.extend(self.onehot_coding(season-1, 4, 2))
            date_coding.extend(self.onehot_coding(month-1, 12, 3))
            date_coding.extend(self.onehot_coding(day-1, 31, 4))
            date_coding.extend(self.onehot_coding(weekday-1, 7, 5)) 
        else:
            date_coding = [holiday, weekend, season, month, day, weekday]

        return date_coding, weekday


    def promotion_coding(self, cur_date, item_id, onehot=False):
        if onehot:
            promotion = np.zeros(9) # way, discount
            promotion.fill(-1)
            if item_id in self.promotions:
                for p in self.promotions[item_id]:
                    # if this item is promoted at this day
                    if p[0] <= cur_date <= p[1]: 
                        promotion[:-1] = self.promo2num[p[-2]]
                        promotion[-1] = p[-1]
                        break
        else:
            promotion = np.zeros(2)
            promotion.fill(-1)
            if item_id in self.promotions:
                for p in self.promotions[item_id]:
                    # if this item is promoted at this day
                    if p[0] <= cur_date <= p[1]:
                        # promotion = p[-2:]
                        promotion[0] = p[-2]
                        promotion[1] = int(p[-1] * 10) # change discount 0.0~1.0 to 1~10
                        break
        return promotion 


    def item_store_city_coding(self, item_id, store_id, onehot=False):
        item_id = item_id if item_id in self.item2int else 'UNK'
        store_id = store_id if store_id in self.store2int else 'UNK'
        if onehot:
            item_coding  = self.item2num[self.item2int[item_id]]
            store_coding = self.store2num[self.store2int[store_id]]
            city_coding  = self.city2num[self.stores[store_id][0]]
        else:
            item_coding  = [self.item2int[item_id]]
            store_coding = [self.store2int[store_id]]
            city_coding  = [self.stores[store_id][0]]
        return item_coding, store_coding, city_coding


    def deliver_city_weather_coding(self, cur_date, store_id, onehot=False):
        tmp_date = cur_date.replace('-','')
        city = self.stores[store_id][0] if store_id in self.stores else -1
        weather = None
        if tmp_date in self.weather:
            if city in self.weather[tmp_date]:
                weather = self.weather[tmp_date][city]
        if onehot:
            deliver_time    = self.dc2num[self.stores[store_id][1]] if store_id in self.stores else [-1]*4
            if weather == None:
                weather_coding = [-1]*7
            else:
                weather_coding  = []
                for i,w in enumerate(weather):
                    weather_coding.extend(self.weather2num[i][w])
        else:
            deliver_time    = [self.stores[store_id][1]] if store_id in self.stores else [-1]
            weather_coding  = weather if weather != None else [-1]*7 
        return deliver_time, weather_coding
        

    def data2matrix_by_store(self, args):
        x_train, x_valid, x_test, y_train, y_valid, y_test  = [], [], [], [], [], []
        count_train = count_test = count_zero = 0

        # ipdb.set_trace()
        print('Loading data ...')
        self.dc2num = self.int2onehot(self.dc2int, 1) # deliver time
        self.city2num = self.int2onehot(self.city2int, 1)
        self.promo2num = self.int2onehot(self.promo2int, 1)
        self.weather2num = [self.int2onehot(d, 1) for d in self.weather2int]
        self.wh_data = json.load(open('data/warehouse_sales_inventory_stock_return.json','r'))
        self.wh_item_store_data = json.load(open('data/datasets.json', 'r'))['wh_item_stores']
        self.item2num = self.int2onehot(self.item2int, 1)
        self.store2num = self.int2onehot(self.store2int, 1)
        try:
            for wh_id in self.wh_data:
                for item_id in self.wh_data[wh_id]:
                    values = self.get_warehouse_item_features(wh_id, item_id)
                    if values[0] > args.valid_days[1] or values[0] < args.valid_days[0]:
                        continue
                    try:
                        self.wh_item_data = json.load(open('data/items_data/%s_%s.json'%(wh_id, item_id),'r'))
                    except:
                        continue
                    
                    self.valid_days, self.orderdays, self.arrivedays, self.transp_time = values
                    for store_id in self.wh_item_data:
                        item, store, city = self.item_store_city_coding(item_id, store_id, args.onehot)
                        for cur_date in self.wh_item_data[store_id]:
                            state, pred_time = self.generate_state_by_store(item, store, city, cur_date, item_id, store_id, args.onehot)
                            target = self.get_target_by_store(cur_date, pred_time, store_id)
                            if target < 0:
                                count_zero += 1
                                continue
                            if cur_date < args.test_start_date:
                                count_train += 1
                                if count_train % args.valid_split == 0:
                                    x_valid.append(state)
                                    y_valid.append(target)
                                else:
                                    x_train.append(state)
                                    y_train.append(target)
                            else:
                                count_test += 1
                                x_test.append(state)
                                y_test.append(target)
                            if count_train % 100000 == 0:
                                print(count_train, count_test)
                            if count_train >= args.max_data:
                                assert count_train < args.max_data
        except AssertionError as e:
            print(e)
        print('train: %d test: %d zeros: %d\n'%(count_train, count_test, count_zero))
        return x_train, y_train, x_valid, y_valid, x_test, y_test


    def generate_state_by_store(self, item, store, city, cur_date, item_id, store_id, onehot):
        date_coding, weekday = self.date_transformation(cur_date, onehot)
        od = sorted(zip(range(len(self.orderdays)), self.orderdays), key=lambda x:x[1]) # sort by weekday
        for i in xrange(len(od)):
            if weekday <= od[i][1]:
                tmp_start = self.arrivedays[od[i][0]] - weekday
                pred_start = tmp_start if tmp_start >= 0 else tmp_start + 7
                tmp_end = self.arrivedays[ (od[i][0] + 1) % len(od) ] - self.arrivedays[od[i][0]]
                pred_end = pred_start + tmp_end if tmp_end >= 0 else pred_start + tmp_end + 7
                # assert pred_end >= pred_start
                break
        pred_time = [pred_start, pred_end]
        
        
        # get store-specific features
        deliver_time, weather = self.deliver_city_weather_coding(cur_date, store_id, onehot)
        promotion = self.promotion_coding(cur_date, item_id, onehot)
        x_i_j = self.wh_item_data[store_id][cur_date]
        state = []
        state.extend(item)                  # dim = 688
        state.extend(store)                 # dim = 15992
        state.extend(city)                  # dim = 21
        state.extend(weather)               # dim = 7*(n1~n7)
        state.append(self.valid_days)       # dim = 1
        state.append(self.transp_time)      # dim = 1
        state.extend(x_i_j)                 # dim = 4
        state.extend(deliver_time)          # dim = 4
        state.extend(promotion)             # dim = 9
        state.extend(date_coding)           # dim = 56
        
        return np.array(state), pred_time


    def get_target_by_store(self, cur_date, pred_time, store_id):
        tmp_date = self.update_date(cur_date, pred_time[0])
        target_sales = target_stocks = target_returns = 0
        valid_count = 0
        for d in xrange(pred_time[0], pred_time[1]):
            try:
                sales, _, stocks, returns = self.wh_item_data[store_id][tmp_date]
                target_sales += sales
                target_stocks += stocks
                target_returns += returns
                valid_count += 1
            except Exception as e: # tmp_date may not be in the keys
                continue
            tmp_date = self.update_date(tmp_date)

        if valid_count == 0:
            return -1
        target_order = target_sales # target_stocks - target_returns # 
        return target_order


    def data2matrix_by_warehouse(self, args):
        x_train, x_valid, x_test, y_train, y_valid, y_test  = [], [], [], [], [], []
        count_train = count_test = count_zero = 0

        #ipdb.set_trace()
        print('Loading data ...')
        self.dc2num = self.int2onehot(self.dc2int, 1) # deliver time
        self.city2num = self.int2onehot(self.city2int, 1)
        self.promo2num = self.int2onehot(self.promo2int, 1)
        self.wh_data = json.load(open('data/warehouse_sales_inventory_stock_return.json','r'))
        self.wh_item_store_data = json.load(open('data/datasets.json', 'r'))['wh_item_stores']
        self.item2num = self.int2onehot(self.item2int, 1)
        #self.valid_days_items = {}
        for wh_id in self.wh_data:
            for item_id in self.wh_data[wh_id]:
                values = self.get_warehouse_item_features(wh_id, item_id)
                # if values[0] not in self.valid_days_items:
                #     self.valid_days_items[values[0]] = 1
                # else:
                #     self.valid_days_items[values[0]] += 1
                # continue
                if values[0] > args.valid_days[1] or values[0] < args.valid_days[0]:
                    continue
                self.valid_days, self.orderdays, self.arrivedays, self.transp_time = values
                for cur_date in self.wh_data[wh_id][item_id]:
                    state, pred_time = self.generate_state_by_warehouse(cur_date, wh_id, item_id, args.onehot)
                    target = self.get_target_by_warehouse(cur_date, pred_time, wh_id, item_id)
                    if target < 0:
                        count_zero += 1
                        continue
                    if cur_date < args.test_start_date:
                        count_train += 1
                        if count_train % args.valid_split == 0:
                            x_valid.append(state)
                            y_valid.append(target)
                        else:
                            x_train.append(state)
                            y_train.append(target)
                    else:
                        count_test += 1
                        x_test.append(state)
                        y_test.append(target)

        print('train: %d test: %d zeros: %d\n'%(count_train, count_test, count_zero))
        return x_train, y_train, x_valid, y_valid, x_test, y_test


    def get_warehouse_item_features(self, wh_id, item_id):
        # get warehouse-item features
        valid_days = float(self.items[item_id][-2])
        orderdays, arrivedays, transp_time = self.warehouse_item_transtime[wh_id][item_id]

        next_week_order_days = [d+7 for d in orderdays]
        next_week_arrive_days = [d+7 for d in arrivedays]
        orderdays.extend(next_week_order_days)
        arrivedays.extend(next_week_arrive_days)
        return valid_days, orderdays, arrivedays, transp_time


    def generate_state_by_warehouse(self, cur_date, wh_id, item_id, onehot):
        date_coding, weekday = self.date_transformation(cur_date)
        od = sorted(zip(range(len(self.orderdays)), self.orderdays), key=lambda x:x[1]) # sort by weekday

        for i in xrange(len(od)):
            if weekday <= od[i][1]:
                tmp_start = self.arrivedays[od[i][0]] - weekday
                pred_start = tmp_start if tmp_start >= 0 else tmp_start + 7
                tmp_end = self.arrivedays[ (od[i][0] + 1) % len(od) ] - self.arrivedays[od[i][0]]
                pred_end = pred_start + tmp_end if tmp_end >= 0 else pred_start + tmp_end + 7
                # assert pred_end >= pred_start
                break
        pred_time = [pred_start, pred_end]
        # get item-specific features
        promotion = self.promotion_coding(cur_date, item_id, onehot)
        n_stores = len(self.wh_item_store_data[wh_id][item_id])
        x_i_j = self.wh_data[wh_id][item_id][cur_date]
        state = []
        state.append(n_stores)              # dim = 1
        state.append(self.valid_days)       # dim = 1
        state.append(self.transp_time)      # dim = 1
        state.extend(x_i_j)                 # dim = 4
        state.extend(promotion)             # dim = 9 or 2
        state.extend(date_coding)           # dim = 56 or 5
        state.extend(self.item2num[item_id])
        return np.array(state), pred_time


    def get_target_by_warehouse(self, cur_date, pred_time, wh_id, item_id):
        tmp_date = self.update_date(cur_date, pred_time[0])
        target_sales = target_stocks = target_returns = 0
        valid_count = 0
        for d in xrange(pred_time[0], pred_time[1]):
            try:
                sales, _, stocks, returns = self.wh_data[wh_id][item_id][tmp_date]
                target_sales += sales
                target_stocks += stocks
                target_returns += returns
                valid_count += 1
            except Exception as e: # tmp_date may not be in the keys
                #ipdb.set_trace()
                continue
            tmp_date = self.update_date(tmp_date)

        if valid_count == 0:
            return -1

        target_order = target_sales # target_stocks - target_returns # 
        return target_order


    def update_date(self, cur_date, days=1):
        date = dtdt.strptime(cur_date, '%Y-%m-%d')
        new_date = date + dt.timedelta(days=days)
        return str(new_date).split()[0]
 

    @timeit
    def compute_inventory(self):
        count = 0 # used for displaying the process
        sales_data = {} # {warehouse: {item: {store: {date: [sale, inventory, stock, return]}}}}
        in_out_data = {} # {warehouse: {item: {store: {date: [stock, return]}}}}
        dates = [str(self.start_date+dt.timedelta(days=i)).split()[0] for i in xrange(self.n_days)]
        # read the stock_return data
        for line in open('data/days_stock_and_return_volume.DAT'):
            count += 1
            if count %1000000 == 0:
                print(count)
            date, store_id, item_id, stock_volume, return_volume = line.split('\t')
            try:
                class_id = self.items[item_id][0].strip()
                supplier_id = self.items[item_id][-1]
                goods2warehouses = list(self.suppliers[supplier_id].keys())
                warehouse_id = self.warehouses[store_id][class_id]
            except Exception as e:
                continue
            if warehouse_id not in goods2warehouses:
                continue 
            stock_volume, return_volume = float(stock_volume), float(return_volume)
            if warehouse_id in in_out_data:
                if item_id in in_out_data[warehouse_id]:
                    if store_id in in_out_data[warehouse_id][item_id]:
                        if date not in in_out_data[warehouse_id][item_id][store_id]:
                            in_out_data[warehouse_id][item_id][store_id][date] = [stock_volume, return_volume]
                        else:
                            in_out_data[warehouse_id][item_id][store_id][date][0] += stock_volume
                            in_out_data[warehouse_id][item_id][store_id][date][1] += return_volume
                    else:
                        in_out_data[warehouse_id][item_id][store_id] = {date: [stock_volume, return_volume]}
                else:
                    in_out_data[warehouse_id][item_id] = {store_id: {date: [stock_volume, return_volume]}}
            else:
                in_out_data[warehouse_id] = {item_id: {store_id: {date: [stock_volume, return_volume]}}}

        # read the sales_volume data
        count = 0
        for line in open('data/days_sales_volume.DAT'):
            count += 1
            if count %1000000 == 0:
                print(count)
            date, store_id, item_id, sales_volume = line.split('\t')
            try:
                class_id = self.items[item_id][0].strip()
                supplier_id = self.items[item_id][-1]
                goods2warehouses = list(self.suppliers[supplier_id].keys())
                warehouse_id = self.warehouses[store_id][class_id]
                delivery_cycle = self.stores[store_id][1]
            except Exception as e:
                continue
            if warehouse_id not in goods2warehouses:
                continue
            sales_volume = float(sales_volume)
            if warehouse_id in sales_data:
                if item_id in sales_data[warehouse_id]:
                    if store_id in sales_data[warehouse_id][item_id]:
                        if date not in sales_data[warehouse_id][item_id][store_id]:
                            sales_data[warehouse_id][item_id][store_id][date] = [sales_volume]
                        else:
                            sales_data[warehouse_id][item_id][store_id][date][0] += sales_volume
                    else:
                        sales_data[warehouse_id][item_id][store_id] = {date: [sales_volume]}
                else:
                    sales_data[warehouse_id][item_id] = {store_id: {date: [sales_volume]}}
            else:
                sales_data[warehouse_id] = {item_id: {store_id: {date: [sales_volume]}}}

            try:
                stock_volume, return_volume = in_out_data[warehouse_id][item_id][store_id][date]
                inventory = stock_volume - return_volume - sales_volume
            except Exception as e:
                inventory = stock_volume = return_volume = 0
            sales_data[warehouse_id][item_id][store_id][date].extend([inventory, stock_volume, return_volume])
        
        # combine sales_volume data and stock_return data by dates
        count = count_error = 0
        for warehouse_id in sales_data:
            for item_id in sales_data[warehouse_id]:
                for store_id in sales_data[warehouse_id][item_id]:
                    try:
                        v = sales_data[warehouse_id][item_id][store_id]
                        v = in_out_data[warehouse_id][item_id][store_id]
                    except:
                        count_error += 1
                        continue
                    for date in dates:
                        if date in sales_data[warehouse_id][item_id][store_id] or date in in_out_data[warehouse_id][item_id][store_id]:
                            try:
                                sales_volume = sales_data[warehouse_id][item_id][store_id][date][0]
                            except:
                                sales_volume = 0
                            try:
                                stock_volume, return_volume = in_out_data[warehouse_id][item_id][store_id][date]
                                inventory = stock_volume - return_volume - sales_volume
                            except:
                                inventory = stock_volume = return_volume = 0
                            sales_data[warehouse_id][item_id][store_id][date] = [sales_volume, inventory, stock_volume, return_volume]
                            count += 1
                            if count %1000000 == 0:
                                print(count)
                self.save_file(sales_data[warehouse_id][item_id], 'items_data/%s_%s'%(warehouse_id, item_id))
        print('\ncount_error: %d\n' % count_error)
        return sales_data

    @timeit
    def compute_inventory_by_warehouse(self):
        sales_data = {}
        in_out_data = {}
        for line in open('data/warehouses_stock_return.DAT').readlines():
            date, warehouse_id, item_id, stock_volume, return_volume = line.split('\t')
            stock_volume, return_volume = float(stock_volume), float(return_volume)
            if warehouse_id in in_out_data:
                if item_id in in_out_data[warehouse_id]:
                    if date not in in_out_data[warehouse_id][item_id]:
                        in_out_data[warehouse_id][item_id][date] = [stock_volume, return_volume]
                else:
                    in_out_data[warehouse_id][item_id] = {date: [stock_volume, return_volume]}
            else:
                in_out_data[warehouse_id] = {item_id: {date: [stock_volume, return_volume]}}

        for line in open('data/warehouses_sales_volume.DAT').readlines():
            date, warehouse_id, item_id, sales_volume = line.strip().split('\t')
            sales_volume = float(sales_volume)
            if warehouse_id in sales_data:
                if item_id in sales_data[warehouse_id]:
                    if date not in sales_data[warehouse_id][item_id]:
                        sales_data[warehouse_id][item_id][date] = [sales_volume]
                else:
                    sales_data[warehouse_id][item_id] = {date: [sales_volume]}
            else:
                sales_data[warehouse_id] = {item_id: {date: [sales_volume]}}
            try:
                stock_volume, return_volume = in_out_data[warehouse_id][item_id][date]
                inventory = stock_volume - return_volume - sales_volume
            except Exception as e:
                inventory = stock_volume = return_volume = 0
            sales_data[warehouse_id][item_id][date].extend([inventory, stock_volume, return_volume])


        dates = [str(self.start_date+dt.timedelta(days=i)).split()[0] for i in xrange(self.n_days)]
        count = count_error = 0
        for warehouse_id in sales_data:
            for item_id in sales_data[warehouse_id]:
                try:
                    v = sales_data[warehouse_id][item_id]
                    v = in_out_data[warehouse_id][item_id]
                except:
                    count_error += 1
                    continue
                for date in dates:
                    if date in sales_data[warehouse_id][item_id] or date in in_out_data[warehouse_id][item_id]:
                        try:
                            sales_volume = sales_data[warehouse_id][item_id][date][0]
                        except:
                            sales_volume = 0
                        try:
                            stock_volume, return_volume = in_out_data[warehouse_id][item_id][date]
                            inventory = stock_volume - return_volume - sales_volume
                        except:
                            inventory = stock_volume = return_volume = 0
                        sales_data[warehouse_id][item_id][date] = [sales_volume, inventory, stock_volume, return_volume]
                        count += 1
                        if count %1000000 == 0:
                            print(count)
        print('\ncount_error: %d\n' % count_error)
        self.save_file(sales_data, 'warehouse_sales_inventory_stock_return')
        return sales_data

    @timeit
    def get_sales_volume_by_warehouse(self):
        data = {}
        dates = []
        count = count_error = 0
        for line in open('data/days_sales_volume.DAT'):
            count += 1
            if count %1000000 == 0:
                print(count)
            date, store_id, item_id, sales_volume = line.split('\t')
            try:
                class_id = self.items[item_id][0].strip()
                supplier_id = self.items[item_id][-1]
                goods2warehouses = list(self.suppliers[supplier_id].keys())
                warehouse_id = self.warehouses[store_id][class_id]
            except Exception as e:
                count_error += 1
                continue
            if warehouse_id not in goods2warehouses:
                continue
            if date in data:
                if (warehouse_id, item_id) in data[date]:
                    data[date][(warehouse_id, item_id)] += float(sales_volume)
                else:
                    data[date][(warehouse_id, item_id)] = float(sales_volume)
            else:
                dates.append(dtdt.strptime(date, '%Y-%m-%d'))
                data[date] = {(warehouse_id, item_id): float(sales_volume)}
        print('days of sales volume: {}\nwarehouse_goods pairs: {}\ncount_error: {}'.format(
            len(data), len(data[date]), count_error))
        with open('data/warehouses_sales_volume.DAT','w') as f:
            print('Saving file ...')
            for date in sorted(dates):
                date = str(date).split()[0]
                for (warehouse_id, item_id), sales_volume in data[date].iteritems():
                    f.write('{}\t{}\t{}\t{}\n'.format(date, warehouse_id, item_id, sales_volume))
            print('Successfully save warehouses_sales_volume.DAT\n')

    @timeit
    def get_stock_return_by_warehouse(self):
        data = {}
        dates = []
        count = count_error = 0
        for line in open('data/days_stock_and_return_volume.DAT'):
            count += 1
            if count %1000000 == 0:
                print(count)
            date, store_id, item_id, stock_volume, return_volume = line.split('\t')
            try:
                class_id = self.items[item_id][0].strip()
                supplier_id = self.items[item_id][-1]
                goods2warehouses = list(self.suppliers[supplier_id].keys())
                warehouse_id = self.warehouses[store_id][class_id]
            except Exception as e:
                count_error += 1
                continue
            if warehouse_id not in goods2warehouses:
                continue
            if date in data:
                if (warehouse_id, item_id) in data[date]:
                    data[date][(warehouse_id, item_id)][0] += float(stock_volume)
                    data[date][(warehouse_id, item_id)][1] += float(return_volume)
                else:
                    data[date][(warehouse_id, item_id)] = [float(stock_volume), float(return_volume)]
            else:
                dates.append(dtdt.strptime(date, '%Y-%m-%d'))
                data[date] = {(warehouse_id, item_id): [float(stock_volume), float(return_volume)]}
        print('days of stock and return: {}\nwarehouse_goods pairs: {}\ncount_error: {}'.format(
            len(data), len(data[date]), count_error))
        with open('data/warehouses_stock_return.DAT','w') as f:
            print('Saving file ...')
            for date in sorted(dates):
                date = str(date).split()[0]
                for (warehouse_id, item_id), (stock_volume, return_volume) in data[date].iteritems():
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(date, warehouse_id, item_id, stock_volume, return_volume))
            print('Successfully save warehouses_stock_return.DAT\n')

    @timeit
    def get_sales_volume_by_day(self):
        data = {}
        dates = []
        count = 0
        file_names = ['门店商品销售流水数据_1', '门店商品销售流水数据_2', '门店商品销售流水数据_2017', '门店商品销售流水数据_2018']
        for name in file_names:
            print('\ncurrent file: %s\n' % name)
            for line in open('data/%s.DAT' % name):
                count += 1
                if count %1000000 == 0:
                    print(count)
                items = line.split()
                date = items[0]
                _, store_id, item_id, sales_volume = items[1].split('|')
                if date in data:
                    if (store_id, item_id) in data[date]:
                        data[date][(store_id, item_id)] += float(sales_volume)
                    else:
                        data[date][(store_id, item_id)] = float(sales_volume)
                else:
                    dates.append(dtdt.strptime(date, '%Y-%m-%d'))
                    data[date] = {(store_id, item_id): float(sales_volume)}

        print('days of sales volume: %d' % len(data))
        with open('data/days_sales_volume.DAT','w') as f:
            print('Saving file ...')     
            for date in sorted(dates):
                date = str(date).split()[0]
                for (store_id, item_id), sales_volume in data[date].iteritems():
                    f.write('{}\t{}\t{}\t{}\n'.format(date, store_id, item_id, sales_volume))
            print('Successfully save days_sales_volume.DAT\n')

    @timeit
    def get_stock_and_return_by_day(self):
        data = {}
        dates = []
        count = 0
        file_names = ['门店商品进退货数据1', '门店商品进退货数据2', '门店商品进退货数据']
        for name in file_names:
            print('\ncurrent file: %s\n' % name)
            for line in open('data/%s.DAT' % name):
                count += 1
                if count %1000000 == 0:
                    print(count)
                items = line.split()
                date = items[0]
                _, store_id, item_id, stock_volume, return_volume = items[1].split('|')
                if date in data:
                    if (store_id, item_id) in data[date]:
                        data[date][(store_id, item_id)][0] += float(stock_volume)
                        data[date][(store_id, item_id)][1] += float(return_volume)
                    else:
                        data[date][(store_id, item_id)] = [float(stock_volume), float(return_volume)]
                else:
                    dates.append(dtdt.strptime(date, '%Y-%m-%d'))
                    data[date] = {(store_id, item_id): [float(stock_volume), float(return_volume)]}
        print('days of stock and return volume: %d' % len(data))
        with open('data/days_stock_and_return_volume.DAT','w') as f:
            print('Saving file ...')     
            for date in sorted(dates):
                date = str(date).split()[0]
                for (store_id, item_id),[stock_volume, return_volume] in data[date].iteritems():
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(date, store_id, item_id, stock_volume, return_volume))
            print('Successfully save days_stock_and_return_volume.DAT\n')


    def get_stores(self):
        for i in xrange(1, self.stores_wb.nrows):
            store_id = self.stores_wb.cell(i, 0).value.strip()
            city = self.stores_wb.cell(i, 1).value.strip()
            dc = self.stores_wb.cell(i, 2).value.strip()
            if city not in self.city2int:
                self.city2int[city] = len(self.city2int)
            if store_id not in self.stores:
                self.stores[store_id] = [self.city2int[city], self.dc2int[dc]]
                # self.stores[store_id] = self.stores_wb.row_values(i, 1, 3)
        self.store2int = {k:len(self.store2int) for k in self.stores}
        print('stores: %d\ncities: %d' % (len(self.stores), len(self.city2int)))


    def get_items(self):
        for i in xrange(1, self.items_wb.nrows):
            item_id = self.items_wb.cell(i, 0).value
            supplier_id = self.items_wb.cell(i, self.items_wb.ncols - 1).value
            if item_id not in self.items:
                self.items[item_id] = [s.strip() for s in self.items_wb.row_values(i, 1, None)[0::2]]
                self.items[item_id].append(supplier_id)
        self.item2int = {k:len(self.item2int) for k in self.items}
        print('items: %d' % len(self.items))


    def get_suppliers(self):
        for i in xrange(1, self.suppliers_wb.nrows):
            warehouse_id = str(int(float(self.suppliers_wb.cell(i, 0).value)))
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
        for i in xrange(1, self.weather_wb.nrows):
            date = self.weather_wb.cell(i, 0).value
            city = self.weather_wb.cell(i, 1).value
            weather = self.weather_wb.row_values(i, 2, None)
            if city not in self.city2int:
                self.city2int[city] = len(self.city2int)
            for j,w in enumerate(weather):
                if w not in self.weather2int[j]:
                    # try:
                    #     self.weather2int[j][w] = int(w)
                    # except:
                    self.weather2int[j][w] = len(self.weather2int[j])
            city = self.city2int[city]
            weather = [self.weather2int[j][w] for j,w in enumerate(weather)]
            if date in self.weather:
                if city not in self.weather[date]: # a day of a city has one kind of weather
                    self.weather[date][city] = weather
            else:
                self.weather[date] = {city: weather}
        print('type of weather: %d\nday with weather: %d' % (len(self.weather2int), len(self.weather)))
            

    def get_promotions(self):
        for i in xrange(1, self.promotions_wb.nrows):
            #date_start = dtdt.strptime(self.promotions_wb.cell(i, 1).value.split()[0], '%Y-%m-%d')
            #date_end = dtdt.strptime(self.promotions_wb.cell(i, 2).value.split()[0], '%Y-%m-%d')
            date_start = self.promotions_wb.cell(i, 1).value.split()[0]
            date_end = self.promotions_wb.cell(i, 2).value.split()[0]
            item_id = self.promotions_wb.cell(i, 4).value
            promote_way = self.promotions_wb.cell(i, 5).value
            discount = self.promotions_wb.cell(i, 6).value
            if promote_way not in self.promo2int:
                self.promo2int[promote_way] = len(self.promo2int)
            if discount == '-': # have bought A and pay more money to buy B
                discount = '0'
            promote_way = self.promo2int[promote_way]
            if item_id in self.promotions:
                self.promotions[item_id].append([date_start, date_end, promote_way, discount])
            else:
                self.promotions[item_id] = [[date_start, date_end, promote_way, discount]]
        print('total promotion ways: %d\nitems with promotions: %d' % (len(self.promo2int), len(self.promotions)))


    def get_warehouses(self):
        for i in xrange(1, self.warehouses_wb.nrows):
            store_id = self.warehouses_wb.cell(i, 0).value
            if store_id == 'NULL': # skip NULL stores
                continue
            class_id = self.warehouses_wb.cell(i, 1).value
            warehouse_id = str(int(float(self.warehouses_wb.cell(i, 2).value)))
            if store_id in self.warehouses:
                if class_id not in self.warehouses[store_id]: 
                # a class of a store corresponds to only one warehouse 
                    self.warehouses[store_id][class_id] = warehouse_id
            else:
                self.warehouses[store_id] = {class_id: warehouse_id}
        print('stores with class_id and warehouse_id: %d' % len(self.warehouses))


    def get_transport_time(self):
        for item_id in self.items:
            supplier_id = self.items[item_id][-1]
            supplier_id = self.items[item_id][-1]
            for warehouse_id, data in self.suppliers[supplier_id].iteritems():
                arrivedays = [int(d) for d in data[0][0].split('/')[:-1]]
                orderdays = [int(d[2]) for d in data]
                transtime = int(data[0][-1])
                if warehouse_id in self.warehouse_item_transtime:
                    if item_id not in self.warehouse_item_transtime[warehouse_id]:
                        self.warehouse_item[warehouse_id].append(item_id)
                        # transport time format = [orderdays, arrivedays, transtime]
                        self.warehouse_item_transtime[warehouse_id][item_id] = [orderdays, arrivedays, transtime] 
                else:
                    self.warehouse_item[warehouse_id] = [item_id]
                    self.warehouse_item_transtime[warehouse_id] = {item_id: [orderdays, arrivedays, transtime]}
        if not os.path.exists('data/warehouse_item_dict.json'):
            self.save_file(self.warehouse_item, 'warehouse_item_dict')


if __name__ == '__main__':
    data = DateProcessing()
    # data.get_sales_volume_by_day()
    # data.get_stock_and_return_by_day()
    # data.get_sales_volume_by_warehouse()
    # data.get_stock_return_by_warehouse()
    # data.compute_inventory_by_warehouse()
    # data.compute_inventory()
    print('\nAll done!\n')