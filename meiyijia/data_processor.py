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
        self.items = {} # {item_id: [labels, ..., supplier_id] }
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


    def int2onehot(self, data_dict, value=1):
        new_dict = {}
        for k,v in data_dict.iteritems():
            one_hot = np.zeros(len(data_dict), dtype=np.int32)
            one_hot[v] = value
            new_dict[v] = one_hot
        return new_dict


    def onehot(self, num, dim, value=1):
        output = np.zeros(dim)
        output[num] = value
        return output


    @timeit
    def compute_inventory(self):
        count = 0
        sales_data = {} # {warehouse: {item: {store: {date: [sale, inventory, stock, return]}}}}
        in_out_data = {} # {warehouse: {item: {store: {date: [stock, return]}}}}
        dates = [str(self.start_date+dt.timedelta(days=i)).split()[0] for i in xrange(self.n_days)]
        #ipdb.set_trace()
        for line in open('data/days_stock_and_return_volume.DAT'):
            count += 1
            # if count > 50000:
            #     break
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
            #date = dtdt.strptime(date, '%Y-%m-%d')
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

        count = 0
        for line in open('data/days_sales_volume.DAT'):
            count += 1
            # if count > 50000:
            #     break
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
            #date = dtdt.strptime(date, '%Y-%m-%d')
            sales_volume = float(sales_volume)
            # order_date = date + dt.timedelta(days=1) if delivery_cycle in ['A', 'B'] else date
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

            # city = self.city2int[self.stores[store_id][0]]
            # weather = self.weather[date.replace('-','')][city]
            try:
                stock_volume, return_volume = in_out_data[warehouse_id][item_id][store_id][date]
                inventory = stock_volume - return_volume - sales_volume
            except Exception as e:
                inventory = stock_volume = return_volume = 0
            sales_data[warehouse_id][item_id][store_id][date].extend([inventory, stock_volume, return_volume])
        
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


    def save_file(self, data, name='save_file'):
        with open('data/%s.json' % name, 'w') as f:
            print('Saving file ...')     
            json.dump(data, f, indent=2)
            print('Successfully save file as data/%s.json\n' % name)                  


    def data2matrix(self, args):
        x_train = [] 
        x_valid = [] 
        x_test  = []
        y_train = [] 
        y_valid = [] 
        y_test  = []
        count_train = count_test = 0

        #ipdb.set_trace()
        self.dc2num = self.int2onehot(self.dc2int, 1) # deliver time
        self.city2num = self.int2onehot(self.city2int, 1)
        self.promo2num = self.int2onehot(self.promo2int, 1)
        self.wh_data = json.load(open('data/warehouse_sales_inventory_stock_return.json','r'))
        for wh_id in self.wh_data:
            for item_id in self.wh_data[wh_id]:
                values = self.get_warehouse_item_features(wh_id, item_id)
                if len(values) == 1:
                    continue
                else:
                    self.valid_days, self.orderdays, self.arrivedays, self.transp_time = values
                    for cur_date in self.wh_data[wh_id][item_id]:
                        state, pred_time = self.generate_state(cur_date, self.wh_data[wh_id][item_id], item_id)
                        target = self.get_target(cur_date, pred_time, self.wh_data[wh_id][item_id])
                        if cur_date < args.test_start_date:
                            count_train += 1
                            if count_train % 10 == 0:
                                x_valid.append(state)
                                y_valid.append(target)
                            else:
                                x_train.append(state)
                                y_train.append(target)
                        else:
                            count_test += 1
                            x_test.append(state)
                            y_test.append(target)
                        # if count_train % 100000 == 0:
                        #     print(count_train, count_test)
        return x_train, y_train, x_valid, y_valid, x_test, y_test



    def get_warehouse_item_features(self, wh_id, item_id):
        # get warehouse-item features
        # try:
        valid_days = float(self.items[item_id][-2])
        orderdays, arrivedays, transp_time = self.warehouse_item_transtime[wh_id][item_id]
        # except Exception as e:
        #     # ipdb.set_trace()
        #     return [-1]
        # if len(self.wh_item_data) < self.min_stores: # no more than 100 stores saling this item
        #     if is_train:
        #         print('wh {} item {} < {} stores\n'.format(self.warehouse_id, self.item_id, self.min_stores))
        #     return [-1]
        next_week_order_days = [d+7 for d in orderdays]
        next_week_arrive_days = [d+7 for d in arrivedays]
        orderdays.extend(next_week_order_days)
        arrivedays.extend(next_week_arrive_days)
        return valid_days, orderdays, arrivedays, transp_time


    def generate_state(self, cur_date, wh_item_data, item_id):
        """ all features of an item_warehouse state: 73 (no city and no weather)
        sales: 1,  inventory: 1, stock: 1, return: 1,  deliver_time: 4, valid_days: 1, transp_time: 1,
        season: 4, month: 12,    day: 31,  weekday: 7, promotion: 8+1,  weather: 7     city: 21
        """
        valid_count = 0
        date_emb, weekday = self.date_transformation(cur_date)
        od = sorted(zip(range(len(self.orderdays)), self.orderdays), key=lambda x:x[1]) # sort by weekday
        #ipdb.set_trace()
        for i in xrange(len(od)):
            if weekday <= od[i][1]:
                tmp_start = self.arrivedays[od[i][0]] - weekday
                pred_start = tmp_start if tmp_start >= 0 else tmp_start + 7
                tmp_end = self.arrivedays[ (od[i][0] + 1) % len(od) ] - self.arrivedays[od[i][0]]
                pred_end = pred_start + tmp_end if tmp_end >= 0 else pred_start + tmp_end + 7
                # assert pred_end >= pred_start
                break
        pred_time = [pred_start, pred_end]
        promotion = np.zeros(9) # way, discount
        if item_id in self.promotions:
            for p in self.promotions[item_id]:
                # if this item is promoted at this day
                if p[0] <= cur_date <= p[1]: 
                    promotion[:-1] = self.promo2num[p[-2]]
                    promotion[-1] = p[-1]
                    break
        # get store-specific features
        #count_less_than_dates = 0
        state, x_i_j, deliver_time = [], np.zeros(4), np.zeros(4)
        # for store_id in self.wh_item_data:
        #     if cur_date in self.wh_item_data[store_id]:
        #         x_i_j += np.array(self.wh_item_data[store_id][cur_date])
        #         valid_count += 1
        #     # pad zero sales, inventory, stock and return values for missing dates
        #     elif self.use_padding:
        #         x_i_j += [0, 0, 0, 0]
        #         valid_count += 1
        #     else:
        #         pass
        #         #continue
        #     if self.stores[store_id][1] in self.dc2num:
        #         deliver_time += self.dc2num[self.stores[store_id][1]]
        x_i_j = wh_item_data[cur_date]
        tmp_count, tmp_day, tmp_trans = np.zeros(1), np.zeros(1), np.zeros(1)
        #tmp_count.fill(valid_count)
        tmp_day.fill(self.valid_days)
        tmp_trans.fill(self.transp_time)
        #state.extend(tmp_count)    # dim = 1
        state.extend(tmp_day)      # dim = 1
        state.extend(tmp_trans)    # dim = 1
        state.extend(x_i_j)        # dim = 4
        #state.extend(deliver_time) # dim = 4
        state.extend(date_emb)     # dim = 4 + 12 + 31 + 7 = 54
        state.extend(promotion)    # dim = 8 + 1 = 9
        #print(self.item_id, self.warehouse_id, deliver_time, x_i_j, valid_count)
        return np.array(state), pred_time#, valid_count


    def get_target(self, cur_date, pred_time, wh_item_data):
        tmp_date = self.update_date(cur_date, pred_time[0])
        target_sales = target_stocks = target_returns = 0
        #ipdb.set_trace()
        for d in xrange(pred_time[0], pred_time[1]):
            try:
                sales, _, stocks, returns = wh_item_data[tmp_date]
                target_sales += sales
                target_stocks += stocks
                target_returns += returns
            except Exception as e: # tmp_date may not be in the keys
                #ipdb.set_trace()
                continue
            tmp_date = self.update_date(tmp_date)

        target_order = target_stocks - target_returns # target_sales # 
        return target_order


    def update_date(self, cur_date, days=1):
        date = dtdt.strptime(cur_date, '%Y-%m-%d')
        new_date = date + dt.timedelta(days=days)
        return str(new_date).split()[0]


    def DAT2Matrix(self, args):
        x_all = []
        y_all = []
        count_error = 0
        # start from 2017-04-17 so each record has at least a week history 
        history_volume = {}
        #ipdb.set_trace()
        for line in open('data/days_sales_volume.DAT'):
            if args.max_data != -1 and len(y_all) >= args.max_data:
                break
            # if len(y_all) % 1000000 == 0:
            #     print(len(y_all))
            try:
                date, store_id, item_id, sales_volume = line.split('\t')
                sales_volume = int(float(sales_volume))
                if (store_id, item_id) not in history_volume:
                    history_volume[(store_id, item_id)] = [sales_volume]
                    continue
                else:
                    history_volume[(store_id, item_id)].append(sales_volume)
                    count = len(history_volume[(store_id, item_id)])
                    if count < 8:
                        continue
                year, month, day = date.split('-')
                tmp_x = list(self.date_transformation(year, month, day))
                city = self.city2int[self.stores[store_id][0]]
                weather = self.weather[date.replace('-','')][city]
                #date = dtdt.strptime(date, '%Y-%m-%d')
                promotion = [0]*8
                if item_id in self.promotions:
                    for p in self.promotions[item_id]:
                        # if this item is promoted at this day
                        if p[0] <= date <= p[1]: 
                            promotion = [p[0].year, p[0].month, p[0].day] 
                            promotion = promotion + [p[0].year, p[1].month, p[1].day] + p[2:]
                            break
                tmp_x.extend([int(store_id), int(item_id), city])
                tmp_x.extend(weather)
                tmp_x.extend(promotion)
                # use last week sales volume
                if args.avg_hist:
                    avg_hist = sum(history_volume[(store_id, item_id)][count-8: count-1])/7
                    tmp_x.append(int(avg_hist))
                else:
                    tmp_x.extend(history_volume[(store_id, item_id)][count-8: count-1])
                x_all.append(np.array(tmp_x, dtype=np.int32))
                y_all.append(sales_volume)
            except Exception as e: 
                # some store_ids are missing, so store_id might be ''
                count_error += 1
                #print(e, count_error)
        self.x_all = np.array(x_all)
        self.y_all = np.array(y_all)
        print('x shape:{}  y shape:{} errors:{}\n'.format(self.x_all.shape, self.y_all.shape, count_error))
        if args.save_data:
            with open('data/store_goods_volume.DAT','w') as f:
                for i in tqdm(xrange(len(x_all))):
                    f.write('{} {}\n'.format(x_all[i], y_all[i]))
            print('File saved to store_goods_volume.DAT\n')


    def date_transformation(self, date):
        # 0~6 denote Monday~Sunday
        # 0~3 denote Spring~Winter
        [year, month, day] = [int(i) for i in date.split('-')]
        weekday = dt.date.weekday(dt.date(year, month, day)) + 1
        if 3 <= month <= 5:
            season = self.onehot(0, 4)
        elif 6 <= month <= 8:
            season = self.onehot(1, 4)
        elif 9 <= month <= 11:
            season = self.onehot(2, 4)
        else:
            season = self.onehot(3, 4)
        date_coding = []
        date_coding.extend(season)
        date_coding.extend(self.onehot(month-1, 12, 2))
        date_coding.extend(self.onehot(day-1, 31, 3))
        date_coding.extend(self.onehot(weekday-1, 7, 4)) # dim of date coding: 4+12+31+7=54
        return date_coding, weekday 

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

        #ipdb.set_trace()
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
        #ipdb.set_trace()
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
                #print(e)
                count_error += 1
                continue
            if warehouse_id not in goods2warehouses:
                #ipdb.set_trace()
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
        #ipdb.set_trace()
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
                #print(e)
                count_error += 1
                continue
            if warehouse_id not in goods2warehouses:
                #ipdb.set_trace()
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
        print('stores: %d\ncities: %d' % (len(self.stores), len(self.city2int)))


    def get_items(self):
        for i in xrange(1, self.items_wb.nrows):
            item_id = self.items_wb.cell(i, 0).value
            supplier_id = self.items_wb.cell(i, self.items_wb.ncols - 1).value
            if item_id not in self.items:
                self.items[item_id] = [s.strip() for s in self.items_wb.row_values(i, 1, None)[0::2]]
                self.items[item_id].append(supplier_id)
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
                    try:
                        self.weather2int[j][w] = int(w)
                    except:
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