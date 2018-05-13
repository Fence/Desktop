# coding: utf-8
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
        self.goods_wb = workbook.sheet_by_name('商品')
        self.weather_wb = workbook.sheet_by_name('天气')
        self.suppliers_wb = workbook.sheet_by_name('供应商')
        self.promotions_wb = workbook.sheet_by_name('半月档促销商品')
        self.warehouses_wb = workbook.sheet_by_name('门店仓位')

        self.dc2int = {'A': 1, 'B': 2, 'C': 3, '': 4} # delivery_cycle to int
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

        self.get_stores()
        self.get_goods()
        self.get_suppliers()
        self.get_weather()
        self.get_promotions()
        self.get_warehouses()
        self.get_transport_time()


    @timeit
    def compute_inventory(self):
        count = 0
        sales_data = {} # {warehouse: {item: {store: {date: [sale, inventory, stock, return]}}}}
        in_out_data = {} # {warehouse: {item: {store: {date: [stock, return]}}}}
        start_date = dtdt.strptime('2017-04-01','%Y-%m-%d')
        dates = [str(start_date+dt.timedelta(days=i)).split()[0] for i in xrange(387)]
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
        print('\ncount_error: %d\n' % count_error)
        # del in_out_data
        # gc.collect()
        # count = 0
        # ipdb.set_trace()
        # self.warehouse_item_store = {'info': 'data[ warehouses[ items[ stores[ [values] ]]]]', 'data':[]}
        # for warehouse_id in sales_data:
        #     tmp_warehouse = []
        #     for item_id in sales_data[warehouse_id]:
        #         tmp_item = []
        #         for store_id in sales_data[warehouse_id][item_id]:
        #             tmp_store = []
        #             for date, values in sorted(sales_data[warehouse_id][item_id][store_id].iteritems(), key=lambda x:x[0]):
        #                 tmp_value = [date]
        #                 tmp_value.extend(values)
        #                 tmp_store.append(tmp_value)
        #                 count += 1
        #                 if count %1000000 == 0:
        #                     print(count)
        #             tmp_item.append(tmp_store)
        #         tmp_warehouse.append(tmp_item)
        #     self.warehouse_item_store['data'].append(tmp_warehouse)
        return sales_data

    @timeit
    def save_sales_inventory_stock_return(self, sales_data, name='save_file', mode='all'):
        if mode == 'year':
            data = {}
            #ipdb.set_trace()
            for warehouse_id in sales_data:
                for item_id in sales_data[warehouse_id]:
                    for store_id in sales_data[warehouse_id][item_id]:
                        data[str((warehouse_id, store_id, item_id))] = []
                        for date, values in sorted(sales_data[warehouse_id][item_id][store_id].iteritems(), key=lambda x:x[0]):
                            tmp = [date]
                            tmp.extend(values)
                            data[str((warehouse_id, store_id, item_id))].append(tmp)
                        if len(data) % 1000000 == 0:
                            print(len(data))
            with open('data/year_%s.json' % name, 'w') as f:
                json.dump(data, f, indent=2)
                print('Successfully save file as data/year_%s.json\n' % name)
        else:
            with open('data/%s.json' % name, 'w') as f:
                json.dump(sales_data, f, indent=2)
                print('Successfully save file as data/%s.json\n' % name)                  


    def DAT2Matrix(self, args):
        x_all = []
        y_all = []
        count_error = 0
        # start from 2017-04-17 so each record has at least a week history 
        # start_date = dtdt.strptime('2017-04-17', '%Y-%m-%d')
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
                # if date < start_date:
                #     continue
                year, month, day = date.split('-')
                tmp_x = list(self.date_transformation(year, month, day))
                city = self.city2int[self.stores[store_id][0]]
                weather = self.weather[date.replace('-','')][city]
                date = dtdt.strptime(date, '%Y-%m-%d')
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
                for i in tqdm(range(len(x_all))):
                    f.write('{} {}\n'.format(x_all[i], y_all[i]))
            print('File saved to store_goods_volume.DAT\n')


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
            for date in sorted(dates):
                date = str(date).split()[0]
                for (warehouse_id, item_id), sales_volume in data[date].iteritems():
                    f.write('{}\t{}\t{}\t{}\n'.format(date, warehouse_id, item_id, sales_volume))
            print('Successfully save warehouses_sales_volume.DAT\n')


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
            for date in sorted(dates):
                date = str(date).split()[0]
                for (warehouse_id, item_id), (stock_volume, return_volume) in data[date].iteritems():
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(date, warehouse_id, item_id, stock_volume, return_volume))
            print('Successfully save warehouses_stock_return.DAT\n')


    def get_sales_volume_by_day(self):
        data = {}
        dates = []
        count = 0
        for line in open('data/门店商品销售流水数据_2017.DAT'):
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

        for line in open('data/门店商品销售流水数据_2018.DAT'):
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
            for date in sorted(dates):
                date = str(date).split()[0]
                for (store_id, item_id), sales_volume in data[date].iteritems():
                    f.write('{}\t{}\t{}\t{}\n'.format(date, store_id, item_id, sales_volume))
            print('Successfully save days_sales_volume.DAT\n')


    def get_stock_and_return_by_day(self):
        data = {}
        dates = []
        count = 0
        for line in open('data/门店商品进退货数据.DAT'):
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
            for date in sorted(dates):
                date = str(date).split()[0]
                for (store_id, item_id),[stock_volume, return_volume] in data[date].iteritems():
                    f.write('{}\t{}\t{}\t{}\t{}\n'.format(date, store_id, item_id, stock_volume, return_volume))
            print('Successfully save days_stock_and_return_volume.DAT\n')


    def get_stores(self):
        for i in range(1, self.stores_wb.nrows):
            store_id = self.stores_wb.cell(i, 0).value.strip()
            city = self.stores_wb.cell(i, 1).value.strip()
            dc = self.stores_wb.cell(i, 2).value.strip()
            if city not in self.city2int:
                self.city2int[city] = len(self.city2int) + 1
            if store_id not in self.stores:
                self.stores[store_id] = [self.city2int[city], self.dc2int[dc]]
                # self.stores[store_id] = self.stores_wb.row_values(i, 1, 3)
        print('stores: %d' % len(self.stores))


    def get_goods(self):
        for i in range(1, self.goods_wb.nrows):
            item_id = self.goods_wb.cell(i, 0).value
            supplier_id = self.goods_wb.cell(i, self.goods_wb.ncols - 1).value
            if item_id not in self.items:
                self.items[item_id] = [s.strip() for s in self.goods_wb.row_values(i, 1, None)[0::2]]
                self.items[item_id].append(supplier_id)
        print('items: %d' % len(self.items))


    def get_suppliers(self):
        for i in range(1, self.suppliers_wb.nrows):
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
        for i in range(1, self.promotions_wb.nrows):
            date_start = dtdt.strptime(self.promotions_wb.cell(i, 1).value.split()[0], '%Y-%m-%d')
            date_end = dtdt.strptime(self.promotions_wb.cell(i, 2).value.split()[0], '%Y-%m-%d')
            item_id = self.promotions_wb.cell(i, 4).value
            promote_way = self.promotions_wb.cell(i, 5).value
            discount = self.promotions_wb.cell(i, 6).value
            if promote_way not in self.promo2int:
                self.promo2int[promote_way] = len(self.promo2int) + 1
            if discount == '-': # have bought A and pay more money to buy B
                discount = '0'
            promote_way = self.promo2int[promote_way]
            if item_id in self.promotions:
                self.promotions[item_id].append([date_start, date_end, promote_way, discount])
            else:
                self.promotions[item_id] = [[date_start, date_end, promote_way, discount]]
        print('items with promotions: %d' % len(self.promotions))


    def get_warehouses(self):
        for i in range(1, self.warehouses_wb.nrows):
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
                orderdays = [int(d) for d in data[0][0].split('/')[:-1]]
                arrivedays = [int(d[2]) for d in data]
                transtime = int(data[0][-1])
                if warehouse_id in self.warehouse_item_transtime:
                    if item_id not in self.warehouse_item_transtime[warehouse_id]:
                        # transport time format = [orderdays, arrivedays, transtime]
                        self.warehouse_item_transtime[warehouse_id][item_id] = [orderdays, arrivedays, transtime] 
                else:
                    self.warehouse_item_transtime[warehouse_id] = {item_id: [orderdays, arrivedays, transtime]}


if __name__ == '__main__':
    data = DateProcessing()
    # data.get_stock_return_by_warehouse()
    # sales_data = data.compute_inventory()
    # data.save_sales_inventory_stock_return(sales_data, name='new_sales_inventory_stock_return', mode='all')