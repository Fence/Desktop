# coding: utf-8
import ipdb
import json
import argparse
import numpy as np
import datetime as dt
from datetime import datetime as dtdt
from data_processor import DateProcessing


class Environment(object):
    """docstring for Environment"""
    def __init__(self, args):
        self.data = DateProcessing()
        self.wh_item = self.data.warehouse_item # {warehouse_id: [item_ids]}
        self.wh_data = self.data.warehouse_item_sales_volume
        self.wh_item_time = self.data.warehouse_item_transtime
        self.wh_ind = self.item_ind = 0
        #self.stores = self.data.stores.keys()
        self.dc2num = self.data.int2onehot(self.data.dc2int)
        self.city2num = self.data.int2onehot(self.data.city2int)
        self.min_stores = args.min_stores
        self.min_dates = args.min_dates
        self.n_stores = args.n_stores
        self.emb_dim = args.emb_dim
        self.random = args.random
        self.start_date = '2017-04-01'
        self.end_date = '2018-04-22'
        self.max_action = 0
        self.action_count = 0
        self.seen_actions = {}
        self.get_valid_resart()


    def get_valid_resart(self):
        status = -1
        while status < 0:
            status = self.restart(self.random)


    def restart(self, random):
        while True:
            if random:
                self.wh_ind = np.random.randint(len(self.wh_item))
                self.warehouse_id = self.wh_item.keys()[self.wh_ind]
                self.item_ind = np.random.randint(len(self.wh_item[self.warehouse_id]))
                self.item_id = self.wh_item[self.warehouse_id][self.item_ind]
            else:
                self.warehouse_id = self.wh_item.keys()[self.wh_ind]
                if self.item_ind >= len(self.wh_item[self.warehouse_id]) - 1:
                    self.wh_ind = (self.wh_ind + 1) % len(self.wh_item)
                    self.warehouse_id = self.wh_item.keys()[self.wh_ind]
                    self.item_ind = 0
                else:
                    self.item_ind += 1
                self.item_id = self.wh_item[self.warehouse_id][self.item_ind]
            try:
                self.wh_item_data = json.load(
                    open('data/items_data/%s_%s.json'%(self.warehouse_id, self.item_id),'r'))
                break
            except:
                continue
        self.cur_date = self.start_date
        return self.get_valid_state()

    
    def get_valid_state(self):
        while True:
            valid = self.generate_state()
            if valid > 0:
                #print('Valid: wh {} item {} date {}.\n'.format(
                #       self.warehouse_id, self.item_id, self.cur_date))
                return valid
            elif valid == -1: # less than 100 stores
                return -1
            else: # current date doesn't exist
                self.cur_date = self.update_date(self.cur_date)
                if self.cur_date > self.end_date:
                    return -2


    def generate_state(self):
        self.state = np.zeros([self.n_stores, self.emb_dim], dtype=np.float32)
        valid_count = 0
        # get warehouse-item features
        try:
            valid_days = float(self.data.items[self.item_id][-2])
            self.transp_time = self.wh_item_time[self.warehouse_id][self.item_id]
            year, month, day = self.cur_date.split('-')
            tmp_date = list(self.data.date_transformation(year, month, day))
            promotion = [0, 0] # way, discount
            if self.item_id in self.data.promotions:
                for p in self.data.promotions[self.item_id]:
                    # if this item is promoted at this day
                    if p[0] <= self.cur_date <= p[1]: 
                        promotion = p[2:]
                        break
        except Exception as e:
            #print(e)
            return 0
        if len(self.wh_item_data) < self.min_stores: # no more than 100 stores saling this item
            print('wh {} item {} < {} stores\n'.format(self.warehouse_id, self.item_id, self.min_stores))
            return -1
        # get store-specific features
        count_less_than_dates = 0
        for store_id in self.wh_item_data:
            if len(self.wh_item_data[store_id]) < self.min_dates:
                count_less_than_dates += 1
                continue
            if self.cur_date in self.wh_item_data[store_id]:
                try:
                    x_i_j = self.wh_item_data[store_id][self.cur_date]
                    deliver_time = self.dc2num[self.data.stores[store_id][1]]
                    # no city and no weather
                    x_i_j.extend(deliver_time)
                    x_i_j.append(valid_days)
                    x_i_j.append(self.transp_time[-1])
                    x_i_j.extend(tmp_date)
                    x_i_j.extend(promotion)
                    #ipdb.set_trace()
                    self.state[valid_count][:len(x_i_j)] = x_i_j
                    valid_count += 1
                    if valid_count >= self.n_stores:
                        break
                except Exception as e:
                    #print(e)
                    continue
            
        # if valid_count == 0:
        #    print('No record: warehouse {} item {} date {}.'.format(
        #               self.warehouse_id, self.item_id, self.cur_date))
        # else:
        #    print('Has record: warehouse {} item {} date {}.\n'.format(
        #               self.warehouse_id, self.item_id, self.cur_date))
        if self.cur_date == self.start_date:
            print('wh {} item {} < {} dates: {}/{} stores\n'.format(
                self.warehouse_id, self.item_id, self.min_dates, count_less_than_dates, len(self.wh_item_data)))
        return valid_count


    def step(self, action):
        tmp_date = self.cur_date
        target_sales = target_stocks = target_returns = 0
        for d in range(int(self.transp_time[-1])):
            try:
                sales, _, stocks, returns = self.wh_data[self.warehouse_id][self.item_id][tmp_date]
                target_sales += sales
                target_stocks += stocks
                target_returns += returns
            except Exception as e: # tmp_date may not be in the keys
                #ipdb.set_trace()
                continue
            tmp_date = self.update_date(tmp_date)

        target_order = target_stocks - target_returns # target_sales
        act = action % 10
        self.action_count += 1
        if act in self.seen_actions:
            self.seen_actions[act] += 1
        else:
            self.seen_actions[act] = 1
        if target_order > self.max_action:
            self.max_action = int(target_order)
        # if self.action_count % 100 == 0:
        #     most_action = sorted(self.seen_actions.iteritems(), key=lambda x:x[1])[-1]
        #     print('max_action: {}  most_action: {}\n'.format(self.max_action, most_action))
        
        # r = f(sales, stocks, returns, inventory)
        reward = 10 - np.sqrt(np.abs(target_order - action))
        self.cur_date = self.update_date(self.cur_date)
        status = self.get_valid_state() # current state is new state
        if self.cur_date > self.end_date or status < 0:
            #ipdb.set_trace()
            terminal = True # change terminal 1
            self.get_valid_resart() # current state is new state
        else:
            terminal = False # change terminal 2
        return self.state, reward, terminal


    def test_act(self):
        self.cur_date = self.update_date(self.cur_date)
        if self.cur_date > self.end_date:
            self.get_valid_resart()
        self.get_valid_state()


    def update_date(self, cur_date):
        date = dtdt.strptime(cur_date, '%Y-%m-%d')
        new_date = date + dt.timedelta(days=1)
        return str(new_date).split()[0]


    def get_state(self):
        return self.state


    def compute_dataset_features(self):
        count = 0 # count = 575
        item_stores = {} # 1~99: 341;      100~999: 79;       >=1000: 316
        store_dates = {} # 1~99: 996,423 (80%);  100~199: 179,158 (14%);  >= 200: 76,647 (6%)
        # all: 1,252,228;  1~33: 640,306 (51%);  34 ~ 66: 229,821 (18%);  67 ~99: 126,296 (10%)
        print('Loading data and computing ...')
        for w, items in self.wh_item.iteritems():
            for i in items:
                try:
                    d = json.load(open('data/items_data/%s_%s.json'%(w, i)))
                    if len(d) not in item_stores:
                        item_stores[len(d)] = 1
                    else:
                        item_stores[len(d)] += 1
                    for store_id in d:
                        dates = [date for date in d[store_id] if self.start_date <= date <= self.end_date]
                        if len(dates) not in store_dates:
                            store_dates[len(dates)] = 1
                        else:
                            store_dates[len(dates)] += 1
                except:
                    count += 1
        a = sorted(item_stores.iteritems(), key=lambda x:x[1], reverse=True)
        b = sorted(store_dates.iteritems(), key=lambda x:x[1], reverse=True)
        with open('data/datasets.json','w') as f:
            print('Saving data ...')
            data = {'count': count,
                    'item_stores': item_stores,
                    'store_dates': store_dates}
            json.dump(data, f, indent=2)
            print('Finish!')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb_dim', type=int, default=30)
    parser.add_argument('-n_stores', type=int, default=2000)
    parser.add_argument('-random', type=bool, default=True)
    args = parser.parse_args()
    #ipdb.set_trace()
    env = Environment(args)
    env.compute_dataset_features()
    # for i in xrange(400):
    #     env.test_act()