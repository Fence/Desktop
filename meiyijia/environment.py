# coding: utf-8
import ipdb
import json
import numpy as np
import datetime as dt
from datetime import datetime as dtdt
from data_processor import DateProcessing


class Environment(object):
    """docstring for Environment"""
    def __init__(self, args):
        self.data = DateProcessing()
        self.wh_item = self.data.warehouse_item
        self.wh_data = json.load(open('data/warehouse_sales_inventory_stock_return.json','r'))
        self.wh_item_time = self.data.warehouse_item_transtime
        self.wh_ind = self.item_ind = 0
        #self.stores = self.data.stores.keys()
        self.dc2num = self.data.int2onehot(self.data.dc2int)
        self.city2num = self.data.int2onehot(self.data.city2int)
        self.n_stores = args.max_stores_per_warehouse
        self.emb_dim = args.emb_dim
        self.start_date = '2017-04-01'
        self.end_date = '2018-04-22'
        self.terminal = False
        self.max_action = 0
        self.action_count = 0
        self.seen_actions = {}
        self.get_valid_resart()


    def get_valid_resart(self, random=False):
        status = -1
        while status < 0:
            status = self.restart(random)


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
                self.wh_item_data = json.load(open('data/items_data/%s_%s.json'%(self.warehouse_id, self.item_id),'r'))
                break
            except:
                continue
        self.cur_date = self.start_date
        return self.get_valid_state()

    
    def get_valid_state(self):
        while True:
            valid = self.generate_state()
            if valid > 0:
                #print('Valid: wh {} item {} date {}.\n'.format(self.warehouse_id, self.item_id, self.cur_date))
                return valid
            else:
                self.cur_date = self.update_date(self.cur_date)
                if self.cur_date > self.end_date:
                    return -1


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
        # get store-specific features
        for store_id in self.wh_item_data:
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
                    print(e)
            
        #if valid_count == 0:
        #    print('No record: warehouse {} item {} date {}.'.format(self.warehouse_id, self.item_id, self.cur_date))
        #else:
        #    print('Has record: warehouse {} item {} date {}.\n'.format(self.warehouse_id, self.item_id, self.cur_date))
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

        act = action % 10
        self.action_count += 1
        if act in self.seen_actions:
            self.seen_actions[act] += 1
        else:
            self.seen_actions[act] = 1
        if target_sales > self.max_action:
            self.max_action = target_sales
        if self.action_count % 100 == 0:
            most_action = sorted(self.seen_actions.iteritems(), key=lambda x:x[1])[-1]
            most_action[0] *= 10
            print('max_action: {}  most_action: {}\n'.format(self.max_action, most_action))
        
        # r = f(sales, stocks, returns, inventory)
        reward = 10 - np.sqrt(np.abs(target_sales - action))
        self.cur_date = self.update_date(self.cur_date)
        status = self.get_valid_state() # current state is new state
        if self.cur_date > self.end_date or status < 0:
            self.terminal = True # change terminal 1
            self.get_valid_resart() # current state is new state
        else:
            self.terminal = False # change terminal 2
        return self.state, reward, self.terminal


    def test_act(self):
        self.cur_date = self.update_date(self.cur_date)
        if self.cur_date > self.end_date:
            self.restart()
        self.get_valid_state()


    def update_date(self, cur_date):
        date = dtdt.strptime(cur_date, '%Y-%m-%d')
        new_date = date + dt.timedelta(days=1)
        return str(new_date).split()[0]


    def get_state(self):
        return self.state

    def is_terminal(self):
        return self.terminal



if __name__ == '__main__':
    ipdb.set_trace()
    env = Environment('')
    for i in xrange(400):
        env.test_act()