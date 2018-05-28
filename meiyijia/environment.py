# coding: utf-8
import ipdb
import json
import argparse
import numpy as np
import datetime as dt
from datetime import datetime as dtdt
from data_processor import DateProcessing, timeit


class Environment(object):
    """docstring for Environment"""
    def __init__(self, args, data):
        #self.data = DateProcessing()
        self.args = args
        self.items = data.items
        self.stores = data.stores
        self.promotions = data.promotions
        self.date_transformation = data.date_transformation
        self.wh_item = json.load(open('data/warehouse_item_dict.json','r')) # data.warehouse_item # {warehouse_id: [item_ids]}
        self.wh_data = json.load(open('data/warehouse_sales_inventory_stock_return.json','r'))
        self.wh_item_time = data.warehouse_item_transtime
        self.wh_ind = self.item_ind = 0
        self.dc2num = data.int2onehot(data.dc2int, 1) # deliver time
        self.city2num = data.int2onehot(data.city2int, 6)
        self.promo2num = data.int2onehot(data.promo2int, 7)
        self.use_padding = args.use_padding
        self.min_stores = args.min_stores
        #self.min_dates = args.min_dates
        #self.n_stores = args.n_stores
        #self.emb_dim = args.emb_dim
        self.random = args.random
        self.train_start_date = args.train_start_date
        self.test_start_date = args.test_start_date
        self.train_end_date = args.train_end_date
        self.test_end_date = args.test_end_date
        #self.max_action = 0
        self.action_count = 0
        self.seen_actions = {}
        self.target_orders = {}
        #self.restart(is_train)


    def restart(self, is_train):
        status = -1
        while status < 0:
            status = self._restart(self.random, is_train)
            if status < 0:
                self.args.count_items_of_missing_day += 1


    def _restart(self, random, is_train):
        #ipdb.set_trace()
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
            except:
                continue
            if len(self.wh_item_data) < self.min_stores: # no more than 100 stores saling this item
                #if is_train:
                #    print('wh {} item {} < {} stores\n'.format(self.warehouse_id, self.item_id, self.min_stores))
                continue
            else:
                values = self.get_warehouse_item_features()
                if len(values) == 1:
                    continue
                else:
                    self.valid_days, self.orderdays, self.arrivedays, self.transp_time = values
                    break
        if is_train:
            self.start_date, self.end_date = self.train_start_date, self.train_end_date
        else:
            self.start_date, self.end_date = self.test_start_date, self.test_end_date
        self.cur_date = self.start_date
        return self.get_valid_state(is_train)

    
    def get_valid_state(self, is_train):
        while True:
            values = self.generate_state(self.cur_date, is_train)
            if values[-1] > 0:
                self.state, self.pred_time = values[0], values[1]
                return values[-1]
            elif values[-1] == -1: # less than 100 stores
                return -1
            else: # values[-1] == 0, current item/warehouse/date doesn't exist
                # self.cur_date = self.update_date(self.cur_date)
                # if self.cur_date > self.end_date:
                #     return -2
                return -2


    def get_warehouse_item_features(self):
        # get warehouse-item features
        try:
            valid_days = float(self.items[self.item_id][-2])
            orderdays, arrivedays, transp_time = self.wh_item_time[self.warehouse_id][self.item_id]
        except Exception as e:
            # ipdb.set_trace()
            return [-1]
        # if len(self.wh_item_data) < self.min_stores: # no more than 100 stores saling this item
        #     if is_train:
        #         print('wh {} item {} < {} stores\n'.format(self.warehouse_id, self.item_id, self.min_stores))
        #     return [-1]
        next_week_order_days = [d+7 for d in orderdays]
        next_week_arrive_days = [d+7 for d in arrivedays]
        orderdays.extend(next_week_order_days)
        arrivedays.extend(next_week_arrive_days)
        return valid_days, orderdays, arrivedays, transp_time


    def generate_state(self, cur_date, is_train):
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
        if self.item_id in self.promotions:
            for p in self.promotions[self.item_id]:
                # if this item is promoted at this day
                if p[0] <= cur_date <= p[1]: 
                    promotion[:-1] = self.promo2num[p[-2]]
                    promotion[-1] = p[-1]
                    break
        # get store-specific features
        count_less_than_dates = 0
        state, x_i_j, deliver_time = [], np.zeros(4), np.zeros(4)
        for store_id in self.wh_item_data:
            if cur_date in self.wh_item_data[store_id]:
                x_i_j += np.array(self.wh_item_data[store_id][cur_date])
                valid_count += 1
            # pad zero sales, inventory, stock and return values for missing dates
            elif self.use_padding:
                x_i_j += [0, 0, 0, 0]
                valid_count += 1
            else:
                pass
                #continue
            if self.stores[store_id][1] in self.dc2num:
                deliver_time += self.dc2num[self.stores[store_id][1]]
        
        tmp_count, tmp_day, tmp_trans = np.zeros(1), np.zeros(1), np.zeros(1)
        tmp_count.fill(valid_count)
        tmp_day.fill(self.valid_days)
        tmp_trans.fill(self.transp_time)
        state.extend(tmp_count)    # dim = 1
        state.extend(tmp_day)      # dim = 1
        state.extend(tmp_trans)    # dim = 1
        state.extend(x_i_j)        # dim = 4
        state.extend(deliver_time) # dim = 4
        state.extend(date_emb)     # dim = 4 + 12 + 31 + 7 = 54
        state.extend(promotion)    # dim = 8 + 1 = 9
        #print(self.item_id, self.warehouse_id, deliver_time, x_i_j, valid_count)
        return np.array(state), pred_time, valid_count


    def generate_state_old(self, cur_date, is_train):
        """ all features of an item_warehouse state: 73 (no city and no weather)
        sales: 1,  inventory: 1, stock: 1, return: 1,  deliver_time: 4, valid_days: 1, transp_time: 1,
        season: 4, month: 12,    day: 31,  weekday: 7, promotion: 8+1,  weather: 7     city: 21
        """
        #state = np.zeros([self.n_stores, self.emb_dim], dtype=np.float32)
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
        if self.item_id in self.promotions:
            for p in self.promotions[self.item_id]:
                # if this item is promoted at this day
                if p[0] <= cur_date <= p[1]: 
                    promotion[:-1] = self.promo2num[p[-2]]
                    promotion[-1] = p[-1]
                    break
        # get store-specific features
        count_less_than_dates = 0
        state, x_i_j, deliver_time = [], np.zeros(4), np.zeros(4)
        for store_id in self.wh_item_data:
            # skip items that ares saled less than min_days
            # if len(self.wh_item_data[store_id]) < self.min_dates:
            #     count_less_than_dates += 1
            #     continue
            if cur_date in self.wh_item_data[store_id]:
                x_i_j += np.array(self.wh_item_data[store_id][cur_date])
                valid_count += 1
            # pad zero sales, inventory, stock and return values for missing dates
            elif self.use_padding:
                x_i_j += [0, 0, 0, 0]
                valid_count += 1
            else:
                pass
                #continue
            if self.stores[store_id][1] in self.dc2num:
                deliver_time += self.dc2num[self.stores[store_id][1]]
            # else:
            #     continue
            # no city and no weather
            # x_i_j.extend(deliver_time)
            # x_i_j.append(self.valid_days)
            # x_i_j.append(self.transp_time)
            # x_i_j.extend(date_emb)
            # x_i_j.extend(promotion)
            # #ipdb.set_trace()
            # if len(x_i_j) > self.emb_dim:
            #     state[valid_count] = x_i_j[:self.emb_dim]
            # else:
            #     state[valid_count][:len(x_i_j)] = x_i_j
            # valid_count += 1
            # if valid_count >= self.n_stores:
            #     break
        #ipdb.set_trace()
        # if cur_date == self.start_date and is_train:
        #     print('wh {} item {} < {} dates: {}/{} stores\n'.format(
        #         self.warehouse_id, self.item_id, self.min_dates, count_less_than_dates, len(self.wh_item_data)))
        tmp_count, tmp_day, tmp_trans = np.zeros(4), np.zeros(4), np.zeros(4)
        tmp_count.fill(valid_count)
        tmp_day.fill(self.valid_days)
        tmp_trans.fill(self.transp_time)
        state.extend(tmp_count)    # dim = 4
        state.extend(tmp_day)      # dim = 4
        state.extend(tmp_trans)    # dim = 4
        state.extend(x_i_j)        # dim = 4
        state.extend(deliver_time) # dim = 4
        state.extend(date_emb)     # dim = 4 + 12 + 31 + 7 = 54
        state.extend(promotion)    # dim = 8 + 1 = 9
        #print(self.item_id, self.warehouse_id, deliver_time, x_i_j, valid_count)
        return np.array(state), pred_time, valid_count


    def get_target(self):
        tmp_date = self.update_date(self.cur_date, self.pred_time[0])
        target_sales = target_stocks = target_returns = 0
        #ipdb.set_trace()
        for d in xrange(self.pred_time[0], self.pred_time[1]):
            try:
                sales, _, stocks, returns = self.wh_data[self.warehouse_id][self.item_id][tmp_date]
                target_sales += sales
                target_stocks += stocks
                target_returns += returns
            except Exception as e: # tmp_date may not be in the keys
                #ipdb.set_trace()
                continue
            tmp_date = self.update_date(tmp_date)

        target_order = target_sales # target_stocks - target_returns #
        return target_order


    def boosting_step(self, is_train):
        self.cur_date = self.update_date(self.cur_date)
        status = self.get_valid_state(is_train) # current state is new state
        if self.cur_date > self.end_date or status < 0:
            #ipdb.set_trace()
            terminal = True # change terminal 1
            self.restart(is_train) # current state is new state
        else:
            terminal = False # change terminal 2
        return terminal


    def step(self, action, is_train):
        target_order = self.get_target() 
        act = action# % 10
        self.action_count += 1
        if act in self.seen_actions:
            self.seen_actions[act] += 1
        else:
            self.seen_actions[act] = 1
        # if target_order > self.max_action:
        #     self.max_action = int(target_order)
        if target_order in self.target_orders:
            self.target_orders[target_order] += 1
        else:
            self.target_orders[target_order] = 1
        # if self.action_count % 100 == 0:
        #     most_action = sorted(self.seen_actions.iteritems(), key=lambda x:x[1])[-1]
        #     print('max_action: {}  most_action: {}\n'.format(self.max_action, most_action))
        
        # r = f(sales, stocks, returns, inventory)
        #reward = 100 - 10*np.sqrt(np.abs(target_order - action))
        reward = -self.clip_reward(np.abs(target_order - action), 1000)
        # if target_order - action >= 0:
        #     reward = -100 * np.log(target_order - action + 1)
        # else:
        #     reward = -100 * np.log2(action - target_order + 1)
        # if target_order > 0:
        #     reward = 10 - np.sqrt(np.abs(target_order - action))
        # else:
        #     reward = -150 # punish that requried future dates are not appear in the datasets
        #print(self.cur_date)
        self.cur_date = self.update_date(self.cur_date)
        status = self.get_valid_state(is_train) # current state is new state
        if self.cur_date > self.end_date or status < 0:
            #ipdb.set_trace()
            terminal = True # change terminal 1
            self.restart(is_train) # current state is new state
        else:
            terminal = False # change terminal 2
        return self.state, reward, terminal


    def clip_reward(self, raw_reward, u_bound):
        new_reward = raw_reward if raw_reward <= u_bound else u_bound
        return new_reward


    def test_act(self):
        self.cur_date = self.update_date(self.cur_date)
        if self.cur_date > self.end_date:
            self.restart(is_train=True)
        self.get_valid_state(is_train=True)


    def update_date(self, cur_date, days=1):
        date = dtdt.strptime(cur_date, '%Y-%m-%d')
        new_date = date + dt.timedelta(days=days)
        return str(new_date).split()[0]


    def get_state(self):
        return self.state

    @timeit
    def compute_dataset_features(self):
        count = 0 # count = 575
        item_stores = {} # 1~99: 341;      100~999: 79;       >=1000: 316
        store_dates = {} # 1~99: 996,423 (80%);  100~199: 179,158 (14%);  >= 200: 76,647 (6%)
        # all: 1,252,228;  1~33: 640,306 (51%);  34 ~ 66: 229,821 (18%);  67 ~99: 126,296 (10%)
        wh_item_stores = {}
        print('Loading data and computing ...')
        for w, items in self.wh_item.iteritems():
            wh_item_stores[w] = {}
            for i in items:
                try:
                    d = json.load(open('data/items_data/%s_%s.json'%(w, i)))
                    wh_item_stores[w][i] = d.keys()
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
                    'store_dates': store_dates,
                    'wh_item_stores': wh_item_stores}
            json.dump(data, f, indent=4)
            print('Finish!')


    def padding_missing_data(self):
        dates = [self.update_date('2016-11-01', i) for i in xrange(538)]
        for w, items in self.wh_item.iteritems():
            for i in items:
                try:
                    item_stores = json.load(open('data/items_data/%s_%s.json'%(w, i)))
                    for store_id in item_stores:
                        valid_days = 0
                        for date in dates:
                            if date not in item_stores[store_id]:
                                item_stores[store_id][date] = [0, 0, 0, 0]
                            else:
                                valid_days += 1
                        item_stores[store_id]['valid_days'] = valid_days
                    with open('data/full_items_data/%s_%s.json'%(w, i),'w') as f:
                        json.dump(item_stores, f, indent=2)
                        print('Saved data/full_items_data/%s_%s.json'%(w, i))

                except:
                    continue
        print('All done!')


    def test_multi_threads(self, n_threads):
        import threading
        self.date_states = {}
        self._restart(self.random, is_train=True)
        records = {'valid': 0}
        def get_a_state(self, thread_id, n_threads, records):
            cur_date = self.update_date(self.start_date, thread_id)
            while cur_date <= self.end_date:
                values = self.generate_state(cur_date, is_train=True)
                if values[-1] <= 0:
                    self.date_states[cur_date] = {'state': [], 'pred_time': []}
                else:
                    records['valid'] += 1
                    self.date_states[cur_date] = {'state': values[0], 'pred_time': values[1]}
                cur_date = self.update_date(cur_date, n_threads)
                print('thread {}\tcur_date {}\tn_dates {}'.format(thread_id, cur_date, len(self.date_states)))

        threads = [threading.Thread(target=get_a_state, args=(self, i, n_threads, records)) for i in xrange(n_threads)]
        for thread in threads:
            thread.start()
            thread.join()
        #ipdb.set_trace()
        print(records['valid'], len(self.date_states))


def test():
    data = json.load(open('data/warehouse_sales_inventory_stock_return.json','r'))
    for w in data:
        for i in data[w]:
            if len(data[w][i]) > 100:
                ipdb.set_trace()
                print('stop')
            print('wh {} item {}: {} dates'.format(w, i, len(data[w][i])))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    data = DateProcessing()
    env = Environment(args, data)
    # ipdb.set_trace()
    # env.padding_missing_data()
    # env.test_multi_threads(4)
    env.compute_dataset_features()
    # for i in xrange(400):
    #     env.test_act()