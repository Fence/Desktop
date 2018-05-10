import ipdb
import time
import argparse
import numpy as np
from optimal_distribution import DistributionModel


def param_init():
    n_stores = 15992
    n_items = 688
    n_warehouses = 2
    param_dict = {}
    param_dict['store_capacity'] = np.zeros((n_stores, n_items))
    param_dict['store_capacity'].fill(10)
    param_dict['warehouse_capacity'] = np.zeros((n_warehouses, n_items))
    param_dict['warehouse_capacity'].fill(80000)
    param_dict['is_store_warehouse'] = np.zeros((n_stores, n_warehouses), dtype=np.bool)
    param_dict['is_store_warehouse'][:8000, 0] = True
    param_dict['is_store_warehouse'][8000:, 1] = True
    param_dict['store_item_to_warehouse'] = {}
    for i in xrange(n_stores):
        warehouse_num = 0 if i < 8000 else 1
        for j in xrange(n_items):
            param_dict['store_item_to_warehouse'][(i, j)] = warehouse_num

    param_dict['order_items_wh'] = np.random.randint(0, 100, (n_warehouses, n_items))
    param_dict['sales_volume_st'] = np.random.randint(0, 10, (n_stores, n_items))
    param_dict['remain_items_st_t1'] = np.zeros((n_stores, n_items))
    param_dict['remain_items_wh_t1'] = np.zeros((n_warehouses, n_items))
    return param_dict


def main():
    ipdb.set_trace()
    param_dict = param_init()
    start = time.time()
    model = DistributionModel(param_dict)
    end = time.time()
    print('Time for build model: %.2fs\n'%(end - start))
    model.buildProblem()
    model.solveProblem(param_dict, verbose=True)


if __name__ == '__main__':
    main()