import numpy as np
import cvxpy as cvx
from cvxpy import Variable, Parameter, Constant
from cvxpy import Minimize, Problem
from cvxpy import sum_entries, min_elemwise

class DistributionModel(object):
    """docstring for Distribution"""
    def __init__(self, param_dict):
        self.n_items = param_dict['store_capacity'].shape[1] # max_item_num
        self.n_stores = param_dict['store_capacity'].shape[0] # max_store_num
        self.n_warehouses = param_dict['warehouse_capacity'].shape[0] # max_warehouse_num
        self.store_item_to_warehouse = param_dict['store_item_to_warehouse']
        self.addConstant()
        self.addParameter()
        self.addVariable()
        self.addConstraint()


    def feed_dict(self, param_dict):
        self.store_capacity.value = param_dict['store_capacity']
        self.warehouse_capacity.value = param_dict['warehouse_capacity']
        self.is_store_warehouse.value = param_dict['is_store_warehouse']
        
        self.order_items_wh.value = param_dict['order_items_wh']
        self.sales_volume_st.value = param_dict['sales_volume_st']
        self.remain_items_st_t1.value = param_dict['remain_items_st_t1']
        self.remain_items_wh_t1.value = param_dict['remain_items_wh_t1']


    def addConstant(self):
        n_items  = self.n_items
        n_stores = self.n_stores
        n_warehouses = self.n_warehouses
        self.warehouse_capacity = Parameter(rows=n_warehouses, cols=n_items)
        self.store_capacity     = Parameter(rows=n_stores, cols=n_items) 
        self.is_store_warehouse = Parameter(rows=n_stores, cols=n_warehouses)
        

    def addParameter(self):
        n_items  = self.n_items
        n_stores = self.n_stores
        n_warehouses = self.n_warehouses
        #self.sales_volume_wh = Parameter(rows=n_warehouses, cols=n_items)
        self.sales_volume_st = Parameter(rows=n_stores, cols=n_items)
        self.order_items_wh  = Parameter(rows=n_warehouses, cols=n_items)

        self.remain_items_wh_t1 = Variable(rows=n_warehouses, cols=n_items) # L_i^{t-1,k}
        self.remain_items_st_t1 = Variable(rows=n_stores, cols=n_items) # L_i^{t-1,j}


    def addVariable(self):
        """st: stores    wh: warehouses"""
        n_items  = self.n_items
        n_stores = self.n_stores
        n_warehouses = self.n_warehouses
        self.return_items_wh  = [Variable(rows=n_stores, cols=n_items) for _ in xrange(n_warehouses)] # R_i^{k,j}
        self.send_items_wh    = [Variable(rows=n_stores, cols=n_items) for _ in xrange(n_warehouses)] # D_i^{t,k,j}
        
        self.remain_items_wh    = Variable(rows=n_warehouses, cols=n_items) # L_i^{t,k}
        #self.expire_items_wh    = Variable(rows=n_warehouses, cols=n_items) # P_i^{t,k}

        self.remain_items_st    = Variable(rows=n_stores, cols=n_items) # L_i^{t,j}
        #self.return_items_st    = Variable(rows=n_stores, cols=n_items) # R_i^{t,j}

    
    def addConstraint(self):
        """stores: send --> sale --> return --> remain"""
        self.constraint1 = [] # R_i^{t,k,j} <= D_i^{t,k,j} + L_i^{t,k,j} - F_i^{t,k,j}
        for k in xrange(self.n_warehouses):
            for j in xrange(self.n_stores):
                for i in xrange(self.n_items):
                    restrict = self.return_items_wh[k][j, i] <= self.send_items_wh[k][j, i] \
                            + (self.remain_items_st[j, i]-self.sales_volume_st[j, i])*self.is_store_warehouse[j, k]
                    self.constraint1.append(restrict)

        self.constraint2 = [] # D_i^{t,k,j} >= F_i^{t,k,j} - L_i^{t-1,k,j}
        for k in xrange(self.n_warehouses):
            for j in xrange(self.n_stores):
                for i in xrange(self.n_items):
                    restrict = self.send_items_wh[k][j, i] >= \
                            (self.sales_volume_st[j, i]-self.remain_items_st_t1[j, i])*self.is_store_warehouse[j, k]
                    self.constraint2.append(restrict)

        self.constraint3 = [] # D_i^{t,k} <= L_i^{t-1,k} + B_i^{t,k}
        for k in xrange(self.n_warehouses):
            for i in xrange(self.n_items):
                restrict = sum_entries(self.send_items_wh[k], axis=0)[0, i] <= \
                        self.remain_items_wh_t1[k, i] + self.order_items_wh[k, i]
                self.constraint3.append(restrict)

        self.constraint4 = [] # L_i^{t-1,k} + B_i^{t,k} <= C_i^k
        for k in xrange(self.n_warehouses):
            for i in xrange(self.n_items):
                restrict = self.remain_items_wh_t1[k, i] + self.order_items_wh[k, i] \
                            <= self.warehouse_capacity[k, i]
                self.constraint4.append(restrict)

        self.constraint5 = [] # L_i^{t,k,j} = L_i^{t-1,k,j} + D_i^{t,k,j} - F_i^{t,k,j} - R_i^{t,k,j}
        for j in xrange(self.n_stores):
            for i in xrange(self.n_items):
                if (j, i) in self.store_item_to_warehouse:
                    k = self.store_item_to_warehouse[(j, i)]
                    restrict = self.remain_items_st[j, i] == self.remain_items_st_t1[j, i] \
                            + self.send_items_wh[k][j, i] - self.sales_volume_st[j, i] - self.return_items_wh[k][j, i]
                    self.constraint5.append(restrict)

        self.constraint6 = [] # L_i^{t-1,j} + D_i^{t,k,j} <= C_i^j
        for j in xrange(self.n_stores):
            for i in xrange(self.n_items):
                if (j, i) in self.store_item_to_warehouse:
                    k = self.store_item_to_warehouse[(j, i)]
                    restrict = self.remain_items_st_t1[j, i] + self.send_items_wh[k][j, i] \
                                <= self.store_capacity[k, i]
                    self.constraint6.append(restrict)

        self.constraint7 = [] # D_i^{t,k,j} >= 0
        for k in xrange(self.n_warehouses):
            for j in xrange(self.n_stores):
                for i in xrange(self.n_items):
                    restrict = self.send_items_wh[k][j, i] >= 0
                    self.constraint7.append(restrict)


    def buildProblem(self):
        return_items = None
        for k in xrange(self.n_warehouses):
            if return_items:
                return_items += sum_entries(self.return_items_wh[k])
            else:
                return_items = sum_entries(self.return_items_wh[k])
        remain_items = sum_entries(self.remain_items_wh)
        self.objective = Minimize(- return_items - remain_items)

        self.constraints = self.constraint1 + self.constraint2 + self.constraint3 \
                + self.constraint4 + self.constraint5 + self.constraint6 + self.constraint7
        self.problem = Problem(self.objective, self.constraints)


    def solveProblem(self, param_dict, verbose=None):
        self.feed_dict(param_dict)
        self.problem.solve(verbose=verbose)