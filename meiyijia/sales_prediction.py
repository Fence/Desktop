import re
import time
import ipdb
import xlrd


class DateProcessing(object):
    """docstring for DateProcessing"""
    def __init__(self):
        workbook = xlrd.open_workbook('商品门店供应商天气半月档基础数据.xlsx')
        #names = workbook.sheet_names()
        #['门店', '商品', '供应商', '天气', '半月档促销商品', '门店仓位']
        self.stores_wb = workbook.sheet_by_name('门店')
        self.goods_wb = workbook.sheet_by_name('商品')
        self.weather_wb = workbook.sheet_by_name('天气')
        self.suppliers_wb = workbook.sheet_by_name('供应商')
        self.promotions_wb = workbook.sheet_by_name('半月档促销商品')
        self.warehouses_wb = workbook.sheet_by_name('门店仓位')

        ipdb.set_trace()
        self.get_stores()
        self.get_goods()
        self.get_weather()


    def get_stores(self):
        self.stores = {} # {store_id: (city, delivery_cycle) }
        for i in range(1, self.stores_wb.nrows):
            store_id = self.stores_wb.cell(i, 0).value
            if store_id not in self.stores:
                self.stores[store_id] = self.stores_wb.row_values(i, 1, 3)


    def get_goods(self):
        self.goods = {} # {goods_id: [labels, supplier_id] }
        for i in range(1, self.goods_wb.nrows):
            goods_id = self.goods_wb.cell(i, 0).value
            supplier_id = self.goods_wb.cell(i, self.goods_wb.ncols - 1).value
            if goods_id not in self.goods:
                self.goods[goods_id] = self.goods_wb.row_values(i, 1, None)[0::2]
                self.goods[goods_id].append(supplier_id)


    def get_suppliers(self):
        self.suppliers = {} # {supplier_id: {warehouse_id: [data]}}
        for i in range(1, self.suppliers_wb.nrows):
            warehouse_id = self.suppliers_wb.cell(i, 0).value
            supplier_id = self.suppliers_wb.cell(i, 1).value
            if supplier_id in self.suppliers:
                if warehouse_id not in self.suppliers[supplier_id]:
                    self.suppliers[supplier_id][warehouse_id] = self.suppliers_wb.row_values(i, 2, None)
            else:
                self.suppliers[supplier_id] = {warehouse_id: self.suppliers_wb.row_values(i, 2, None)}


    def get_weather(self):
        self.weather = {} # {date: {city: [weather]} }
        for i in range(1, self.weather_wb.nrows):
            date = self.weather_wb.cell(i, 0).value
            city = self.weather_wb.cell(i, 1).value
            if date in self.weather:
                if city not in self.weather[date]:
                    self.weather[date][city] = self.weather_wb.row_values(i, 2, None)
            else:
                self.weather[date] = {city: self.weather_wb.row_values(i, 2, None)}
            

    def get_promotion(self):
        pass


if __name__ == '__main__':
    start = time.time()
    data = DateProcessing()
    end = time.time()
    print('\nTotal time cost: %.2fs\n' % (end - start))