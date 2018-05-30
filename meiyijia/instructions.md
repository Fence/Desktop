# 1 问题建模

##1.1 订货模型

​	1、美宜佳总部估算每个商店每个商品的需求，结合每个仓库当前库存，向供应商订货

​	2、供应商将商品送往仓库

PS：供应商送货需要时间，订货需要依据未来的商品销量，以basic_data.xlsx中的供应商那个表中第2行为例：

​	‘’仓库27，供应商1000973，可到货周期4/6，订货日期2/4，在途时间2‘’

假设今天是周三（5.30），因此只能周四订货，周六（6.2）到货，一天内配送到对应的商店（要考虑商店的配送周期，当前的版本还没有考虑），下一次到货时间是下周四（6.7），也就是说今天订货量（即预测的数值）应该是6.2~6.6共五天的对应商店对应商品的销量之和。

## 1.2 配送模型

​	1、美宜佳总部估计每个商店每个商品的订货量（配送量），即根据订货到货量，仓库存量，商店存量，历史进退货等，预测配送量

​	2、预测模型的特征输入与订货模型的类似，区别在于预测的数值。这里预测的数值可以（但不限定）是$\bar{d}_t^{w,s,i} = d_t^{w,s,i} - d_{t+\tau}^{w,s,i}$ 其中t是时间（天），w是仓库warehouse，s是商店store，i是商品item，$\tau$是商品i的保质期（天）， $\bar{d}_t^{w,s,i}$是预测当天的目标配送量，也就是等于数据集中当天的进货量减去过保质期之后退货量。



# 2 数据介绍

#2.1 基本数据

​	1、basic_data.xlsx中包含门店，商品，供应商，天气，促销，仓库等信息

​	2、problem_definition.docx里面是问题描述

​	3、中文的数据.DAT是原始数据，数据格式见问题定义第五点

# 2.2 预处理后的数据

​	由于原始数据是每个时刻的数据，因此需要整理为每天的数据：

​	1、days_sales_volume.DAT 是整理后的格式为 “日期  门店  商品  销量” 的数据

​	2、days_stock_and_return_volume.DAT 是整理后的格式为 “日期  门店  商品  进货量  退货量” 的数据



​	由于订货量是以“仓库——商品”来确定，每种商品只有一个供应商提供，因此整理了基于仓库的数据，即把对应相同一个仓库的所有商店的同一个商品的销量进退货量分别求和

​	3、warehouses_sales_volume.DAT 是整理后的格式为 “日期  仓库  商品  销量” 的数据

​	4、warehouses_stock_return.DAT 是整理后的格式为 “日期  仓库  商品  进货量  退货量” 的数据



​	为了便于程序处理，将DAT中的数值整合成Python dict的格式保存为json

​	5、sales_inventory_stock_return.json 的数据格式是：

```python
# denote warehouse by wh in all codes  
data = {wh_id: {item_id: {store_id: {date: [sales, inventory, stock, return]}}}} 
# 即： 仓库——商品——商店——日期——销量——库存——进货——退货
# 目前的“库存=进货-退货-销量”，但实际应该由美宜佳提供数据，以后你们把这一项修改掉
```

​	6、warehouse_sales_inventory_stock_return.json 的数据格式是：

```python
data = {wh_id: {item_id: {date: [sales, inventory, stock, return]}}}
# 同理： 仓库——商品——日期——销量——库存——进货——退货
```

​	7、items_data文件夹中的每个json文件都可以看做是从sales_inventory_stock_return.json中提取出来的，即每个文件名是对应 “wh_id_item_id”，里面的数据格式是

```python
data = {store_id: {date: [sales, inventory, stock, return]}}
```

​	8、其他数据就不一一介绍了，自己看一下就懂了，都是一个套路



# 3 代码模块

# 3.1 sales_prediction

​	需要用到的代码：

​	数据处理：data_processor.py 

​	主要模块：sales_prediction.py

​	这是最基础的一个模型，也是上阶段用的模型，简而言之就是基于Gradient Boosting Machine的预测模型（只实现了订货模型，没有写配送的，两个很类似，你们接下来需要实现）



#3.2 double_actor_critic

​	需要用到的代码：

​	数据处理：data_processor.py 

​	主要模块：double_actor_critic.py，environment.py

​	这是最初想实现的模型，也可能是最后你们需要完成的模型，是基于强化学习的，目前的问题是结果不收敛，具体问题待定



# 3.3 optimal_distribution

​	需要用到的代码：

​	主函数：main_of_optimal_distribution.py

​	算法模型：optimal_distribution.py

​	这是采用凸优化的方法来做，主要就是建立目标函数（比如最低的退货量，利益最大化等），然后根据约束条件一个个补充，目前的问题是约束量太大（超过一亿）导致无法求解