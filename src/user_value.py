import pandas as pd
import numpy as np


class UserValueAnalyzer:
    """
    模块二：用户价值与行为探究 (取代原有的前端转化漏斗)
    专注于构建分析宽表(ABT)、同期群留存分析(Cohort)与购物篮跨品类挖掘(Market Basket)
    """

    def __init__(self, df_orders, df_payments, df_customers, df_items, df_products):
        self.df_orders = df_orders.copy()
        self.df_payments = df_payments.copy()
        self.df_customers = df_customers.copy()
        self.df_items = df_items.copy()
        self.df_products = df_products.copy()
        self.df_master = None

    def build_analytical_base_table(self) -> pd.DataFrame:
        """
        核心方法：拼装终极 ABT (Analytical Base Table) 宽表
        面试亮点：在此处理金额聚合，防止表连接时产生笛卡尔积导致金额膨胀
        """
        # 1. 基础时间特征提取
        self.df_orders['purchase_dt'] = pd.to_datetime(self.df_orders['order_purchase_timestamp'])
        self.df_orders['purchase_month'] = self.df_orders['purchase_dt'].dt.to_period('M')

        # 【本次修复核心：补回履约时长计算，这是模块三 XGBoost 预测流失率的关键特征】
        self.df_orders['order_delivered_customer_date'] = pd.to_datetime(
            self.df_orders['order_delivered_customer_date'])
        self.df_orders['days_total_fulfillment'] = (
                                                           self.df_orders['order_delivered_customer_date'] -
                                                           self.df_orders['purchase_dt']
                                                   ).dt.total_seconds() / 86400

        # 2. 连结地理信息
        df_geo = pd.merge(
            self.df_orders,
            self.df_customers[['customer_id', 'customer_unique_id', 'customer_state']],
            on='customer_id',
            how='inner'
        )

        # 3. 核心修复：先按 order_id 聚合总支付金额，再合入主表
        order_pay = self.df_payments.groupby('order_id')['payment_value'].sum().reset_index()
        df_geo_pay = pd.merge(df_geo, order_pay, on='order_id', how='inner')

        # 4. 连结商品属性
        df_inv = pd.merge(
            self.df_items,
            self.df_products[['product_id', 'product_category_name', 'product_weight_g']],
            on='product_id',
            how='inner'
        )

        # 5. 拼装终极宽表并缓存
        self.df_master = pd.merge(df_geo_pay, df_inv, on='order_id', how='inner')
        return self.df_master

    def calculate_cohort_matrix(self) -> pd.DataFrame:
        """业务分析 1：计算用户月度同期群留存矩阵"""
        if self.df_master is None:
            raise ValueError("请先调用 build_analytical_base_table() 生成主表数据！")

        df_timeline = self.df_master[['customer_unique_id', 'purchase_month']].drop_duplicates()
        df_timeline['first_month'] = df_timeline.groupby('customer_unique_id')['purchase_month'].transform('min')

        df_timeline['cohort_index'] = (
                (df_timeline['purchase_month'].dt.year - df_timeline['first_month'].dt.year) * 12 +
                (df_timeline['purchase_month'].dt.month - df_timeline['first_month'].dt.month)
        )

        cohort_pivot = df_timeline.groupby(['first_month', 'cohort_index'])['customer_unique_id'] \
            .nunique().reset_index() \
            .pivot(index='first_month', columns='cohort_index', values='customer_unique_id') \
            .fillna(0).astype(int)

        cohort_matrix = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0).round(4) * 100
        return cohort_matrix

    def calculate_market_basket(self) -> pd.Series:
        """业务分析 2：计算跨品类购物篮协同矩阵"""
        if self.df_master is None:
            raise ValueError("请先调用 build_analytical_base_table() 生成主表数据！")

        df_basket = self.df_master[['order_id', 'product_category_name']].drop_duplicates()
        df_paired = pd.merge(df_basket, df_basket, on='order_id', suffixes=('_A', '_B'))
        df_pairs = df_paired[df_paired['product_category_name_A'] != df_paired['product_category_name_B']]

        basket_rules = df_pairs.groupby(['product_category_name_A', 'product_category_name_B']) \
            .size().sort_values(ascending=False)

        return basket_rules