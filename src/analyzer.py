import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


class OlistBIEngine:
    def __init__(self, df_orders, df_payments, df_customers, df_items, df_products):
        self.df_orders = df_orders.copy()
        self.df_payments = df_payments.copy()
        self.df_customers = df_customers.copy()
        self.df_items = df_items.copy()
        self.df_products = df_products.copy()
        self.df_master = None

    def build_analytical_base_table(self):
        self.df_orders['purchase_dt'] = pd.to_datetime(self.df_orders['order_purchase_timestamp'])
        self.df_orders['purchase_month'] = self.df_orders['purchase_dt'].dt.to_period('M')
        self.df_orders['purchase_hour'] = self.df_orders['purchase_dt'].dt.hour
        self.df_orders['purchase_dow'] = self.df_orders['purchase_dt'].dt.dayofweek
        self.df_orders['days_total_fulfillment'] = (pd.to_datetime(self.df_orders['order_delivered_customer_date']) -
                                                    self.df_orders['purchase_dt']).dt.total_seconds() / 86400

        # 1. 连结地理信息
        df_geo = pd.merge(self.df_orders, self.df_customers[['customer_id', 'customer_unique_id', 'customer_state']],
                          on='customer_id', how='inner')

        # 2. 核心修复：先按 order_id 聚合总支付金额，再合入主表，防止后续发生金额膨胀
        order_pay = self.df_payments.groupby('order_id')['payment_value'].sum().reset_index()
        df_geo_pay = pd.merge(df_geo, order_pay, on='order_id', how='inner')

        # 3. 连结商品属性
        df_inv = pd.merge(self.df_items, self.df_products[['product_id', 'product_category_name', 'product_weight_g']],
                          on='product_id', how='inner')

        # 4. 拼装终极 ABT 宽表
        self.df_master = pd.merge(df_geo_pay, df_inv, on='order_id', how='inner')
        return self.df_master

    def calculate_market_basket(self):
        df_basket = self.df_master[['order_id', 'product_category_name']].drop_duplicates()
        df_paired = pd.merge(df_basket, df_basket, on='order_id', suffixes=('_A', '_B'))
        df_pairs = df_paired[df_paired['product_category_name_A'] != df_paired['product_category_name_B']]
        return df_pairs.groupby(['product_category_name_A', 'product_category_name_B']).size().sort_values(
            ascending=False)

    def calculate_cohort_matrix(self):
        df_timeline = self.df_master[['customer_unique_id', 'purchase_month']].drop_duplicates()
        df_timeline['first_month'] = df_timeline.groupby('customer_unique_id')['purchase_month'].transform('min')
        df_timeline['cohort_index'] = (df_timeline['purchase_month'].dt.year - df_timeline[
            'first_month'].dt.year) * 12 + (df_timeline['purchase_month'].dt.month - df_timeline[
            'first_month'].dt.month)

        cohort_pivot = df_timeline.groupby(['first_month', 'cohort_index'])[
            'customer_unique_id'].nunique().reset_index().pivot(index='first_month', columns='cohort_index',
                                                                values='customer_unique_id').fillna(0).astype(int)
        return cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0).round(4) * 100

    def extract_ml_feature_importance(self):
        df_ml = self.df_master[
            ['days_total_fulfillment', 'price', 'freight_value', 'product_weight_g', 'purchase_hour', 'purchase_dow',
             'customer_state']].dropna()
        df_encoded = pd.get_dummies(df_ml, columns=['customer_state'], drop_first=True)

        X = df_encoded.drop(columns=['days_total_fulfillment'])
        y = df_encoded['days_total_fulfillment']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    def generate_rfm_segments(self):
        """
        端到端计算用户 RFM 资产，并使用 K-Means 进行无监督聚类分群
        """
        if self.df_master is None:
            raise ValueError("请先运行 build_analytical_base_table() 生成主表！")

        # 1. 提取 RFM 基础资产 (已包含 M 值的防笛卡尔积去重机制)
        snapshot_date = self.df_master['purchase_dt'].max() + pd.Timedelta(days=1)

        df_rf = self.df_master.groupby('customer_unique_id').agg({
            'purchase_dt': lambda x: (snapshot_date - x.max()).days,
            'order_id': 'nunique'
        }).reset_index()

        df_m = self.df_master[['customer_unique_id', 'order_id', 'payment_value']].drop_duplicates()
        df_m = df_m.groupby('customer_unique_id')['payment_value'].sum().reset_index()

        df_rfm = pd.merge(df_rf, df_m, on='customer_unique_id')
        df_rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

        # 2. 特征工程：抗偏态对数压缩与标准化
        rfm_log = np.log1p(df_rfm[['Recency', 'Frequency', 'Monetary']])
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)

        # 3. K-Means 机器学习聚类
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        # 4. 商业标签映射 (基于我们之前测试的 42 号随机种子输出)
        cluster_names = {
            2: 'VIP Loyalists',
            3: 'Promising New',
            0: 'Sleeping Big-Spenders',
            1: 'Low-Value Churned'
        }
        df_rfm['Segment'] = df_rfm['Cluster'].map(cluster_names).fillna('Other')

        return df_rfm