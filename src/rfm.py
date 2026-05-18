import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import xgboost as xgb


class RFMAnalyzer:
    def __init__(self, df_master: pd.DataFrame):
        self.df = df_master.copy()

    def generate_rfm_segments(self) -> pd.DataFrame:
        """计算 RFM 资产并执行 K-Means 聚类打标"""
        snapshot_date = self.df['purchase_dt'].max() + pd.Timedelta(days=1)

        # 计算 R 和 F
        df_rf = self.df.groupby('customer_unique_id').agg({
            'purchase_dt': lambda x: (snapshot_date - x.max()).days,
            'order_id': 'nunique'
        }).reset_index()

        # 计算 M (去重防膨胀)
        df_m = self.df[['customer_unique_id', 'order_id', 'payment_value']].drop_duplicates()
        df_m = df_m.groupby('customer_unique_id')['payment_value'].sum().reset_index()

        df_rfm = pd.merge(df_rf, df_m, on='customer_unique_id')
        df_rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']

        # 特征工程与聚类
        rfm_log = np.log1p(df_rfm[['Recency', 'Frequency', 'Monetary']])
        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm_log)

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        # 标签映射
        cluster_names = {
            2: 'VIP Loyalists',
            3: 'Promising New',
            0: 'Sleeping Big-Spenders',
            1: 'Low-Value Churned'
        }
        df_rfm['Segment'] = df_rfm['Cluster'].map(cluster_names).fillna('Other')
        return df_rfm

    def train_churn_model(self, df_rfm: pd.DataFrame):
        """训练流失预测分类器并返回模型对象及评估指标"""
        df_rfm['is_churn'] = (df_rfm['Recency'] > 180).astype(int)

        # 提取体验特征
        df_exp = self.df.groupby('customer_unique_id').agg({
            'days_total_fulfillment': 'mean',
            'freight_value': 'mean',
            'product_weight_g': 'mean'
        }).reset_index()

        df_model = pd.merge(df_rfm, df_exp, on='customer_unique_id')

        X = df_model[['Frequency', 'Monetary', 'days_total_fulfillment', 'freight_value', 'product_weight_g']]
        y = df_model['is_churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42,
                                  eval_metric='logloss')
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

        return model, auc_score, importance