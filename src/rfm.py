import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


class RFMAnalyzer:
    def __init__(self, df_master: pd.DataFrame):
        self.df = df_master.copy()
        self.scaler = StandardScaler()
        self.kmeans_model = None

    def evaluate_k_choices(self, max_k: int = 8, sample_size: int = 10000, time_limit_sec: int = 60) -> dict:
        """
        评估不同的 K 值，引入降采样和超时熔断机制以优化算力消耗。

        :param max_k: 最大评估簇数
        :param sample_size: 轮廓系数降采样规模
        :param time_limit_sec: 最大允许计算时间（秒），超时则熔断
        :return: 包含 k, inertia, silhouette 评估指标的字典
        """
        df_rfm_raw = self._calculate_base_rfm()
        rfm_scaled = self._preprocess_features(df_rfm_raw)

        evaluation = {'k': [], 'inertia': [], 'silhouette': []}
        start_time = time.time()

        # 确保采样量不超过实际数据量
        n_samples = min(len(rfm_scaled), sample_size)

        for k in range(2, max_k + 1):
            # 熔断机制检查
            if time.time() - start_time > time_limit_sec:
                print(f"⚠️ 达到计算时间上限 ({time_limit_sec}s)，熔断机制触发，停止评估。当前已完成 K={k - 1}")
                break

            # 探索性评估阶段使用 n_init='auto' 加速计算
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(rfm_scaled)

            evaluation['k'].append(k)
            evaluation['inertia'].append(kmeans.inertia_)

            # 使用采样数据计算轮廓系数，极大降低 O(N^2) 的算力消耗
            score = silhouette_score(rfm_scaled, labels, sample_size=n_samples, random_state=42)
            evaluation['silhouette'].append(score)

        return evaluation

    def _calculate_base_rfm(self) -> pd.DataFrame:
        snapshot_date = self.df['purchase_dt'].max() + pd.Timedelta(days=1)

        df_rf = self.df.groupby('customer_unique_id').agg({
            'purchase_dt': lambda x: (snapshot_date - x.max()).days,
            'order_id': 'nunique'
        }).reset_index()

        df_m = self.df[['customer_unique_id', 'order_id', 'payment_value']].drop_duplicates()
        df_m = df_m.groupby('customer_unique_id')['payment_value'].sum().reset_index()

        df_rfm = pd.merge(df_rf, df_m, on='customer_unique_id')
        df_rfm.columns = ['customer_unique_id', 'Recency', 'Frequency', 'Monetary']
        return df_rfm

    def _preprocess_features(self, df_rfm: pd.DataFrame) -> np.ndarray:
        rfm_log = np.log1p(df_rfm[['Recency', 'Frequency', 'Monetary']])
        return self.scaler.fit_transform(rfm_log)

    def _dynamic_labeling(self, df_rfm: pd.DataFrame) -> pd.DataFrame:
        cluster_means = df_rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()

        labels = {}
        for cluster_id in cluster_means.index:
            r = cluster_means.loc[cluster_id, 'Recency']
            f = cluster_means.loc[cluster_id, 'Frequency']
            m = cluster_means.loc[cluster_id, 'Monetary']

            if m > cluster_means['Monetary'].median() and f > cluster_means['Frequency'].median():
                labels[cluster_id] = 'VIP Loyalists'
            elif r > cluster_means['Recency'].median() and m > cluster_means['Monetary'].median():
                labels[cluster_id] = 'Sleeping Big-Spenders'
            elif r < cluster_means['Recency'].median() and f <= cluster_means['Frequency'].median():
                labels[cluster_id] = 'Promising New'
            else:
                labels[cluster_id] = 'Low-Value Churned'

        df_rfm['Segment'] = df_rfm['Cluster'].map(labels)
        return df_rfm

    def generate_rfm_segments(self) -> pd.DataFrame:
        df_rfm = self._calculate_base_rfm()
        rfm_scaled = self._preprocess_features(df_rfm)

        # 正式生成分类标签时，恢复严格的初始化策略 (n_init=10) 保证稳定性
        self.kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_rfm['Cluster'] = self.kmeans_model.fit_predict(rfm_scaled)

        df_rfm = self._dynamic_labeling(df_rfm)
        return df_rfm

    def train_churn_model(self, df_rfm: pd.DataFrame):
        df_rfm['is_churn'] = (df_rfm['Recency'] > 180).astype(int)

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