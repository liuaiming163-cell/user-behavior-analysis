import pandas as pd


class FunnelAnalyzer:
    def __init__(self, df_master: pd.DataFrame):
        """
        初始化漏斗分析器
        :param df_master: 包含订单、客户、商品明细的合并宽表 (ABT)
        """
        self.df = df_master.copy()

    def calculate_cohort_matrix(self) -> pd.DataFrame:
        """计算用户月度群组留存矩阵"""
        df_timeline = self.df[['customer_unique_id', 'purchase_month']].drop_duplicates()
        df_timeline['first_month'] = df_timeline.groupby('customer_unique_id')['purchase_month'].transform('min')

        # 计算相对月份差异
        df_timeline['cohort_index'] = (
                (df_timeline['purchase_month'].dt.year - df_timeline['first_month'].dt.year) * 12 +
                (df_timeline['purchase_month'].dt.month - df_timeline['first_month'].dt.month)
        )

        cohort_pivot = df_timeline.groupby(['first_month', 'cohort_index'])['customer_unique_id'] \
            .nunique().reset_index() \
            .pivot(index='first_month', columns='cohort_index', values='customer_unique_id') \
            .fillna(0).astype(int)

        # 转换为百分比留存率
        cohort_matrix = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0).round(4) * 100
        return cohort_matrix

    def calculate_market_basket(self) -> pd.Series:
        """计算跨品类购物篮协同矩阵"""
        df_basket = self.df[['order_id', 'product_category_name']].drop_duplicates()
        df_paired = pd.merge(df_basket, df_basket, on='order_id', suffixes=('_A', '_B'))

        # 剔除同品类自连接
        df_pairs = df_paired[df_paired['product_category_name_A'] != df_paired['product_category_name_B']]

        return df_pairs.groupby(['product_category_name_A', 'product_category_name_B']) \
            .size().sort_values(ascending=False)