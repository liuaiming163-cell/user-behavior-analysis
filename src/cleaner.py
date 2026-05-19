import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ==========================================
# 1. 数据契约定义 (Data Schema / Contracts)
# 面试亮点：体现数据治理思维，将业务约束前置，与清洗执行逻辑分离
# ==========================================

class PaymentSchema(BaseModel):
    """支付数据契约：约束数值边界与枚举值"""
    order_id: str
    payment_sequential: int = Field(ge=1, description="支付顺序至少为1")
    payment_installments: int = Field(ge=1, description="分期数至少为1")
    payment_value: float = Field(gt=0, description="支付金额必须大于0")
    payment_type: str

    @field_validator('payment_type')
    def type_must_be_valid(cls, v):
        valid_types = ['credit_card', 'boleto', 'voucher', 'debit_card']
        if v not in valid_types:
            raise ValueError(f"出现未知/无效的支付类型: {v}")
        return v


class OrderSchema(BaseModel):
    """订单数据契约：基础结构定义"""
    order_id: str
    customer_id: str
    order_status: str
    # 实际工程中可在此处继续添加时间的业务合法性校验


# ==========================================
# 2. 数据质量报告监控类 (Quality Report)
# 面试亮点：用量化指标证明你对"清洗掉了什么数据"了如指掌
# ==========================================

class QualityReport:
    """记录每一步数据清洗的投入产出，生成血统追踪表"""

    def __init__(self):
        self.metrics = []

    def add_metric(self, step_name: str, raw_count: int, clean_count: int):
        dropped = raw_count - clean_count
        rate = f"{(dropped / raw_count) * 100:.2f}%" if raw_count > 0 else "0.00%"
        self.metrics.append({
            "清洗步骤/检查项": step_name,
            "输入记录数": raw_count,
            "淘汰记录数": dropped,
            "淘汰率": rate,
            "输出记录数": clean_count
        })

    def get_lineage_df(self) -> pd.DataFrame:
        """返回 DataFrame 格式的质量报告，方便在 Notebook 渲染和 Dashboard 展示"""
        return pd.DataFrame(self.metrics)


# ==========================================
# 3. 核心清洗管道 (Cleaner Pipeline)
# 面试亮点：职责分离，结构清晰，利用 Pandas 向量化保证处理十万级数据的性能
# ==========================================

class OlistDataCleaner:
    def __init__(self, df_orders: pd.DataFrame, df_payments: pd.DataFrame, df_items: pd.DataFrame):
        self.df_orders = df_orders.copy()
        self.df_payments = df_payments.copy()
        self.df_items = df_items.copy()
        self.report = QualityReport()

    def _format_datatypes(self):
        """类型转换：统一处理时间戳"""
        time_cols = [
            'order_purchase_timestamp', 'order_approved_at',
            'order_delivered_carrier_date', 'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
        for col in time_cols:
            self.df_orders[col] = pd.to_datetime(self.df_orders[col])

    def _apply_schema_validation(self):
        """执行 Schema 约束（实际处理时为了性能，将 Pydantic 逻辑映射为 Pandas 向量化过滤）"""
        initial_count = len(self.df_payments)

        valid_types = ['credit_card', 'boleto', 'voucher', 'debit_card']
        mask = (
                (self.df_payments['payment_installments'] >= 1) &
                (self.df_payments['payment_value'] > 0) &
                (self.df_payments['payment_type'].isin(valid_types))
        )
        self.df_payments = self.df_payments[mask]

        self.report.add_metric("Schema校验: 剔除越界与未知枚举值的支付数据", initial_count, len(self.df_payments))

    def _check_timeline_anomalies(self):
        """逻辑检测：检查物流状态与时间戳序列的合理性"""
        initial_count = len(self.df_orders)

        # 1. 状态倒置/缺失
        c1 = (self.df_orders['order_status'] == 'delivered') & (
            self.df_orders['order_delivered_customer_date'].isnull())
        c2 = (self.df_orders['order_status'] != 'delivered') & (
            self.df_orders['order_delivered_customer_date'].notnull())

        # 2. 时间线穿越 (比如：送到客户手上早于发货)
        c3 = self.df_orders['order_delivered_customer_date'] < self.df_orders['order_delivered_carrier_date']
        c4 = self.df_orders['order_delivered_carrier_date'] < self.df_orders['order_approved_at']

        # 3. 极值检测 (物流时间长得离谱，> 180天)
        delivery_days = (self.df_orders['order_delivered_customer_date'] - self.df_orders[
            'order_purchase_timestamp']).dt.days
        c5 = delivery_days > 180

        anomalies_mask = c1 | c2 | c3 | c4 | c5
        self.df_orders = self.df_orders[~anomalies_mask.fillna(False)]

        self.report.add_metric("业务扫描: 剔除状态倒置与时间线穿越订单", initial_count, len(self.df_orders))

    def _check_financial_discrepancy(self):
        """对账检测：订单包含的商品及运费总金额，必须等于用户的实际支付金额"""
        initial_count = len(self.df_orders)

        # 计算每单商品总价+运费
        item_total = self.df_items.groupby('order_id')[['price', 'freight_value']].sum().sum(axis=1)
        # 计算每单实际支付汇总
        pay_total = self.df_payments.groupby('order_id')['payment_value'].sum()

        financial_check = pd.merge(item_total.to_frame('item_amt'), pay_total.to_frame('pay_amt'), on='order_id')

        # 容忍浮点数运算带来的 0.05 精度误差
        financial_anomalies = financial_check[np.abs(financial_check['item_amt'] - financial_check['pay_amt']) > 0.05]

        # 从订单和支付表中同步剔除这批账目不平的异常订单
        valid_order_ids = set(self.df_orders['order_id']).difference(financial_anomalies.index)
        self.df_orders = self.df_orders[self.df_orders['order_id'].isin(valid_order_ids)]
        self.df_payments = self.df_payments[self.df_payments['order_id'].isin(valid_order_ids)]

        self.report.add_metric("对账扫描: 剔除商品总金额与支付金额不符订单", initial_count, len(self.df_orders))

    def run_pipeline(self) -> tuple:
        """执行流：按次序调用清洗规则，返回清洗后的 DF 以及质量报告"""
        self._format_datatypes()
        self._apply_schema_validation()
        self._check_timeline_anomalies()
        self._check_financial_discrepancy()

        return self.df_orders, self.df_payments, self.report