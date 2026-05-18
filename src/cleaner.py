import pandas as pd
import numpy as np


class OlistDataCleaner:
    def __init__(self, df_orders, df_payments, df_items):
        self.df_orders = df_orders.copy()
        self.df_payments = df_payments.copy()
        self.df_items = df_items.copy()
        self.report = {}

    def run_pipeline(self):
        self.report['raw_orders_count'] = len(self.df_orders)
        self.report['raw_payments_count'] = len(self.df_payments)

        time_cols = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]
        for col in time_cols:
            self.df_orders[col] = pd.to_datetime(self.df_orders[col])

        self.df_payments = self.df_payments[
            (self.df_payments['payment_installments'] >= 1) &
            (self.df_payments['payment_value'] > 0) &
            (self.df_payments['payment_type'] != 'not_defined')
            ]

        c1 = (self.df_orders['order_status'] == 'delivered') & (
            self.df_orders['order_delivered_customer_date'].isnull())
        c2 = (self.df_orders['order_status'] != 'delivered') & (
            self.df_orders['order_delivered_customer_date'].notnull())
        c3 = self.df_orders['order_delivered_customer_date'] < self.df_orders['order_delivered_carrier_date']
        c4 = self.df_orders['order_delivered_carrier_date'] < self.df_orders['order_approved_at']

        delivery_days = (self.df_orders['order_delivered_customer_date'] - self.df_orders[
            'order_purchase_timestamp']).dt.days
        c5 = delivery_days > 180

        anomalies_mask = c1 | c2 | c3 | c4 | c5

        self.report['status_delivered_missing_date'] = int(c1.sum())
        self.report['status_not_delivered_has_date'] = int(c2.sum())
        self.report['timeline_reverse_carrier_customer'] = int(c3.sum())
        self.report['timeline_reverse_approved_carrier'] = int(c4.sum())
        self.report['outlier_delivery_duration'] = int(c5.fillna(False).sum())

        self.df_orders = self.df_orders[~anomalies_mask.fillna(False)]

        item_total = self.df_items.groupby('order_id')[['price', 'freight_value']].sum().sum(axis=1)
        pay_total = self.df_payments.groupby('order_id')['payment_value'].sum()

        financial_check = pd.merge(item_total.to_frame('item_amt'), pay_total.to_frame('pay_amt'), on='order_id')
        financial_anomalies = financial_check[np.abs(financial_check['item_amt'] - financial_check['pay_amt']) > 0.05]

        self.report['financial_amount_discrepancy'] = len(financial_anomalies)

        valid_order_ids = set(self.df_orders['order_id']).difference(financial_anomalies.index)

        self.df_orders = self.df_orders[self.df_orders['order_id'].isin(valid_order_ids)]
        self.df_payments = self.df_payments[self.df_payments['order_id'].isin(valid_order_ids)]

        self.report['clean_orders_count'] = len(self.df_orders)
        self.report['clean_payments_count'] = len(self.df_payments)

        return self.df_orders, self.df_payments, self.report