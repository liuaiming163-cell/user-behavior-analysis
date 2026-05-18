import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Visualizer:
    @staticmethod
    def plot_cohort_heatmap(cohort_matrix: pd.DataFrame):
        """绘制用户群组留存热力图"""
        fig, ax = plt.subplots(figsize=(12, 6))
        # 截取主要运营时间段
        cohort_summary = cohort_matrix.iloc[3:10, 1:13]
        sns.heatmap(cohort_summary, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax)

        ax.set_title("User Cohort Retention Rate %", fontsize=14, pad=15)
        ax.set_xlabel("Months After First Purchase (Cohort Index)", fontsize=12)
        ax.set_ylabel("First Purchase Cohort", fontsize=12)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_rfm_revenue_share(df_rfm: pd.DataFrame):
        """绘制 RFM 用户分群人数与营收贡献对比双饼图"""
        segment_summary = df_rfm.groupby('Segment').agg({
            'customer_unique_id': 'count',
            'Monetary': 'sum'
        }).rename(columns={'customer_unique_id': 'Total_Users', 'Monetary': 'Total_Revenue'})

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        colors = sns.color_palette("Set2")

        axes[0].pie(segment_summary['Total_Users'], labels=segment_summary.index, autopct='%1.1f%%',
                    startangle=90, colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        axes[0].set_title("User Population Share", fontsize=14, fontweight='bold')

        axes[1].pie(segment_summary['Total_Revenue'], labels=segment_summary.index, autopct='%1.1f%%',
                    startangle=90, colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
        axes[1].set_title("Total Revenue Contribution Share", fontsize=14, fontweight='bold')

        plt.tight_layout()
        return fig