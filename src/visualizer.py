import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class DashboardVisualizer:
    """
    模块四：最终交付物 Dashboard 生成器
    利用 Plotly 生成高交互图表，并拼装为带 KPI 卡片的 HTML 网页
    """

    @staticmethod
    def plot_data_quality(lineage_df: pd.DataFrame) -> str:
        """绘制模块一的数据质量漏斗条形图"""
        fig = px.bar(
            lineage_df,
            y="清洗步骤/检查项",
            x="淘汰记录数",
            orientation='h',
            title="Data Quality Scans: Dropped Records",
            color="淘汰记录数",
            color_continuous_scale="Reds"
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def plot_cohort_heatmap(cohort_matrix: pd.DataFrame) -> str:
        """绘制模块二的留存热力图 (Plotly 交互版)"""
        # 为了展示美观，截取前 12 个月，使用 .copy() 避免修改原始数据
        plot_data = cohort_matrix.iloc[:, :12].copy()

        # 🔧 核心修复：将 Pandas 的 Period 对象强制转为字符串，解决 JSON 序列化报错
        plot_data.index = plot_data.index.astype(str)
        plot_data.columns = plot_data.columns.astype(str)

        fig = px.imshow(
            plot_data,
            text_auto=".1f",
            aspect="auto",
            color_continuous_scale="YlGnBu",
            title="User Cohort Retention Rate (%)",
            labels=dict(x="Months After First Purchase", y="First Purchase Cohort", color="Retention %")
        )
        # 将 Y 轴时间格式化为字符串，避免被当作连续数值
        fig.update_yaxes(type='category')
        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def plot_rfm_radar(df_rfm: pd.DataFrame) -> str:
        """绘制模块三的 RFM 群体画像雷达图"""
        # 计算每个群体的 RFM 均值
        segment_stats = df_rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()

        # 数据归一化，使得雷达图在一个量级上展示
        for col in ['Recency', 'Frequency', 'Monetary']:
            max_val = segment_stats[col].max()
            segment_stats[f'{col}_norm'] = segment_stats[col] / max_val if max_val > 0 else 0

        fig = go.Figure()
        categories = ['Recency', 'Frequency', 'Monetary']

        for _, row in segment_stats.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Recency_norm'], row['Frequency_norm'], row['Monetary_norm']],
                theta=categories,
                fill='toself',
                name=row['Segment']
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="RFM Segment Profiles (Normalized)",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        return fig.to_html(full_html=False, include_plotlyjs=False)

    @staticmethod
    def build_dashboard(html_path: str, kpi_data: dict, plots_html: dict):
        """
        组装终极仪表盘：将 KPI 卡片与图表 HTML 组合
        """
        # 手写一段极简的现代 Dashboard CSS 布局
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Olist E-Commerce Analytics Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f7f6; margin: 0; padding: 20px; }}
                h1 {{ text-align: center; color: #333; }}
                .kpi-container {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .kpi-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 22%; text-align: center; }}
                .kpi-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
                .kpi-title {{ font-size: 14px; color: #7f8c8d; margin-top: 5px; }}
                .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
                .chart-card {{ background: white; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); padding: 10px; }}
                .full-width {{ grid-column: 1 / -1; }}
            </style>
        </head>
        <body>
            <h1>Olist E-Commerce Analytics Dashboard</h1>

            <div class="kpi-container">
                <div class="kpi-card"><div class="kpi-value">{kpi_data.get('total_users', 0):,}</div><div class="kpi-title">Total Valid Users</div></div>
                <div class="kpi-card"><div class="kpi-value">${kpi_data.get('total_revenue', 0):,.0f}</div><div class="kpi-title">Total Revenue</div></div>
                <div class="kpi-card"><div class="kpi-value">{kpi_data.get('vip_ratio', '0%')}</div><div class="kpi-title">VIP User Ratio</div></div>
                <div class="kpi-card"><div class="kpi-value">{kpi_data.get('data_quality_score', '100%')}</div><div class="kpi-title">Data Quality Pass Rate</div></div>
            </div>

            <div class="grid-container">
                <div class="chart-card full-width">
                    {plots_html.get('cohort_heatmap', '')}
                </div>
                <div class="chart-card">
                    {plots_html.get('rfm_radar', '')}
                </div>
                <div class="chart-card">
                    {plots_html.get('data_quality', '')}
                </div>
            </div>
        </body>
        </html>
        """

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"✅ Dashboard successfully generated at: {html_path}")