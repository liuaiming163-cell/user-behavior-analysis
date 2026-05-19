Olist E-Commerce Data Intelligence Pipeline
基于 Olist 真实交易数据集的全链路数据分析与算法建模项目

项目概述
本项目基于巴西电商平台 Olist 提供的公开交易数据集，构建了端到端的数据科学管线。项目内容涵盖基于 Schema 的数据质量检验、多表合并防膨胀处理、同期群留存分析、商品关联规则挖掘，以及基于无监督聚类与监督分类的机器学习建模，最终输出支持交互的可视化网页看板。核心目标是通过量化的数据工程手段，揭示用户生命周期特征与流失归因。

技术栈与核心方法论
数据治理与数据工程：Pandas, Pydantic (数据契约校验), Parquet 列式存储, 规避笛卡尔积膨胀

用户行为分析：同期群留存分析 (Cohort Analysis), 购物篮协同矩阵 (Market Basket Analysis)

无监督机器学习：K-Means 聚类, 肘部法则 (Elbow Method), 轮廓系数 (Silhouette Score), 特征对数变换 (Log1p Transformation), StandardScaler, 动态聚类打标

监督机器学习：XGBoost 分类器, 规避目标泄露 (Target Leakage), 分层交叉切分, ROC-AUC 评估, 树模型特征重要性 (Feature Importance)

可视化与商业智能：Plotly, 基于 HTML/CSS 的交互式前端大屏渲染

核心业务模块拆解
模块一：数据清洗与质量追踪 (Data Cleaning & Quality Assessment)
构建 Pydantic 契约定义字段边界，拦截无效枚举值数据。执行系统级的业务逻辑检测（如物流时间线倒置、极值耗时）与财务对账检验（订单包含商品总金额与实际支付总金额比对）。通过封装 QualityReport 类，记录各清洗阶段的数据淘汰率，输出量化的数据血统追踪表。

模块二：分析宽表构建与用户行为洞察 (ABT & Behavior Analysis)
处理一单多件商品与一单多笔支付的对应关系，采用先聚合后连结的处理策略，规避因多对多关联产生的数据与金额膨胀，构建底层分析宽表 (Analytical Base Table)。基于此宽表计算绝对时间差，输出用户月度同期群留存矩阵，并提取同订单内的商品联合频数，输出跨品类交叉搭售规则。

模块三：动态 RFM 聚类与流失预警 (RFM Clustering & Churn Prediction)
提取独立用户的近度 (Recency)、频度 (Frequency) 与额度 (Monetary) 特征并进行抗偏态处理。引入降采样与时间熔断机制评估算法 K 值，基于评价指标选定 K=4 执行无监督聚类，并根据实际特征均值进行动态标签映射，避免硬编码。同时引入总履约时长与运费作为物流体验变量，训练 XGBoost 树模型，预测高价值用户流失概率并提取关键影响因子。

模块四：交互式商业智能看板 (Interactive BI Dashboard)
使用 Plotly 框架对分析结果进行可视化封装。将数据质量检查漏斗、留存热力图、RFM 群体画像雷达图与全局核心业务指标 (KPI) 进行逻辑整合，通过 Python 脚本直接生成并组装为支持数值悬停与视图缩放的独立 HTML 网页看板。

核心商业洞察结论
留存特征与运营方向：同期群留存矩阵显示，用户次月留存率整体低于 1%。数据表明该平台当前的用户交易行为具有单次博弈特征，长期复购粘性较弱。基于此数据表现，建议运营资源适度向提高单次客单价及关联商品交叉销售倾斜。

用户资产结构：RFM 聚类模型识别出占比约 20% 的核心高价值用户群体，该群体贡献了平台主要的营收份额，印证了帕累托法则在当前业务中的适用性。

流失核心归因：XGBoost 模型（测试集 ROC-AUC 达到 0.88）特征重要性分析表明，订单总履约时长（days_total_fulfillment）是对用户流失预测贡献度最高的自变量，其次为运费。缩短干线物流时间是降低高价值用户流失率、提升体验的核心优化方向。

物理目录结构
Plaintext
├── data/
│   ├── raw/                            # 原始 CSV 数据集目录（不进入版本控制）
│   └── processed/                      # 清洗后生成的 Parquet 列式存储文件
├── notebooks/
│   ├── 01_data_cleaning.ipynb          # 模块一：数据清理与质量核查
│   ├── 02_user_value_analysis.ipynb    # 模块二：宽表构建、同期群与购物篮分析
│   ├── 03_rfm_segmentation.ipynb       # 模块三：RFM 聚类评估与 XGBoost 建模
│   └── 04_dashboard.ipynb              # 模块四：HTML 交互式大屏看板生成
├── src/
│   ├── cleaner.py                      # 数据校验与清洗引擎类
│   ├── user_value.py                   # 宽表聚合与行为计算类
│   ├── rfm.py                          # 聚类评估、动态打标与分类器类
│   └── visualizer.py                   # 可视化图表生成与 HTML 渲染类
├── requirements.txt                    # Python 运行环境依赖清单
└── README.md                           # 项目说明文档