# E-Commerce Data Intelligence Pipeline: From EDA to Predictive Modeling
**基于 Olist 真实交易数据集的全链路电商数据科学与算法建模项目**

## 📖 项目概述 (Project Overview)
本项目基于巴西头部电商平台 Olist 提供的 10 万+ 真实脱敏交易数据集，构建了端到端的数据分析与机器学习管线（Pipeline）。项目旨在解决电商平台普遍面临的用户留存率衰减、干线物流卡点以及长尾用户运营成本过高等业务痛点。

通过多模块的数据工程设计，本项目实现了从基础的**描述性统计（Descriptive Statistics）**，到**多变量因果推断（Multivariate Causal Inference）**，最终落地于**无监督聚类（Unsupervised Clustering）**与**监督分类模型（Supervised Classification）**的完整商业智能（BI）闭环。

---

## 🛠️ 技术栈与长尾知识点 (Tech Stack & Domain Expertise)
* **数据工程与清洗**：Pandas, Parquet 列式存储优化, 异常值截断 (Winsorization), 规避笛卡尔积膨胀 (Cartesian Product Expansion)
* **统计学与推断**：Spearman 秩相关系数, 独立样本 t 检验 (Welch's t-test), 卡方独立性检验 (Chi-square test), 统计显著性 (p-value)
* **无监督机器学习**：K-Means 聚类, 对数变换平滑 (Log1p Transformation), 特征标准化 (StandardScaler), RFM 资产价值模型
* **监督机器学习**：XGBoost 分类器, 规避数据穿越/目标泄露 (Target Leakage), 交叉验证, ROC-AUC 评估, 树模型特征重要性 (Feature Importance)
* **商业智能与可视化**：时序同群组分析 (Cohort Analysis), 关联规则协同矩阵 (Co-occurrence Matrix), Plotly 交互式前端看板

---

## 🏗️ 核心业务模块拆解 (Modular Architecture)

### 模块一：探索性数据分析与多变量推断 (EDA & Statistical Inference)
本模块主要负责大盘数据的质量核查与供应链时效的降维诊断。
* **单变量分析 (Univariate Analysis)**：运用直方图与核密度估计 (KDE) 探明 `payment_value` (客单价) 的长尾偏态分布特征；通过频数统计界定各物流节点的时间绝对跨度。
* **双变量分析 (Bivariate Analysis)**：
  * **连续型 vs 连续型**：运用 Spearman 相关系数验证“订单金额”与“分期期数”的正相关单调性。
  * **离散型 vs 连续型**：运用异方差 t 检验测算“信用卡”与“现金汇票 (Boleto)”用户客单价均值的统计学差异。
* **多变量分层分析 (Multivariate Stratification)**：引入地理栅格 (Geo-spatial) 与时间截面作为控制变量，构建 7x24 小时热力矩阵，剥离混淆变量，量化圣保罗 (SP) 与亚马逊偏远州 (AM) 之间的物流时效代沟。

### 模块二：高维特征工程与同群组留存阵列 (Feature Engineering & Cohort Matrix)
* **关联规则协同矩阵 (Market Basket Co-occurrence)**：基于单笔订单内的商品明细 (Items)，计算跨品类 SKU 的联合频数，输出支持交叉销售 (Cross-selling) 的高频搭档组合。
* **时序留存衰减模型 (Cohort Retention Analysis)**：以用户首次交易时间为锚点 (Anchor Date)，计算生命周期内的绝对月份差 (Cohort Index)，构建并可视化用户月度留存率衰减矩阵，客观评估平台的长期获客转化健康度。

### 模块三：基于无监督学习的用户资产分群 (Unsupervised Clustering: RFM Segmentation)
* **特征构造与预处理**：提取每位独立访客的 Recency (近度)、Frequency (频度)、Monetary (额度) 原始特征。应用 `np.log1p` 处理长尾偏态，并利用 `StandardScaler` 消除量纲维度差异（Scale Mismatch）。
* **K-Means 迭代聚类**：设定超参数 `n_clusters=4`，将 9 万量级用户划分为四个异质性群体。输出各簇的质心 (Centroids) 参数，从数学层面实证“帕累托法则 (Pareto Principle)”——锁定贡献平台 50% 以上营收的 3% 核心金主。

### 模块四：规避数据穿越的流失预警分类器 (Supervised Learning: Churn Prediction)
* **防泄露特征工程**：在构建训练集时，严格剔除 `Recency` 等直接包含目标变量信息（是否超过 180 天未复购）的穿越特征，引入 `days_total_fulfillment` (总履约时长) 与 `freight_value` (运费) 作为用户体验代理变量。
* **XGBoost 建模与归因**：训练极端梯度提升树 (Extreme Gradient Boosting) 分类器。模型在测试集上的 ROC-AUC 达到 `0.8826`。通过提取 `feature_importances_`，客观证实“干线物流延误”与“高昂运费”为导致新客流失的首要负向驱动因子。

---

## 📊 核心商业洞察结论 (Key Business Insights)

1. **供应链物理卡点界定**：系统审批与仓储打包环节的均值耗时均小于 2 天，而干线运输阶段耗时占比达 74%（均值 9.33 天），确认为全链路最大效率瓶颈。
2. **留存断崖与单次博弈机制**：Cohort 矩阵显示，次月留存率呈断崖式下跌（不足 1%），证实平台当前处于“高成本拉新、低粘性留存”的单次博弈（One-off Transaction）状态。
3. **沉睡客群唤醒价值**：聚类算法识别出占比 34.18% 的“单次大额沉睡客群 (Sleeping Big-Spenders)”，其历史客单价均值为全局均值的 2.7 倍，应作为后续定向触达 (Retargeting) 的第一优选序列。

---

## 📁 物理目录结构 (Repository Structure)

```text
├── data/
│   ├── raw/                 # 原始 CSV 数据（因体积限制未上传，请参阅 data_dictionary.md）
│   └── processed/           # 经过 cleaner.py 处理的纯净 Parquet 数据列式存储文件
├── notebooks/
│   ├── 01_data_cleaning.ipynb        # 异常值处理与 ETL 预实验
│   ├── 02_user_value_analysis.ipynb  # 宽表构建、同期群与购物篮分析
│   ├── 03_rfm_segmentation.ipynb     # 无监督聚类与 XGBoost 分类器实验
│   └── 04_dashboard.ipynb            # 基于 Plotly 的交互式前端 BI 看板
├── src/
│   ├── cleaner.py           # 数据预处理模块
│   ├── user_value.py        # 同群组与关联矩阵计算模块
│   ├── rfm.py               # 聚类与预测建模封装模块
│   └── visualizer.py        # 绘图渲染与视图隔离模块
├── requirements.txt         # 依赖项清单
└── README.md                # 项目架构文档