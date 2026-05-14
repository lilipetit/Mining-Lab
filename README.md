# Mining-Lab 数据挖掘算法实验平台

一个面向《数据仓库与数据挖掘》课程的交互式 Web 实验平台，集成了经典数据挖掘算法的可视化实验和 AI 智能导师功能。

## 🌟 功能特性

### 算法实验模块
- **K-Means 聚类** - 二维/三维数据聚类、肘部法则
- **决策树分类** - 树结构可视化、特征重要性分析
- **朴素贝叶斯** - Gaussian/MultinomialNB、分类边界可视化
- **Apriori 关联规则** - 频繁项集挖掘、规则热力图
- **KNN 分类** - 决策边界、交叉验证曲线
- **PCA 降维** - 主成分分析、维度对比可视化
- **回归分析** - 线性回归、逻辑回归、残差分析

### AI 智能助手
- Coze API 集成，24小时在线答疑
- 算法原理讲解、参数调优建议
- 实验结果解读、编程指导

### 数据管理
- 内置经典数据集（鸢尾花、酒馆数据、购物篮等）
- CSV 数据上传
- 数据预览与统计描述

### 实验报告
- 一键生成 Markdown 格式报告
- 包含完整参数、数据、图表、分析结论

## 🚀 快速开始

### 本地运行

```bash
# 克隆项目
git clone https://github.com/yourusername/Mining-Lab.git
cd Mining-Lab

# 安装依赖
pip install -r requirements.txt

# 启动应用
streamlit run app.py
```

### 云端部署

支持一键部署到 Streamlit Cloud：

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📁 项目结构

```
Mining-Lab/
├── app.py                      # Streamlit 主入口
├── requirements.txt            # 依赖列表
├── README.md                   # 项目说明
├── 算法模块/
│   ├── clustering.py           # K-Means 聚类
│   ├── classification.py       # 分类算法
│   ├── association.py          # Apriori 关联规则
│   ├── dimension_reduction.py  # PCA 降维
│   └── regression.py           # 回归分析
├── 工具/
│   ├── data_loader.py          # 数据加载
│   ├── visualizer.py           # 可视化工具
│   └── coze_assistant.py       # Coze API
├── 资源/
│   └── datasets/               # 内置数据集
└── 文档/
    ├── DEPLOY.md               # 部署指南
    └── USER_GUIDE.md           # 使用手册
```

## 🎯 内置数据集

| 数据集 | 类型 | 样本数 | 特征数 | 适用算法 |
|--------|------|--------|--------|----------|
| 鸢尾花 | 分类 | 150 | 4 | K-Means, 决策树, KNN, PCA |
| 酒馆数据 | 聚类 | 178 | 13 | K-Means, PCA |
| 乳腺癌 | 分类 | 569 | 30 | 决策树, KNN, 朴素贝叶斯 |
| 糖尿病 | 分类 | 768 | 8 | 逻辑回归, KNN |
| 购物篮 | 关联 | 1000 | 119 | Apriori |

## 🔧 配置 Coze AI 助手

首次使用时，在侧边栏输入您的 Coze API Key：

1. 访问 [Coze 控制台](https://console.coze.cn)
2. 创建或获取 API Key
3. 在侧边栏设置中粘贴 Key

## 📚 课程适用

- 《数据仓库与数据挖掘》
- 《机器学习基础》
- 《数据分析实践》

## 📄 许可证

MIT License

## 👨‍🏫 关于

Mining-Lab 由课程教学团队开发维护，如有问题请联系助教。
