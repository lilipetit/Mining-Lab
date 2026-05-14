# -*- coding: utf-8 -*-
"""
可视化工具模块
提供各类图表生成功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import seaborn as sns
from sklearn.metrics import confusion_matrix, silhouette_score
import streamlit as st
from io import BytesIO
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 颜色方案
COLORS = px.colors.qualitative.Set2
DISCRETE_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']


def get_figure_html(fig) -> str:
    """
    将Plotly图表转换为HTML字符串
    
    Args:
        fig: Plotly图表对象
        
    Returns:
        str: HTML字符串
    """
    return fig.to_html(full_html=False, include_plotlyjs='cdn')


def download_plotly_image(fig, format: str = 'png', width: int = 800, height: int = 600) -> bytes:
    """
    下载Plotly图表为图片
    
    Args:
        fig: Plotly图表对象
        format: 图片格式
        width: 宽度
        height: 高度
        
    Returns:
        bytes: 图片二进制数据
    """
    return fig.to_image(format=format, width=width, height=height)


def plot_clustering_result(X: np.ndarray, labels: np.ndarray, centers: np.ndarray = None, 
                          title: str = "聚类结果", dimensions: int = 2) -> go.Figure:
    """
    绘制聚类结果散点图
    
    Args:
        X: 特征数据
        labels: 聚类标签
        centers: 聚类中心
        title: 图表标题
        dimensions: 维度（2或3）
        
    Returns:
        go.Figure: Plotly图表
    """
    n_clusters = len(np.unique(labels))
    
    if dimensions == 2:
        fig = go.Figure()
        
        # 绘制每个簇的点
        for i in range(n_clusters):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                name=f'簇 {i+1}',
                marker=dict(size=8, opacity=0.7)
            ))
        
        # 绘制聚类中心
        if centers is not None:
            fig.add_trace(go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode='markers',
                name='中心点',
                marker=dict(size=20, symbol='x', color='black', line=dict(width=2))
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='特征 1',
            yaxis_title='特征 2',
            height=500,
            width=700
        )
        
    else:  # 3D
        fig = go.Figure()
        
        for i in range(n_clusters):
            mask = labels == i
            fig.add_trace(go.Scatter3d(
                x=X[mask, 0],
                y=X[mask, 1],
                z=X[mask, 2] if X.shape[1] > 2 else np.zeros(mask.sum()),
                mode='markers',
                name=f'簇 {i+1}',
                marker=dict(size=6, opacity=0.7)
            ))
        
        if centers is not None:
            fig.add_trace(go.Scatter3d(
                x=centers[:, 0],
                y=centers[:, 1],
                z=centers[:, 2] if centers.shape[1] > 2 else np.zeros(len(centers)),
                mode='markers',
                name='中心点',
                marker=dict(size=12, symbol='x', color='black')
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='特征 1',
                yaxis_title='特征 2',
                zaxis_title='特征 3'
            ),
            height=600
        )
    
    return fig


def plot_elbow_curve(inertias: list, k_range: range, title: str = "肘部法则") -> go.Figure:
    """
    绘制肘部法则曲线
    
    Args:
        inertias: 各K值对应的惯性
        k_range: K值范围
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(k_range),
        y=inertias,
        mode='lines+markers',
        name='惯性 (Inertia)',
        line=dict(width=3, color='#4ECDC4'),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='K 值 (聚类数)',
        yaxis_title='惯性 (SSE)',
        height=400,
        width=600
    )
    
    return fig


def plot_silhouette_scores(scores: dict, title: str = "轮廓系数分析") -> go.Figure:
    """
    绘制轮廓系数曲线
    
    Args:
        scores: K值到轮廓系数的映射
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    k_values = list(scores.keys())
    score_values = list(scores.values())
    
    # 找到最佳K
    best_k = k_values[np.argmax(score_values)]
    best_score = max(score_values)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k_values,
        y=score_values,
        mode='lines+markers',
        name='轮廓系数',
        line=dict(width=3, color='#FF6B6B'),
        marker=dict(size=10)
    ))
    
    # 标注最佳K
    fig.add_annotation(
        x=best_k,
        y=best_score,
        text=f"最佳 K={best_k}",
        showarrow=True,
        arrowhead=2,
        arrowcolor='red'
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='K 值',
        yaxis_title='轮廓系数',
        height=400,
        width=600
    )
    
    return fig


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                          labels: list = None, title: str = "混淆矩阵") -> go.Figure:
    """
    绘制混淆矩阵热力图
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 类别标签
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f'类别 {i}' for i in range(cm.shape[0])]
    
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        text_auto=True,
        title=title
    )
    
    fig.update_layout(
        xaxis_title='预测类别',
        yaxis_title='真实类别',
        height=500,
        width=500
    )
    
    return fig


def plot_decision_boundary(X: np.ndarray, y: np.ndarray, model, 
                          title: str = "决策边界", resolution: float = 0.1) -> go.Figure:
    """
    绘制分类决策边界
    
    Args:
        X: 特征数据（2维）
        y: 标签
        model: 训练好的分类器
        title: 图表标题
        resolution: 网格分辨率
        
    Returns:
        go.Figure: Plotly图表
    """
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 创建图形
    fig = go.Figure()
    
    # 添加决策边界等高线
    fig.add_trace(go.Contour(
        x=np.arange(x_min, x_max, resolution),
        y=np.arange(y_min, y_max, resolution),
        z=Z,
        colorscale='Pastel1',
        showscale=False,
        opacity=0.6
    ))
    
    # 添加数据点
    unique_labels = np.unique(y)
    for i, label in enumerate(unique_labels):
        mask = y == label
        fig.add_trace(go.Scatter(
            x=X[mask, 0],
            y=X[mask, 1],
            mode='markers',
            name=f'类别 {label}',
            marker=dict(size=10, color=DISCRETE_COLORS[i])
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='特征 1',
        yaxis_title='特征 2',
        height=600,
        width=700
    )
    
    return fig


def plot_feature_importance(importance: np.ndarray, feature_names: list, 
                           title: str = "特征重要性") -> go.Figure:
    """
    绘制特征重要性柱状图
    
    Args:
        importance: 重要性分数
        feature_names: 特征名称
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    # 排序
    sorted_idx = np.argsort(importance)[::-1]
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    fig = px.bar(
        x=sorted_importance,
        y=sorted_names,
        orientation='h',
        title=title,
        color=sorted_importance,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='重要性分数',
        yaxis_title='特征',
        height=max(400, len(feature_names) * 40),
        width=600,
        showlegend=False
    )
    
    return fig


def plot_pca_comparison(X_original: np.ndarray, X_pca: np.ndarray, 
                        explained_variance: np.ndarray, 
                        original_features: list = None) -> go.Figure:
    """
    绘制PCA降维前后对比图
    
    Args:
        X_original: 原始数据
        X_pca: PCA降维后的数据
        explained_variance: 各主成分的解释方差比例
        original_features: 原始特征名
        
    Returns:
        go.Figure: Plotly图表
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('原始数据 (前2特征)', 'PCA降维后'),
        horizontal_spacing=0.15
    )
    
    # 原始数据
    fig.add_trace(
        go.Scatter(
            x=X_original[:, 0],
            y=X_original[:, 1],
            mode='markers',
            marker=dict(size=8, opacity=0.7, color=X_original[:, 1] if X_original.shape[1] > 1 else 'blue'),
            name='原始数据'
        ),
        row=1, col=1
    )
    
    # PCA后数据
    fig.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            mode='markers',
            marker=dict(size=8, opacity=0.7, color=X_pca[:, 1] if X_pca.shape[1] > 1 else 'green'),
            name='PCA后'
        ),
        row=1, col=2
    )
    
    # 添加方差解释注释
    total_variance = sum(explained_variance[:2]) * 100 if len(explained_variance) >= 2 else 0
    
    fig.update_layout(
        title=f'PCA降维效果对比 (保留 {total_variance:.1f}% 方差)',
        height=500,
        width=1000,
        showlegend=True
    )
    
    return fig


def plot_explained_variance(explained_variance: np.ndarray, cumulative: bool = False) -> go.Figure:
    """
    绘制解释方差比例图
    
    Args:
        explained_variance: 各主成分的解释方差比例
        cumulative: 是否显示累积方差
        
    Returns:
        go.Figure: Plotly图表
    """
    n_components = len(explained_variance)
    
    fig = go.Figure()
    
    if cumulative:
        cumulative_var = np.cumsum(explained_variance)
        fig.add_trace(go.Scatter(
            x=list(range(1, n_components + 1)),
            y=cumulative_var * 100,
            mode='lines+markers',
            name='累积方差',
            line=dict(width=3),
            fill='tozeroy'
        ))
        
        # 添加80%阈值线
        fig.add_hline(y=80, line_dash="dash", annotation_text="80%阈值")
        
        y_label = '累积方差解释比例 (%)'
    else:
        fig.add_trace(go.Bar(
            x=list(range(1, n_components + 1)),
            y=explained_variance * 100,
            name='单个方差',
            marker_color='#4ECDC4'
        ))
        y_label = '方差解释比例 (%)'
    
    fig.update_layout(
        title='主成分方差解释比例',
        xaxis_title='主成分',
        yaxis_title=y_label,
        height=400,
        width=600
    )
    
    return fig


def plot_correlation_matrix(df: pd.DataFrame, title: str = "特征相关性矩阵") -> go.Figure:
    """
    绘制特征相关性矩阵热力图
    
    Args:
        df: 数据集
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    fig = px.imshow(
        corr,
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1],
        title=title
    )
    
    fig.update_layout(
        height=max(500, len(corr) * 30),
        width=max(600, len(corr) * 30)
    )
    
    return fig


def plot_regression_fit(X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, 
                       title: str = "回归拟合结果") -> go.Figure:
    """
    绘制回归拟合结果
    
    Args:
        X: 特征数据
        y: 真实值
        y_pred: 预测值
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    # 排序以绘制线条
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx].flatten()
    y_sorted = y[sort_idx]
    y_pred_sorted = y_pred[sort_idx]
    
    fig = go.Figure()
    
    # 真实值
    fig.add_trace(go.Scatter(
        x=X_sorted,
        y=y_sorted,
        mode='markers',
        name='真实值',
        marker=dict(size=8, opacity=0.7)
    ))
    
    # 预测值
    fig.add_trace(go.Scatter(
        x=X_sorted,
        y=y_pred_sorted,
        mode='lines',
        name='预测值',
        line=dict(width=3, color='red')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='特征',
        yaxis_title='目标值',
        height=500,
        width=700
    )
    
    return fig


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "残差分析") -> go.Figure:
    """
    绘制残差分析图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('残差分布', '残差 vs 预测值')
    )
    
    # 残差直方图
    fig.add_trace(
        go.Histogram(x=residuals, name='残差', marker_color='#4ECDC4', nbinsx=30),
        row=1, col=1
    )
    
    # 残差vs预测值
    fig.add_trace(
        go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            name='残差',
            marker=dict(size=8, opacity=0.7)
        ),
        row=1, col=2
    )
    
    # 添加零线
    fig.add_hline(y=0, line_dash="dash", row=1, col=2)
    
    fig.update_layout(
        title=title,
        height=400,
        width=900,
        showlegend=False
    )
    
    return fig


def plot_association_heatmap(rules_df: pd.DataFrame, metric: str = 'confidence') -> go.Figure:
    """
    绘制关联规则热力图
    
    Args:
        rules_df: 关联规则DataFrame
        metric: 显示的指标
        
    Returns:
        go.Figure: Plotly图表
    """
    if rules_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="没有找到满足条件的关联规则", showarrow=False)
        return fig
    
    # 选择前N个规则
    top_rules = rules_df.nlargest(15, metric)
    
    # 创建标签
    labels = [f"{row['antecedents']} → {row['consequents']}" 
              for _, row in top_rules.iterrows()]
    
    fig = go.Figure(data=go.Heatmap(
        z=top_rules[[metric]].values,
        y=labels,
        x=[metric],
        colorscale='YlOrRd',
        text=top_rules[[metric]].values,
        texttemplate='%{text:.3f}',
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title=f'Top 15 关联规则 - {metric}',
        height=max(400, len(labels) * 30),
        width=600,
        yaxis=dict(autorange='reversed')
    )
    
    return fig


def plot_cross_validation_curve(cv_scores: dict, title: str = "交叉验证曲线") -> go.Figure:
    """
    绘制交叉验证曲线（如KNN的K值选择）
    
    Args:
        cv_scores: K值到交叉验证分数的映射
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    k_values = list(cv_scores.keys())
    scores = [np.mean(v) for v in cv_scores.values()]
    std_scores = [np.std(v) for v in cv_scores.values()]
    
    # 找到最佳K
    best_idx = np.argmax(scores)
    best_k = k_values[best_idx]
    
    fig = go.Figure()
    
    # 添加误差带
    fig.add_trace(go.Scatter(
        x=k_values,
        y=[s + e for s, e in zip(scores, std_scores)],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        name='+1 std'
    ))
    
    fig.add_trace(go.Scatter(
        x=k_values,
        y=[s - e for s, e in zip(scores, std_scores)],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        name='±1 std'
    ))
    
    # 添加均值线
    fig.add_trace(go.Scatter(
        x=k_values,
        y=scores,
        mode='lines+markers',
        name='平均准确率',
        line=dict(width=3, color='#4ECDC4'),
        marker=dict(size=10)
    ))
    
    # 标注最佳K
    fig.add_annotation(
        x=best_k,
        y=scores[best_idx],
        text=f"最佳 K={best_k}",
        showarrow=True,
        arrowhead=2
    )
    
    fig.update_layout(
        title=title,
        xaxis_title='K 值',
        yaxis_title='交叉验证准确率',
        height=450,
        width=700
    )
    
    return fig


def plot_distribution(df: pd.DataFrame, column: str, target: str = None) -> go.Figure:
    """
    绘制特征分布图
    
    Args:
        df: 数据集
        column: 列名
        target: 目标列（用于分组）
        
    Returns:
        go.Figure: Plotly图表
    """
    if target and target in df.columns:
        fig = px.histogram(
            df, x=column, color=target,
            marginal='box',
            title=f'{column} 分布（按 {target} 分组）'
        )
    else:
        fig = px.histogram(
            df, x=column,
            marginal='box',
            title=f'{column} 分布'
        )
    
    fig.update_layout(height=450, width=700)
    return fig


def plot_frequent_itemsets(itemsets: dict, title: str = "频繁项集") -> go.Figure:
    """
    绘制频繁项集条形图
    
    Args:
        itemsets: 项集到支持度的映射
        title: 图表标题
        
    Returns:
        go.Figure: Plotly图表
    """
    # 排序并选择前15
    sorted_items = sorted(itemsets.items(), key=lambda x: x[1], reverse=True)[:15]
    
    labels = [str(item[0]) for item in sorted_items]
    values = [item[1] for item in sorted_items]
    
    fig = px.bar(
        x=labels,
        y=values,
        color=values,
        color_continuous_scale='Blues',
        title=title
    )
    
    fig.update_layout(
        xaxis_title='频繁项集',
        yaxis_title='支持度',
        height=450,
        width=800,
        xaxis=dict(tickangle=45)
    )
    
    return fig
