# -*- coding: utf-8 -*-
"""
聚类算法模块
包含 K-Means 聚类算法的实现和可视化
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualizer import (
    plot_clustering_result, plot_elbow_curve, plot_silhouette_scores,
    get_figure_html, plot_distribution, plot_correlation_matrix
)
from utils.data_loader import load_builtin_dataset, prepare_clustering_data, DATASET_INFO


def render_kmeans_page():
    """渲染 K-Means 聚类算法页面"""
    
    st.markdown("## 📊 K-Means 聚类算法")
    st.markdown("---")
    
    # 参数设置
    with st.sidebar:
        st.markdown("### ⚙️ 参数设置")
        
        # 数据集选择
        dataset_name = st.selectbox(
            "选择数据集",
            options=['iris', 'wine', 'breast_cancer'],
            format_func=lambda x: DATASET_INFO[x]['name']
        )
        
        # 显示数据集信息
        ds_info = DATASET_INFO[dataset_name]
        with st.expander("📋 数据集信息"):
            st.markdown(f"""
            - **样本数**: {ds_info['n_samples']}
            - **特征数**: {ds_info['n_features']}
            - **类别数**: {ds_info['n_classes']}
            - **描述**: {ds_info['description']}
            """)
        
        st.markdown("---")
        
        # 算法参数
        st.markdown("#### 算法参数")
        
        n_clusters = st.slider("K 值 (聚类数)", 2, 10, 3)
        
        max_iter = st.slider("最大迭代次数", 50, 500, 300, 50)
        
        n_init = st.select_slider(
            "初始化次数",
            options=[5, 10, 20, 50],
            value=10
        )
        
        init_method = st.selectbox(
            "初始化方式",
            options=['k-means++', 'random'],
            format_func=lambda x: 'K-Means++ (推荐)' if x == 'k-means++' else 'Random'
        )
        
        # 可视化维度
        viz_dimensions = st.radio(
            "可视化维度",
            options=[2, 3],
            format_func=lambda x: f'{x}D',
            horizontal=True
        )
        
        # 分析选项
        st.markdown("---")
        st.markdown("#### 分析选项")
        
        show_elbow = st.checkbox("肘部法则曲线", value=True)
        show_silhouette = st.checkbox("轮廓系数分析", value=True)
        show_pca = st.checkbox("PCA降维对比", value=True)
    
    # 加载数据
    try:
        df = load_builtin_dataset(dataset_name)
        
        # 保存当前数据集信息
        st.session_state.current_dataset = {
            'name': DATASET_INFO[dataset_name]['name'],
            'shape': df.shape
        }
        st.session_state.current_algorithm = {
            'name': 'K-Means',
            'params': {
                'n_clusters': n_clusters,
                'max_iter': max_iter,
                'init': init_method
            }
        }
        
        # 数据预览
        with st.expander("📁 数据预览"):
            st.dataframe(df.head(10))
            
            col1, col2, col3 = st.columns(3)
            col1.metric("样本数", df.shape[0])
            col2.metric("特征数", df.shape[1] - 2)  # 排除target和target_name
            col3.metric("K 值", n_clusters)
        
        # 特征选择
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in ['target']]
        
        if len(feature_cols) < 2:
            st.error("数据集特征不足")
            return
        
        # 默认选择前2-4个特征
        default_features = feature_cols[:min(4, len(feature_cols))]
        selected_features = st.multiselect(
            "选择用于聚类的特征",
            options=feature_cols,
            default=default_features
        )
        
        if len(selected_features) < 2:
            st.warning("请至少选择2个特征")
            return
        
        # 数据预处理
        X = df[selected_features].values
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 特征分布
        with st.expander("📈 特征分布"):
            for col in selected_features:
                fig = plot_distribution(df, col, 'target')
                st.plotly_chart(fig, use_container_width=True)
        
        # 相关性矩阵
        if show_pca:
            with st.expander("🔗 特征相关性"):
                corr_df = df[selected_features]
                fig = plot_correlation_matrix(corr_df)
                st.plotly_chart(fig, use_container_width=True)
        
        # 运行 K-Means
        if st.button("🚀 开始聚类", type="primary", use_container_width=True):
            
            with st.spinner("正在执行 K-Means 聚类..."):
                
                # 训练模型
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    max_iter=max_iter,
                    n_init=n_init,
                    init=init_method,
                    random_state=42
                )
                
                labels = kmeans.fit_predict(X_scaled)
                centers = kmeans.cluster_centers_
                
                # 评估指标
                inertia = kmeans.inertia_
                silhouette = silhouette_score(X_scaled, labels)
                
                # 保存结果
                st.session_state.experiment_results = {
                    'labels': labels,
                    'centers': centers,
                    'inertia': inertia,
                    'silhouette': silhouette,
                    'summary': f'惯性: {inertia:.2f}, 轮廓系数: {silhouette:.3f}'
                }
            
            # 显示结果
            st.markdown("---")
            st.markdown("### 📊 聚类结果")
            
            # 评估指标卡片
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("惯性 (SSE)", f"{inertia:.2f}")
            col2.metric("轮廓系数", f"{silhouette:.3f}")
            col3.metric("聚类数", n_clusters)
            col4.metric("迭代次数", kmeans.n_iter_)
            
            # 聚类可视化
            if len(selected_features) >= 2:
                # 使用PCA降到指定维度
                pca = PCA(n_components=viz_dimensions)
                X_viz = pca.fit_transform(X_scaled)
                centers_viz = pca.transform(centers)
                
                fig = plot_clustering_result(
                    X_viz, labels, centers_viz,
                    title=f"K-Means 聚类结果 (K={n_clusters})",
                    dimensions=viz_dimensions
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 各簇统计
            st.markdown("#### 📋 各簇统计")
            
            df['cluster'] = labels
            cluster_stats = df.groupby('cluster')[selected_features].agg(['mean', 'std'])
            
            for cluster_id in range(n_clusters):
                with st.expander(f"簇 {cluster_id + 1} ({(labels == cluster_id).sum()} 样本)"):
                    cluster_data = df[df['cluster'] == cluster_id][selected_features]
                    st.dataframe(cluster_data.describe())
            
            # 肘部法则
            if show_elbow:
                st.markdown("---")
                st.markdown("### 📈 肘部法则分析")
                
                # 计算不同K值的惯性
                inertias = []
                k_range = range(2, min(11, df.shape[0]))
                
                progress_bar = st.progress(0)
                for i, k in enumerate(k_range):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    km.fit(X_scaled)
                    inertias.append(km.inertia_)
                    progress_bar.progress((i + 1) / len(k_range))
                
                fig = plot_elbow_curve(inertias, k_range)
                st.plotly_chart(fig, use_container_width=True)
                
                # 分析建议
                st.info("💡 **分析建议**: 肘部法则通过寻找惯性下降速度明显变缓的'肘点'来确定最佳K值。")
            
            # 轮廓系数分析
            if show_silhouette:
                st.markdown("---")
                st.markdown("### 📊 轮廓系数分析")
                
                silhouette_scores = {}
                for k in range(2, min(11, df.shape[0])):
                    km = KMeans(n_clusters=k, random_state=42, n_init=10)
                    sil_score = silhouette_score(X_scaled, km.fit_predict(X_scaled))
                    silhouette_scores[k] = sil_score
                
                fig = plot_silhouette_scores(silhouette_scores)
                st.plotly_chart(fig, use_container_width=True)
                
                # 建议
                best_k = max(silhouette_scores, key=silhouette_scores.get)
                st.success(f"🎯 根据轮廓系数分析，最佳K值建议为 **{best_k}** (轮廓系数: {silhouette_scores[best_k]:.3f})")
            
            # PCA降维对比
            if show_pca:
                st.markdown("---")
                st.markdown("### 🔄 PCA 降维对比")
                
                # 原始数据（仅前2特征）
                X_2d_original = X_scaled[:, :2]
                
                # PCA降维
                pca_2d = PCA(n_components=2)
                X_2d_pca = pca_2d.fit_transform(X_scaled)
                
                # 创建对比图
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('原始数据 (前2特征)', f'PCA降维 (保留 {pca_2d.explained_variance_ratio_.sum()*100:.1f}% 方差)')
                )
                
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
                
                for i in range(n_clusters):
                    mask = labels == i
                    fig.add_trace(
                        go.Scatter(
                            x=X_2d_original[mask, 0],
                            y=X_2d_original[mask, 1],
                            mode='markers',
                            name=f'簇 {i+1}',
                            marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7)
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=X_2d_pca[mask, 0],
                            y=X_2d_pca[mask, 1],
                            mode='markers',
                            name=f'簇 {i+1}',
                            marker=dict(size=8, color=colors[i % len(colors)], opacity=0.7),
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                
                fig.update_layout(height=500, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # 总结
            st.markdown("---")
            st.markdown("### 📝 实验总结")
            
            st.markdown(f"""
            **实验参数：**
            - 数据集：{DATASET_INFO[dataset_name]['name']}
            - 特征：{', '.join(selected_features)}
            - K 值：{n_clusters}
            - 初始化方式：{init_method}
            
            **关键发现：**
            1. 最终惯性（SSE）：{inertia:.2f}
            2. 轮廓系数：{silhouette:.3f}（{get_silhouette_interpretation(silhouette)}）
            3. 聚类迭代次数：{kmeans.n_iter_}
            
            **结论：**
            {get_clustering_conclusion(silhouette, n_clusters, inertia, dataset_name)}
            """)
            
            # 保存实验结果供报告使用
            st.session_state.experiment_results = {
                'algorithm': 'K-Means',
                'params': {
                    'n_clusters': n_clusters,
                    'max_iter': max_iter,
                    'init': init_method
                },
                'metrics': {
                    'inertia': inertia,
                    'silhouette': silhouette
                },
                'labels': labels.tolist(),
                'n_iter': kmeans.n_iter_
            }
            
    except Exception as e:
        st.error(f"发生错误: {str(e)}")


def get_silhouette_interpretation(score: float) -> str:
    """根据轮廓系数返回解释"""
    if score >= 0.7:
        return "结构合理，簇分离明显"
    elif score >= 0.5:
        return "结构合理，簇之间有重叠"
    elif score >= 0.25:
        return "结构较弱，建议尝试其他K值"
    else:
        return "结构不合理，数据可能不适合K-Means"


def get_clustering_conclusion(silhouette: float, n_clusters: int, 
                               inertia: float, dataset: str) -> str:
    """生成聚类结论"""
    
    conclusions = []
    
    # 轮廓系数评价
    if silhouette >= 0.5:
        conclusions.append(f"轮廓系数({silhouette:.3f})表明聚类效果良好，各簇边界清晰。")
    elif silhouette >= 0.25:
        conclusions.append(f"轮廓系数({silhouette:.3f})一般，建议调整K值或尝试其他聚类算法。")
    else:
        conclusions.append(f"轮廓系数({silhouette:.3f})较低，当前聚类可能不够理想。")
    
    # K值评价
    if dataset == 'iris' and n_clusters == 3:
        conclusions.append("K=3与鸢尾花的自然分类(3种)一致，聚类与真实标签有较好对应。")
    elif n_clusters > 5:
        conclusions.append("较大的K值可能导致过细的划分，建议根据业务需求选择。")
    
    # 惯性评价
    conclusions.append(f"惯性值为{inertia:.2f}，反映簇内紧密度。")
    
    return " ".join(conclusions)


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="K-Means 聚类 - Mining-Lab")
    render_kmeans_page()
