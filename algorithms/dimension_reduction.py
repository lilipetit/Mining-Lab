# -*- coding: utf-8 -*-
"""
降维算法模块
包含 PCA 主成分分析的的实现和可视化
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_builtin_dataset, DATASET_INFO
from utils.visualizer import plot_pca_comparison, plot_explained_variance


def render_pca_page():
    """渲染 PCA 降维分析页面"""
    
    st.markdown("## 🎯 PCA 主成分分析")
    st.markdown("---")
    
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
            - **描述**: {ds_info['description']}
            """)
        
        st.markdown("---")
        st.markdown("#### 算法参数")
        
        n_components = st.slider(
            "主成分数量",
            min_value=2,
            max_value=min(10, DATASET_INFO[dataset_name]['n_features']),
            value=2
        )
        
        analysis_mode = st.radio(
            "分析模式",
            options=['降维可视化', '方差分析', '特征分析'],
            horizontal=False
        )
        
        st.markdown("---")
        st.markdown("#### 可视化选项")
        
        show_3d = st.checkbox("3D 可视化", value=True)
        show_comparison = st.checkbox("降维前后对比", value=True)
        show_biplot = st.checkbox("双标图 (Biplot)", value=False)
        show_reconstruction = st.checkbox("重构误差分析", value=False)
    
    # 加载数据
    try:
        df = load_builtin_dataset(dataset_name)
        
        st.session_state.current_algorithm = {
            'name': 'PCA',
            'params': {'n_components': n_components}
        }
        
        # 数据预览
        with st.expander("📁 数据预览"):
            st.dataframe(df.head(10))
            
            col1, col2 = st.columns(2)
            col1.metric("样本数", df.shape[0])
            col2.metric("原始特征数", df.shape[1] - 2)
        
        # 特征选择
        feature_cols = [c for c in df.columns if c not in ['target', 'target_name']]
        selected_features = st.multiselect(
            "选择用于PCA的特征",
            options=feature_cols,
            default=feature_cols[:min(6, len(feature_cols))]
        )
        
        if len(selected_features) < 2:
            st.warning("请至少选择2个特征")
            return
        
        # 数据预处理
        X = df[selected_features].values
        y = df['target'].values
        y_names = df['target_name'].values if 'target_name' in df.columns else y
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 执行 PCA
        if st.button("🚀 开始分析", type="primary", use_container_width=True):
            
            with st.spinner("正在执行 PCA..."):
                
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X_scaled)
                
                # 各成分方差
                explained_variance = pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                # 重构误差
                X_reconstructed = pca.inverse_transform(X_pca)
                reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
                
                # 保存结果
                st.session_state.experiment_results = {
                    'algorithm': 'PCA',
                    'n_components': n_components,
                    'explained_variance': explained_variance.tolist(),
                    'cumulative_variance': cumulative_variance.tolist(),
                    'reconstruction_error': reconstruction_error
                }
            
            # 结果展示
            st.markdown("---")
            st.markdown("### 📊 PCA 分析结果")
            
            # 方差解释
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("主成分数", n_components)
            col2.metric("PC1 方差", f"{explained_variance[0]:.1%}")
            col3.metric("PC2 方差", f"{explained_variance[1]:.1%}" if n_components > 1 else "N/A")
            col4.metric("累积方差", f"{cumulative_variance[-1]:.1%}")
            
            # 方差解释图
            st.markdown("#### 📈 方差解释比例")
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('单个主成分方差', '累积方差')
            )
            
            # 单个方差
            fig.add_trace(
                go.Bar(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=explained_variance * 100,
                    name='方差比例',
                    marker_color='#4ECDC4'
                ),
                row=1, col=1
            )
            
            # 累积方差
            fig.add_trace(
                go.Scatter(
                    x=[f'PC{i+1}' for i in range(n_components)],
                    y=cumulative_variance * 100,
                    name='累积方差',
                    line=dict(width=3, color='#FF6B6B'),
                    fill='tozeroy'
                ),
                row=1, col=2
            )
            
            # 添加80%阈值线
            fig.add_hline(y=80, line_dash="dash", annotation_text="80%", row=1, col=2)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 降维结果可视化
            st.markdown("---")
            st.markdown("### 🎯 降维结果可视化")
            
            # 2D散点图
            fig = px.scatter(
                X_pca,
                x=0, y=1,
                color=y_names,
                title=f'PCA 降维结果 (PC1 vs PC2, 保留 {cumulative_variance[1]*100:.1f}% 方差)' if n_components > 1 else 'PCA 结果',
                labels={'0': 'PC1', '1': 'PC2', 'color': '类别'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            fig.update_layout(height=550)
            fig.update_traces(marker=dict(size=10, opacity=0.7))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 3D可视化
            if show_3d and n_components >= 3:
                st.markdown("#### 🌐 3D 可视化")
                
                fig = px.scatter_3d(
                    X_pca,
                    x=0, y=1, z=2,
                    color=y_names,
                    title=f'3D PCA 结果 (保留 {cumulative_variance[2]*100:.1f}% 方差)',
                    labels={'0': 'PC1', '1': 'PC2', '2': 'PC3', 'color': '类别'}
                )
                
                fig.update_layout(height=600)
                fig.update_traces(marker=dict(size=8, opacity=0.7))
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 降维前后对比
            if show_comparison:
                st.markdown("---")
                st.markdown("### 🔄 降维前后对比")
                
                # 原始数据（取前2个特征）
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=(
                        f'原始数据 (前2特征)', 
                        f'PCA 降维 (保留 {cumulative_variance[1]*100:.1f}%)' if n_components > 1 else 'PCA 降维'
                    )
                )
                
                # 原始
                fig.add_trace(
                    go.Scatter(
                        x=X_scaled[:, 0],
                        y=X_scaled[:, 1],
                        mode='markers',
                        marker=dict(size=8, opacity=0.7, color=y),
                        name='原始'
                    ),
                    row=1, col=1
                )
                
                # PCA后
                fig.add_trace(
                    go.Scatter(
                        x=X_pca[:, 0],
                        y=X_pca[:, 1],
                        mode='markers',
                        marker=dict(size=8, opacity=0.7, color=y),
                        name='PCA后',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **对比说明**：左图为原始特征空间，右图为PCA降维后的主成分空间。主成分是原始特征的线性组合。")
            
            # 主成分载荷
            st.markdown("---")
            st.markdown("### 📊 主成分载荷分析")
            
            # 载荷矩阵
            loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            loadings_df = pd.DataFrame(
                loadings[:, :min(3, n_components)],
                index=selected_features,
                columns=[f'PC{i+1}' for i in range(min(3, n_components))]
            )
            
            st.markdown("**主成分载荷矩阵** (前3个主成分):")
            st.dataframe(loadings_df.style.format('{:.3f}'))
            
            # 载荷热力图
            fig = px.imshow(
                loadings_df.values,
                x=loadings_df.columns,
                y=loadings_df.index,
                color_continuous_scale='RdBu_r',
                range_color=[-1, 1],
                title='主成分载荷热力图'
            )
            
            fig.update_layout(height=max(300, len(selected_features) * 30))
            st.plotly_chart(fig, use_container_width=True)
            
            # 载荷解释
            with st.expander("📖 载荷解读"):
                for i in range(min(3, n_components)):
                    top_positive = loadings_df.iloc[:, i].nlargest(3)
                    top_negative = loadings_df.iloc[:, i].nsmallest(3)
                    
                    st.markdown(f"""
                    **PC{i+1}** (解释 {explained_variance[i]*100:.1f}% 方差):
                    
                    - 高正向贡献: {', '.join([f'{feat}({val:.3f})' for feat, val in top_positive.items()])}
                    - 高负向贡献: {', '.join([f'{feat}({val:.3f})' for feat, val in top_negative.items()])}
                    """)
            
            # 重构误差分析
            if show_reconstruction:
                st.markdown("---")
                st.markdown("### 📉 重构误差分析")
                
                col1, col2 = st.columns(2)
                col1.metric("均方误差 (MSE)", f"{reconstruction_error:.4f}")
                col2.metric("信息保留率", f"{(1 - reconstruction_error / np.var(X_scaled)) * 100:.1f}%")
                
                # 各特征重构误差
                feature_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=0)
                
                fig = px.bar(
                    x=selected_features,
                    y=feature_errors,
                    title='各特征重构误差',
                    color=feature_errors,
                    color_continuous_scale='Reds'
                )
                
                fig.update_layout(
                    xaxis_title='特征',
                    yaxis_title='均方误差',
                    xaxis=dict(tickangle=45)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 主成分得分
            st.markdown("---")
            st.markdown("### 📋 主成分得分")
            
            scores_df = pd.DataFrame(
                X_pca[:, :min(5, n_components)],
                columns=[f'PC{i+1}' for i in range(min(5, n_components))]
            )
            scores_df['类别'] = y_names
            
            st.dataframe(scores_df.head(20), use_container_width=True)
            
            # 总结
            st.markdown("---")
            st.markdown("### 📝 实验总结")
            
            # 计算建议的主成分数
            n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1 if any(cumulative_variance >= 0.80) else n_components
            
            st.markdown(f"""
            **实验参数：**
            - 数据集：{DATASET_INFO[dataset_name]['name']}
            - 原始特征数：{len(selected_features)}
            - 选择主成分数：{n_components}
            
            **关键发现：**
            1. PC1 解释 **{explained_variance[0]*100:.1f}%** 的方差
            2. PC2 解释 **{explained_variance[1]*100:.1f}%** 的方差
            3. 前 {n_components} 个主成分累计解释 **{cumulative_variance[-1]*100:.1f}%** 的方差
            4. 重构均方误差：{reconstruction_error:.4f}
            
            **建议：**
            - 若需保留 80% 信息，建议使用 **{n_components_80}** 个主成分
            - PC1 主要反映 **{loadings_df.iloc[:, 0].abs().idxmax()}** 特征的信息
            {"- 数据降维效果明显，可有效减少计算复杂度" if cumulative_variance[-1] > 0.7 else "- 部分信息在降维过程中丢失"}
            
            **PCA 适用场景：**
            - 高维数据可视化
            - 特征降维加速模型训练
            - 去除特征冗余
            - 数据压缩
            """)
    
    except Exception as e:
        st.error(f"发生错误: {str(e)}")


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="PCA 降维 - Mining-Lab")
    render_pca_page()
