# -*- coding: utf-8 -*-
"""
分类算法模块
包含决策树、KNN、朴素贝叶斯等分类算法
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.metrics import (classification_report, accuracy_score, 
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from 工具.visualizer import (
    plot_confusion_matrix, plot_decision_boundary, plot_feature_importance,
    plot_cross_validation_curve, get_figure_html, plot_distribution
)
from 工具.data_loader import load_builtin_dataset, DATASET_INFO


def render_decision_tree_page():
    """渲染决策树分类器页面"""
    
    st.markdown("## 🌳 决策树分类器")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("### ⚙️ 参数设置")
        
        # 数据集选择
        dataset_name = st.selectbox(
            "选择数据集",
            options=['iris', 'wine', 'breast_cancer'],
            format_func=lambda x: DATASET_INFO[x]['name']
        )
        
        st.markdown("---")
        st.markdown("#### 算法参数")
        
        criterion = st.selectbox(
            "分裂准则",
            options=['gini', 'entropy', 'log_loss'],
            format_func=lambda x: {'gini': 'Gini (默认)', 'entropy': 'Entropy (信息增益)', 'log_loss': 'Log Loss'}[x]
        )
        
        max_depth = st.slider("最大深度", 1, 20, 5, help="0表示不限制")
        if max_depth == 0:
            max_depth = None
        
        min_samples_split = st.slider("最小分割样本数", 2, 50, 2)
        min_samples_leaf = st.slider("最小叶节点样本数", 1, 20, 1)
        
        max_features = st.selectbox(
            "最大特征数",
            options=[None, 'sqrt', 'log2'],
            format_func=lambda x: '全部特征' if x is None else f"√n ({x})"
        )
        
        st.markdown("---")
        st.markdown("#### 分析选项")
        show_tree = st.checkbox("显示决策树结构", value=True)
        show_importance = st.checkbox("特征重要性", value=True)
        show_report = st.checkbox("分类报告", value=True)
        show_cv = st.checkbox("交叉验证曲线", value=True)
    
    # 加载数据
    try:
        df = load_builtin_dataset(dataset_name)
        
        st.session_state.current_algorithm = {
            'name': '决策树',
            'params': {
                'criterion': criterion,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split
            }
        }
        
        # 数据预览
        with st.expander("📁 数据预览"):
            st.dataframe(df.head(10))
        
        # 特征选择
        feature_cols = [c for c in df.columns if c not in ['target', 'target_name']]
        selected_features = st.multiselect(
            "选择特征",
            options=feature_cols,
            default=feature_cols[:4]
        )
        
        # 目标变量
        target_col = st.selectbox(
            "选择目标变量",
            options=['target'],
            format_func=lambda x: '目标类别'
        )
        
        if len(selected_features) < 2:
            st.warning("请至少选择2个特征")
            return
        
        # 准备数据
        X = df[selected_features].values
        y = df[target_col].values
        
        # 类别名称映射
        if 'target_name' in df.columns:
            class_names = df['target_name'].unique().tolist()
        else:
            class_names = [f"类别 {i}" for i in range(len(np.unique(y)))]
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("训练样本", len(X_train))
        col2.metric("测试样本", len(X_test))
        col3.metric("类别数", len(np.unique(y)))
        
        # 训练模型
        if st.button("🚀 训练模型", type="primary", use_container_width=True):
            
            with st.spinner("正在训练决策树..."):
                clf = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=42
                )
                
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred_proba = clf.predict_proba(X_test)
                
                # 评估指标
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # 交叉验证
                cv_scores = cross_val_score(clf, X, y, cv=5)
                
                st.session_state.experiment_results = {
                    'algorithm': '决策树',
                    'accuracy': accuracy,
                    'metrics': {'precision': precision, 'recall': recall, 'f1': f1}
                }
            
            # 结果展示
            st.markdown("---")
            st.markdown("### 📊 分类结果")
            
            # 评估指标
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("准确率", f"{accuracy:.2%}")
            col2.metric("精确率", f"{precision:.2%}")
            col3.metric("召回率", f"{recall:.2%}")
            col4.metric("F1分数", f"{f1:.2%}")
            
            # 交叉验证
            if show_cv:
                st.markdown("#### 📈 交叉验证结果")
                col1, col2 = st.columns(2)
                col1.metric("CV 准确率 (均值)", f"{cv_scores.mean():.2%}")
                col2.metric("CV 准确率 (标准差)", f"{cv_scores.std():.2%}")
                
                # CV曲线
                cv_curve_placeholder = st.empty()
            
            # 混淆矩阵
            st.markdown("#### 🔢 混淆矩阵")
            fig = plot_confusion_matrix(y_test, y_pred, labels=class_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # 分类报告
            if show_report:
                st.markdown("#### 📋 详细分类报告")
                report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format({
                    'precision': '{:.2%}',
                    'recall': '{:.2%}',
                    'f1-score': '{:.2%}',
                    'support': '{:,.0f}'
                }))
            
            # 决策树可视化
            if show_tree:
                st.markdown("---")
                st.markdown("### 🌳 决策树结构")
                
                # 使用matplotlib绘制
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(
                    clf,
                    feature_names=selected_features,
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    ax=ax,
                    fontsize=10
                )
                st.pyplot(fig)
                
                # 导出树文本表示
                with st.expander("📝 决策规则 (文本)"):
                    tree_rules = export_text(clf, feature_names=selected_features)
                    st.code(tree_rules, language=None)
            
            # 特征重要性
            if show_importance:
                st.markdown("---")
                st.markdown("### 📊 特征重要性")
                
                importance = clf.feature_importances_
                fig = plot_feature_importance(importance, selected_features, title="特征重要性")
                st.plotly_chart(fig, use_container_width=True)
                
                # 重要性详情
                importance_df = pd.DataFrame({
                    '特征': selected_features,
                    '重要性': importance
                }).sort_values('重要性', ascending=False)
                st.dataframe(importance_df.style.format({'重要性': '{:.4f}'}))
            
            # 总结
            st.markdown("---")
            st.markdown("### 📝 实验总结")
            
            st.markdown(f"""
            **实验参数：**
            - 分裂准则：{criterion}
            - 最大深度：{max_depth if max_depth else '无限制'}
            - 最小分割样本：{min_samples_split}
            - 最小叶节点样本：{min_samples_leaf}
            
            **关键发现：**
            1. 准确率：{accuracy:.2%}
            2. 最重要特征：{selected_features[np.argmax(importance)]}
            3. 树深度：{clf.get_depth()}
            4. 叶节点数：{clf.get_n_leaves()}
            
            **结论：**
            决策树模型在该数据集上取得了{accuracy:.2%}的准确率。
            {f"特征'{selected_features[np.argmax(importance)]}'对分类贡献最大。" if max(importance) > 0.1 else ""}
            """)
    
    except Exception as e:
        st.error(f"发生错误: {str(e)}")


def render_knn_page():
    """渲染 KNN 分类器页面"""
    
    st.markdown("## 📍 KNN 分类器 (K近邻)")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("### ⚙️ 参数设置")
        
        # 数据集选择
        dataset_name = st.selectbox(
            "选择数据集",
            options=['iris', 'breast_cancer'],
            format_func=lambda x: DATASET_INFO[x]['name']
        )
        
        st.markdown("---")
        st.markdown("#### 算法参数")
        
        n_neighbors = st.slider("K 值 (近邻数)", 1, 30, 5)
        
        weights = st.selectbox(
            "权重方式",
            options=['uniform', 'distance'],
            format_func=lambda x: {'uniform': '统一权重', 'distance': '距离加权'}[x]
        )
        
        metric = st.selectbox(
            "距离度量",
            options=['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
            format_func=lambda x: {
                'euclidean': '欧氏距离',
                'manhattan': '曼哈顿距离',
                'chebyshev': '切比雪夫距离',
                'minkowski': '闵可夫斯基距离'
            }[x]
        )
        
        if metric == 'minkowski':
            p = st.slider("p 值 (Minkowski参数)", 1, 5, 2)
        else:
            p = 2
        
        st.markdown("---")
        st.markdown("#### 分析选项")
        show_boundary = st.checkbox("决策边界", value=True)
        show_cv_curve = st.checkbox("K值选择曲线", value=True)
    
    try:
        df = load_builtin_dataset(dataset_name)
        
        st.session_state.current_algorithm = {
            'name': 'KNN',
            'params': {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric}
        }
        
        # 数据预览
        with st.expander("📁 数据预览"):
            st.dataframe(df.head(10))
        
        # 特征选择（只选2个用于可视化）
        feature_cols = [c for c in df.columns if c not in ['target', 'target_name']]
        selected_features = st.multiselect(
            "选择特征 (选2个用于可视化)",
            options=feature_cols,
            default=feature_cols[:2],
            max_selections=2
        )
        
        if len(selected_features) != 2:
            st.info("请选择恰好2个特征以查看决策边界")
            # 自动补全
            if len(selected_features) < 2:
                selected_features = feature_cols[:2]
        
        # 数据准备
        X = df[selected_features].values
        y = df['target'].values
        
        class_names = df['target_name'].unique().tolist() if 'target_name' in df.columns else [f"类别 {i}" for i in range(len(np.unique(y)))]
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("样本数", len(X))
        col2.metric("K 值", n_neighbors)
        col3.metric("类别数", len(class_names))
        
        # 训练和评估
        if st.button("🚀 训练模型", type="primary", use_container_width=True):
            
            with st.spinner("正在训练 KNN..."):
                
                clf = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    p=p if metric == 'minkowski' else 2
                )
                
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            
            st.markdown("---")
            st.markdown("### 📊 分类结果")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("准确率", f"{accuracy:.2%}")
            col2.metric("精确率", f"{precision:.2%}")
            col3.metric("召回率", f"{recall:.2%}")
            col4.metric("F1分数", f"{f1:.2%}")
            
            # 混淆矩阵
            st.markdown("#### 🔢 混淆矩阵")
            fig = plot_confusion_matrix(y_test, y_pred, labels=class_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # 决策边界
            if show_boundary:
                st.markdown("#### 🗺️ 决策边界")
                
                clf_full = KNeighborsClassifier(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    metric=metric,
                    p=p
                )
                clf_full.fit(X_scaled, y)
                
                fig = plot_decision_boundary(X_scaled, y, clf_full, title="KNN 决策边界")
                st.plotly_chart(fig, use_container_width=True)
            
            # K值选择曲线
            if show_cv_curve:
                st.markdown("---")
                st.markdown("### 📈 K值选择曲线")
                
                cv_scores_dict = {}
                k_range = range(1, min(21, len(X_train)))
                
                for k in k_range:
                    knn = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)
                    scores = cross_val_score(knn, X_scaled, y, cv=5)
                    cv_scores_dict[k] = scores
                
                fig = plot_cross_validation_curve(cv_scores_dict, title="KNN 交叉验证曲线")
                st.plotly_chart(fig, use_container_width=True)
                
                # 最佳K
                best_k = max(range(1, len(cv_scores_dict) + 1), key=lambda k: np.mean(cv_scores_dict[k]))
                st.success(f"🎯 根据交叉验证，最佳K值为 **{best_k}**")
            
            st.markdown("---")
            st.markdown("### 📝 实验总结")
            st.markdown(f"""
            **参数配置：**
            - K值：{n_neighbors}
            - 权重方式：{weights}
            - 距离度量：{metric}
            
            **结论：**
            KNN模型准确率达到{accuracy:.2%}。
            {f"建议使用K={best_k}以获得更好的泛化性能。" if show_cv_curve else ""}
            """)
    
    except Exception as e:
        st.error(f"发生错误: {str(e)}")


def render_naive_bayes_page():
    """渲染朴素贝叶斯分类器页面"""
    
    st.markdown("## 📊 朴素贝叶斯分类器")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("### ⚙️ 参数设置")
        
        # 数据集选择
        dataset_name = st.selectbox(
            "选择数据集",
            options=['iris', 'breast_cancer', 'wine'],
            format_func=lambda x: DATASET_INFO[x]['name']
        )
        
        st.markdown("---")
        st.markdown("#### 算法参数")
        
        model_type = st.selectbox(
            "模型类型",
            options=['gaussian', 'multinomial'],
            format_func=lambda x: {
                'gaussian': 'GaussianNB (高斯朴素贝叶斯)',
                'multinomial': 'MultinomialNB (多项式朴素贝叶斯)'
            }[x]
        )
        
        alpha = st.slider("平滑因子 (alpha)", 0.0, 2.0, 1.0, 0.1)
        
        st.markdown("---")
        st.markdown("#### 分析选项")
        show_boundary = st.checkbox("分类边界", value=True)
        show_proba = st.checkbox("预测概率", value=True)
    
    try:
        df = load_builtin_dataset(dataset_name)
        
        st.session_state.current_algorithm = {
            'name': '朴素贝叶斯',
            'params': {'model_type': model_type, 'alpha': alpha}
        }
        
        with st.expander("📁 数据预览"):
            st.dataframe(df.head(10))
        
        # 特征选择
        feature_cols = [c for c in df.columns if c not in ['target', 'target_name']]
        selected_features = st.multiselect(
            "选择特征",
            options=feature_cols,
            default=feature_cols[:4]
        )
        
        X = df[selected_features].values
        y = df['target'].values
        
        class_names = df['target_name'].unique().tolist() if 'target_name' in df.columns else [f"类别 {i}" for i in range(len(np.unique(y)))]
        
        # 数据预处理
        if model_type == 'multinomial':
            # MultinomialNB需要非负数据
            X = MinMaxScaler().fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("训练样本", len(X_train))
        col2.metric("测试样本", len(X_test))
        col3.metric("类别数", len(class_names))
        
        if st.button("🚀 训练模型", type="primary", use_container_width=True):
            
            with st.spinner("正在训练..."):
                
                if model_type == 'gaussian':
                    clf = GaussianNB(alpha=alpha)
                else:
                    clf = MultinomialNB(alpha=alpha)
                
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            
            st.markdown("---")
            st.markdown("### 📊 分类结果")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("准确率", f"{accuracy:.2%}")
            col2.metric("精确率", f"{precision:.2%}")
            col3.metric("召回率", f"{recall:.2%}")
            col4.metric("F1分数", f"{f1:.2%}")
            
            # 混淆矩阵
            st.markdown("#### 🔢 混淆矩阵")
            fig = plot_confusion_matrix(y_test, y_pred, labels=class_names)
            st.plotly_chart(fig, use_container_width=True)
            
            # 分类边界（使用前2个特征）
            if show_boundary and len(selected_features) >= 2:
                st.markdown("#### 🗺️ 分类边界")
                
                clf_2d = GaussianNB(alpha=alpha) if model_type == 'gaussian' else MultinomialNB(alpha=alpha)
                clf_2d.fit(X_train[:, :2], y_train)
                
                fig = plot_decision_boundary(X_train[:, :2], y_train, clf_2d, title="朴素贝叶斯分类边界")
                st.plotly_chart(fig, use_container_width=True)
            
            # 预测概率
            if show_proba:
                st.markdown("#### 📈 预测概率分布")
                
                y_pred_proba = clf.predict_proba(X_test)
                
                proba_df = pd.DataFrame(
                    y_pred_proba[:10],
                    columns=[f'{cn} ({i})' for i, cn in enumerate(class_names)]
                )
                proba_df.index.name = '样本'
                st.dataframe(proba_df.style.format('{:.2%}'))
            
            # 算法解释
            st.markdown("---")
            st.markdown("### 📖 算法原理解释")
            st.markdown(f"""
            **{model_type}朴素贝叶斯** 基于贝叶斯定理，假设特征之间条件独立。
            
            - **先验概率**：P(类别) = 该类别样本数 / 总样本数
            - **似然**：P(特征|类别) = 在该类别下特征的条件概率
            - **平滑因子 α**：防止零概率问题（拉普拉斯平滑）
            
            **优势：**
            - 计算效率高，适合大规模数据
            - 对特征尺度不敏感
            - 可以处理多分类问题
            
            **劣势：**
            - 特征独立性假设在现实中往往不成立
            - 对输入数据分布有一定假设
            """)
    
    except Exception as e:
        st.error(f"发生错误: {str(e)}")


# 路由函数
def render_classification_subpage(algorithm: str):
    """根据算法类型渲染相应的页面"""
    if algorithm == 'decision_tree':
        render_decision_tree_page()
    elif algorithm == 'knn':
        render_knn_page()
    elif algorithm == 'naive_bayes':
        render_naive_bayes_page()


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="分类算法 - Mining-Lab")
    render_decision_tree_page()
