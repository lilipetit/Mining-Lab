# -*- coding: utf-8 -*-
"""
回归分析模块
包含线性回归和逻辑回归的实现和可视化
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             r2_score, accuracy_score, classification_report,
                             roc_curve, auc, confusion_matrix)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_builtin_dataset, DATASET_INFO
from utils.visualizer import plot_regression_fit, plot_residuals


def render_regression_page():
    """渲染回归分析页面"""
    
    st.markdown("## 📈 回归分析")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("### ⚙️ 参数设置")
        
        # 算法选择
        regression_type = st.radio(
            "回归类型",
            options=['linear', 'logistic'],
            format_func=lambda x: {'linear': '线性回归', 'logistic': '逻辑回归'}[x],
            horizontal=True
        )
        
        if regression_type == 'linear':
            dataset_name = st.selectbox(
                "选择数据集",
                options=['diabetes'],
                format_func=lambda x: DATASET_INFO[x]['name']
            )
        else:
            dataset_name = st.selectbox(
                "选择数据集",
                options=['breast_cancer', 'iris', 'diabetes'],
                format_func=lambda x: DATASET_INFO[x]['name']
            )
        
        st.markdown("---")
        st.markdown("#### 算法参数")
        
        if regression_type == 'linear':
            fit_intercept = st.checkbox("拟合截距", value=True)
            use_polynomial = st.checkbox("多项式特征", value=False)
            poly_degree = st.slider("多项式阶数", 2, 5, 2) if use_polynomial else 2
            regularization = st.selectbox(
                "正则化",
                options=['none', 'ridge', 'lasso'],
                format_func=lambda x: {'none': '无', 'ridge': 'Ridge (L2)', 'lasso': 'Lasso (L1)'}[x]
            )
            alpha = st.slider("正则化强度", 0.001, 1.0, 0.5) if regularization != 'none' else 1.0
        else:
            C = st.slider("正则化参数 C", 0.01, 10.0, 1.0)
            max_iter = st.slider("最大迭代次数", 100, 1000, 200)
            solver = st.selectbox(
                "优化算法",
                options=['lbfgs', 'liblinear', 'newton-cg'],
                format_func=lambda x: {'lbfgs': 'L-BFGS', 'liblinear': 'Liblinear', 'newton-cg': 'Newton-CG'}[x]
            )
        
        st.markdown("---")
        st.markdown("#### 可视化选项")
        show_residuals = st.checkbox("残差分析", value=True)
        show_coefficients = st.checkbox("系数分析", value=True)
        show_prediction = st.checkbox("预测 vs 真实", value=True)
    
    # 加载数据
    try:
        df = load_builtin_dataset(dataset_name)
        
        # 数据预览
        with st.expander("📁 数据预览"):
            st.dataframe(df.head(10))
        
        # 特征和目标选择
        feature_cols = [c for c in df.columns if c not in ['target', 'target_name']]
        
        if regression_type == 'linear':
            # 线性回归：选择一个特征进行可视化
            selected_feature = st.selectbox(
                "选择特征变量",
                options=feature_cols[:min(6, len(feature_cols))],
                format_func=lambda x: x
            )
            
            X = df[[selected_feature]].values
            y = df['target'].values
            
            st.session_state.current_algorithm = {
                'name': '线性回归',
                'params': {'fit_intercept': fit_intercept, 'polynomial': use_polynomial}
            }
        else:
            # 逻辑回归：多特征分类
            selected_features = st.multiselect(
                "选择特征",
                options=feature_cols,
                default=feature_cols[:min(4, len(feature_cols))]
            )
            
            if len(selected_features) < 2:
                st.warning("请至少选择2个特征")
                return
            
            X = df[selected_features].values
            y = df['target'].values
            
            st.session_state.current_algorithm = {
                'name': '逻辑回归',
                'params': {'C': C, 'solver': solver}
            }
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("训练样本", len(X_train))
        col2.metric("测试样本", len(X_test))
        if regression_type == 'logistic':
            col3.metric("类别数", len(np.unique(y)))
        
        # 训练模型
        if st.button("🚀 训练模型", type="primary", use_container_width=True):
            
            with st.spinner("正在训练..."):
                
                if regression_type == 'linear':
                    # 多项式特征
                    if use_polynomial:
                        poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                        X_train_poly = poly.fit_transform(X_train)
                        X_test_poly = poly.transform(X_test)
                    else:
                        X_train_poly = X_train
                        X_test_poly = X_test
                    
                    # 训练模型
                    model = LinearRegression(fit_intercept=fit_intercept)
                    model.fit(X_train_poly, y_train)
                    
                    # 预测
                    y_train_pred = model.predict(X_train_poly)
                    y_test_pred = model.predict(X_test_poly)
                    
                    # 评估指标
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    train_mae = mean_absolute_error(y_train, y_train_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    
                    # 交叉验证
                    cv_scores = cross_val_score(
                        LinearRegression() if not use_polynomial else model,
                        X_scaled, y, cv=5, scoring='r2'
                    )
                
                else:
                    # 逻辑回归
                    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
                    model.fit(X_train, y_train)
                    
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    y_test_proba = model.predict_proba(X_test)
                    
                    # 评估指标
                    train_acc = accuracy_score(y_train, y_train_pred)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    precision = np.mean([accuracy_score(y_test, y_test_pred)])  # 简化
                    
                    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            
            # 结果展示
            st.markdown("---")
            
            if regression_type == 'linear':
                st.markdown("### 📊 线性回归结果")
                
                # 评估指标
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("训练 R²", f"{train_r2:.4f}")
                col2.metric("测试 R²", f"{test_r2:.4f}")
                col3.metric("训练 MSE", f"{train_mse:.2f}")
                col4.metric("测试 MSE", f"{test_mse:.2f}")
                
                # 拟合曲线
                st.markdown("#### 📈 拟合结果")
                
                # 排序用于绘制曲线
                sort_idx = np.argsort(X_train.flatten())
                X_sorted = X_train[sort_idx]
                y_sorted = y_train[sort_idx]
                y_pred_sorted = y_train_pred[sort_idx]
                
                fig = go.Figure()
                
                # 真实点
                fig.add_trace(go.Scatter(
                    x=X_sorted.flatten(),
                    y=y_sorted,
                    mode='markers',
                    name='真实值',
                    marker=dict(size=8, opacity=0.6)
                ))
                
                # 拟合线
                fig.add_trace(go.Scatter(
                    x=X_sorted.flatten(),
                    y=y_pred_sorted,
                    mode='lines',
                    name='拟合曲线',
                    line=dict(width=3, color='#FF6B6B')
                ))
                
                fig.update_layout(
                    title=f'线性回归拟合 (R² = {train_r2:.4f})',
                    xaxis_title=selected_feature,
                    yaxis_title='目标值',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 残差分析
                if show_residuals:
                    st.markdown("#### 📉 残差分析")
                    
                    residuals = y_train - y_train_pred
                    
                    fig = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('残差分布', '残差 vs 预测值')
                    )
                    
                    fig.add_trace(
                        go.Histogram(x=residuals, name='残差', marker_color='#4ECDC4', nbinsx=30),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=y_train_pred,
                            y=residuals,
                            mode='markers',
                            name='残差',
                            marker=dict(size=8, opacity=0.6)
                        ),
                        row=1, col=2
                    )
                    
                    fig.add_hline(y=0, line_dash="dash", row=1, col=2)
                    
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # 系数分析
                if show_coefficients:
                    st.markdown("#### 📊 模型系数")
                    
                    if use_polynomial:
                        coef_df = pd.DataFrame({
                            '特征': [f'{selected_feature}^{i+1}' for i in range(len(model.coef_))],
                            '系数': model.coef_
                        })
                    else:
                        coef_df = pd.DataFrame({
                            '特征': [selected_feature, '截距'],
                            '系数': [model.coef_[0], model.intercept_]
                        })
                    
                    st.dataframe(coef_df.style.format({'系数': '{:.4f}'}))
                    
                    fig = px.bar(
                        x=coef_df['特征'],
                        y=coef_df['系数'],
                        title='回归系数',
                        color=coef_df['系数'],
                        color_continuous_scale='RdBu_r'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # 预测 vs 真实
                if show_prediction:
                    st.markdown("#### 🎯 预测 vs 真实")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=y_test,
                        y=y_test_pred,
                        mode='markers',
                        name='预测点',
                        marker=dict(size=10, opacity=0.7)
                    ))
                    
                    # 理想线
                    fig.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='理想预测',
                        line=dict(dash='dash', color='red')
                    ))
                    
                    fig.update_layout(
                        title=f'测试集: 预测 vs 真实 (R² = {test_r2:.4f})',
                        xaxis_title='真实值',
                        yaxis_title='预测值',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 总结
                st.markdown("---")
                st.markdown("### 📝 实验总结")
                
                st.markdown(f"""
                **实验参数：**
                - 特征：{selected_feature}
                - 截距：{'是' if fit_intercept else '否'}
                - 多项式：{'是' if use_polynomial else '否'}{f' (阶数={poly_degree})' if use_polynomial else ''}
                
                **关键发现：**
                1. 训练集 R² = {train_r2:.4f} ({'良好拟合' if train_r2 > 0.7 else '拟合一般' if train_r2 > 0.4 else '拟合较差'})
                2. 测试集 R² = {test_r2:.4f}
                3. 交叉验证 R² = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})
                4. 回归系数 = {model.coef_[0]:.4f}
                
                **结论：**
                {"模型拟合效果良好，能够解释数据中的大部分变异。" if test_r2 > 0.7 else "模型拟合效果一般，建议尝试非线性模型或增加特征。"}
                {"存在过拟合风险（训练/测试差距大）" if train_r2 - test_r2 > 0.1 else "模型泛化能力良好。"}
                """)
            
            else:  # 逻辑回归
                st.markdown("### 📊 逻辑回归结果")
                
                # 评估指标
                col1, col2, col3 = st.columns(3)
                col1.metric("训练准确率", f"{train_acc:.2%}")
                col2.metric("测试准确率", f"{test_acc:.2%}")
                col3.metric("CV 准确率", f"{cv_scores.mean():.2%}")
                
                # 混淆矩阵
                st.markdown("#### 🔢 混淆矩阵")
                
                cm = confusion_matrix(y_test, y_test_pred)
                
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['预测 0', '预测 1'],
                    y=['真实 0', '真实 1'],
                    colorscale='Blues',
                    text=cm,
                    texttemplate='%{text}',
                    textfont={"size": 20}
                ))
                
                fig.update_layout(height=400, width=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # 分类报告
                st.markdown("#### 📋 分类报告")
                
                report = classification_report(y_test, y_test_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format('{:.2%}'))
                
                # ROC曲线
                if len(np.unique(y)) == 2:
                    st.markdown("#### 📈 ROC 曲线")
                    
                    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        mode='lines',
                        name=f'ROC (AUC = {roc_auc:.3f})',
                        line=dict(width=3)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        name='随机分类器',
                        line=dict(dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='ROC 曲线',
                        xaxis_title='假正率 (FPR)',
                        yaxis_title='真正率 (TPR)',
                        height=500
                    )
                    
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        text=f'AUC = {roc_auc:.3f}',
                        showarrow=False,
                        font=dict(size=20)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 系数分析
                if show_coefficients:
                    st.markdown("#### 📊 特征系数")
                    
                    coef_df = pd.DataFrame({
                        '特征': selected_features,
                        '系数': model.coef_[0]
                    }).sort_values('系数', key=abs, ascending=False)
                    
                    fig = px.bar(
                        x=coef_df['特征'],
                        y=coef_df['系数'],
                        title='特征系数（对数几率）',
                        color=coef_df['系数'],
                        color_continuous_scale='RdBu_r'
                    )
                    
                    fig.update_layout(xaxis=dict(tickangle=45))
                    st.plotly_chart(fig, use_container_width=True)
                
                # 总结
                st.markdown("---")
                st.markdown("### 📝 实验总结")
                
                st.markdown(f"""
                **实验参数：**
                - 特征：{', '.join(selected_features[:3])}{'...' if len(selected_features) > 3 else ''}
                - 正则化参数 C = {C}
                - 优化算法：{solver}
                
                **关键发现：**
                1. 测试准确率：{test_acc:.2%}
                2. 交叉验证准确率：{cv_scores.mean():.2%} (±{cv_scores.std():.2%})
                3. AUC：{roc_auc:.3f}
                
                **最重要特征（按系数绝对值）：**
                {coef_df.iloc[0]['特征']}、{coef_df.iloc[1]['特征'] if len(coef_df) > 1 else 'N/A'}
                
                **结论：**
                {"逻辑回归模型在该数据集上表现良好。" if test_acc > 0.85 else "模型准确率一般，可尝试其他分类算法或调参。"}
                """)
    
    except Exception as e:
        st.error(f"发生错误: {str(e)}")


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="回归分析 - Mining-Lab")
    render_regression_page()
