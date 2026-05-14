# -*- coding: utf-8 -*-
"""
关联规则挖掘模块
包含 Apriori 算法的实现和可视化
"""

import numpy as np
import pandas as pd
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import plotly.express as px
import plotly.graph_objects as go

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from 工具.data_loader import generate_shopping_basket_data
from 工具.visualizer import plot_frequent_itemsets, plot_association_heatmap


def render_apriori_page():
    """渲染 Apriori 关联规则挖掘页面"""
    
    st.markdown("## 🔗 Apriori 关联规则挖掘")
    st.markdown("---")
    
    with st.sidebar:
        st.markdown("### ⚙️ 参数设置")
        
        # 数据来源
        data_source = st.radio(
            "数据来源",
            options=['内置数据', '上传数据'],
            horizontal=True
        )
        
        if data_source == '内置数据':
            n_transactions = st.slider(
                "事务数量",
                min_value=100,
                max_value=5000,
                value=1000,
                step=100
            )
            
            if st.button("🔄 生成新数据", use_container_width=True):
                st.cache_data.clear()
        
        st.markdown("---")
        st.markdown("#### 算法参数")
        
        min_support = st.slider(
            "最小支持度",
            min_value=0.01,
            max_value=0.5,
            value=0.05,
            step=0.01,
            format="%.2f"
        )
        
        min_confidence = st.slider(
            "最小置信度",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            format="%.2f"
        )
        
        min_threshold = st.slider(
            "规则度量阈值",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        st.markdown("---")
        st.markdown("#### 显示选项")
        max_rules = st.slider("最多显示规则数", 5, 50, 20)
        show_frequent = st.checkbox("频繁项集", value=True)
        show_heatmap = st.checkbox("规则热力图", value=True)
    
    # 加载或生成数据
    if data_source == '内置数据':
        df = generate_shopping_basket_data(n_transactions=n_transactions)
        st.info(f"📦 使用内置购物篮数据，共 {n_transactions} 条事务")
    else:
        uploaded_file = st.file_uploader("上传购物篮数据 (CSV)", type=['csv'])
        
        if uploaded_file is None:
            st.warning("请上传 CSV 格式的购物篮数据")
            st.info("📋 CSV格式要求：每行代表一个事务，列名为商品名称，值为布尔值(1/0或True/False)")
            
            # 显示示例格式
            st.markdown("**示例数据格式：**")
            example_df = pd.DataFrame({
                '面包': [True, False, True],
                '牛奶': [True, True, False],
                '鸡蛋': [False, True, True],
                '奶酪': [True, True, False]
            })
            st.dataframe(example_df)
            return
        
        df = pd.read_csv(uploaded_file)
        
        # 转换为布尔型
        for col in df.columns:
            if df[col].dtype in [int, float]:
                df[col] = df[col] > 0
            elif df[col].dtype == object:
                df[col] = df[col].str.lower().isin(['true', '1', 'yes', 'y'])
    
    # 数据统计
    with st.expander("📁 数据统计"):
        total_transactions = len(df)
        total_items = len(df.columns)
        avg_items_per_transaction = df.sum(axis=1).mean()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("事务数", total_transactions)
        col2.metric("商品种类", total_items)
        col3.metric("平均商品数", f"{avg_items_per_transaction:.1f}")
        
        # 显示最常购买的商品
        item_counts = df.sum().sort_values(ascending=False)
        
        fig = px.bar(
            x=item_counts.head(15).values,
            y=item_counts.head(15).index,
            orientation='h',
            title='最常购买的商品 (Top 15)',
            color=item_counts.head(15).values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # 执行 Apriori
    if st.button("🔍 挖掘关联规则", type="primary", use_container_width=True):
        
        with st.spinner("正在执行 Apriori 算法..."):
            
            try:
                # 频繁项集挖掘
                frequent_itemsets = apriori(
                    df,
                    min_support=min_support,
                    use_colnames=True,
                    low_memory=True
                )
                
                if frequent_itemsets.empty:
                    st.warning(f"未找到支持度 >= {min_support} 的频繁项集，请降低最小支持度")
                    return
                
                # 生成关联规则
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=min_confidence
                )
                
                # 过滤有意义的规则
                if not rules.empty:
                    rules = rules[
                        (rules['antecedents'].apply(len) <= 3) &
                        (rules['consequents'].apply(len) <= 3)
                    ].sort_values('lift', ascending=False)
                
            except Exception as e:
                st.error(f"算法执行出错: {str(e)}")
                return
        
        st.success(f"✅ 找到 {len(rules)} 条关联规则")
        
        # 更新session state
        st.session_state.current_algorithm = {
            'name': 'Apriori',
            'params': {
                'min_support': min_support,
                'min_confidence': min_confidence
            }
        }
        
        st.session_state.experiment_results = {
            'algorithm': 'Apriori',
            'n_rules': len(rules),
            'metrics': {
                'avg_support': rules['support'].mean() if not rules.empty else 0,
                'avg_confidence': rules['confidence'].mean() if not rules.empty else 0,
                'avg_lift': rules['lift'].mean() if not rules.empty else 0
            }
        }
        
        # 结果展示
        st.markdown("---")
        st.markdown("### 📊 频繁项集")
        
        # 支持度分布
        fig = px.histogram(
            frequent_itemsets,
            x='support',
            nbins=30,
            title='频繁项集支持度分布',
            color_discrete_sequence=['#4ECDC4']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # 按项集长度分组
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        
        for length in sorted(frequent_itemsets['length'].unique()):
            subset = frequent_itemsets[frequent_itemsets['length'] == length].nlargest(10, 'support')
            
            with st.expander(f"📦 {length}项集 (Top 10)"):
                display_df = pd.DataFrame({
                    '项集': [', '.join(list(items)) for items in subset['itemsets']],
                    '支持度': subset['support'].round(4)
                })
                st.dataframe(display_df, use_container_width=True)
        
        # 关联规则
        st.markdown("---")
        st.markdown("### 🔗 关联规则")
        
        if rules.empty:
            st.info("未找到满足条件的关联规则，请尝试降低置信度阈值")
        else:
            # 规则指标卡片
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("规则总数", len(rules))
            col2.metric("平均支持度", f"{rules['support'].mean():.2%}")
            col3.metric("平均置信度", f"{rules['confidence'].mean():.2%}")
            col4.metric("平均提升度", f"{rules['lift'].mean():.2f}")
            
            # 规则表格
            display_rules = rules.head(max_rules).copy()
            display_rules['antecedents'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
            display_rules['consequents'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
            
            display_rules = display_rules[[
                'antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction'
            ]].rename(columns={
                'antecedents': '前项',
                'consequents': '后项',
                'support': '支持度',
                'confidence': '置信度',
                'lift': '提升度',
                'leverage': '杠杆率',
                'conviction': '确信度'
            })
            
            st.dataframe(
                display_rules.style.format({
                    '支持度': '{:.2%}',
                    '置信度': '{:.2%}',
                    '提升度': '{:.2f}',
                    '杠杆率': '{:.4f}',
                    '确信度': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # 指标解释
            with st.expander("📖 指标解释"):
                st.markdown("""
                | 指标 | 含义 | 理想值 |
                |------|------|--------|
                | **支持度** | 该规则出现的频率 | 较高表示规则普遍 |
                | **置信度** | 购买A后购买B的概率 | 较高表示规则可靠 |
                | **提升度** | 规则的实际提升程度 | >1表示正相关 |
                | **杠杆率** | 支持度差异 | 越大表示关联越强 |
                | **确信度** | 规则的反向强度 | 越大表示规则越强 |
                """)
            
            # 热力图
            if show_heatmap:
                st.markdown("#### 🔥 规则热力图")
                
                # 创建前20个规则的简化热力图
                top_rules = rules.nlargest(20, 'lift')
                
                labels = [f"{r['antecedents'].__str__()} → {r['consequents'].__str__()}" 
                         for _, r in top_rules.iterrows()]
                
                fig = go.Figure(data=go.Heatmap(
                    z=[top_rules['support'].values, top_rules['confidence'].values, top_rules['lift'].values],
                    y=['支持度', '置信度', '提升度'],
                    x=labels,
                    colorscale='RdYlGn',
                    text=[[f"{s:.2%}", f"{c:.2%}", f"{l:.2f}"] for s, c, l in 
                          zip(top_rules['support'], top_rules['confidence'], top_rules['lift'])],
                    texttemplate="%{text}"
                ))
                
                fig.update_layout(
                    title='Top 20 关联规则指标热力图',
                    height=300,
                    xaxis=dict(tickangle=45)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 散点图
            st.markdown("#### 📈 规则散点图")
            
            fig = px.scatter(
                rules,
                x='support',
                y='confidence',
                size='lift',
                color='lift',
                hover_data=['antecedents', 'consequents'],
                title='支持度 vs 置信度',
                labels={
                    'support': '支持度',
                    'confidence': '置信度',
                    'lift': '提升度'
                }
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # 分析建议
        st.markdown("---")
        st.markdown("### 💡 分析建议")
        
        if not rules.empty:
            best_rule = rules.nlargest(1, 'lift').iloc[0]
            
            st.markdown(f"""
            **最强关联规则：**
            
            > **{', '.join(list(best_rule['antecedents']))}** → **{', '.join(list(best_rule['consequents']))}**
            
            - 支持度：{best_rule['support']:.2%}
            - 置信度：{best_rule['confidence']:.2%}
            - 提升度：{best_rule['lift']:.2f}
            
            **商业解读：**
            当顾客购买 **{', '.join(list(best_rule['antecedents']))}** 时，
            有 **{best_rule['confidence']:.1%}** 的概率会同时购买 **{', '.join(list(best_rule['consequents']))}**。
            这个购买组合的频率是随机情况下的 **{best_rule['lift']:.1f}** 倍，具有很强的关联性。
            
            **营销建议：**
            1. 可以在 **{', '.join(list(best_rule['antecedents']))}** 附近摆放 **{', '.join(list(best_rule['consequents']))}**
            2. 对购买前项商品的顾客推荐后项商品
            3. 设计捆绑促销套餐
            """)
        
        # 总结
        st.markdown("---")
        st.markdown("### 📝 实验总结")
        
        st.markdown(f"""
        **实验参数：**
        - 数据集：{'内置购物篮数据' if data_source == '内置数据' else '用户上传数据'}
        - 事务数：{total_transactions}
        - 商品种类：{total_items}
        - 最小支持度：{min_support:.2%}
        - 最小置信度：{min_confidence:.2%}
        
        **关键发现：**
        1. 找到 {len(frequent_itemsets)} 个频繁项集
        2. 生成 {len(rules)} 条关联规则
        3. 平均支持度：{rules['support'].mean() if not rules.empty else 0:.2%}
        4. 平均置信度：{rules['confidence'].mean() if not rules.empty else 0:.2%}
        5. 最高提升度：{rules['lift'].max() if not rules.empty else 0:.2f}
        
        **结论：**
        Apriori算法成功挖掘出多条有价值的关联规则。
        提升度大于1的规则表明商品之间存在正相关关系，
        可以用于指导商品陈列、推荐系统和促销策略的制定。
        """)


if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Apriori 关联规则 - Mining-Lab")
    render_apriori_page()
