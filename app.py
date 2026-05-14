# -*- coding: utf-8 -*-
"""
Mining-Lab: 数据挖掘算法实验平台
主入口文件

一个面向《数据仓库与数据挖掘》课程的交互式 Web 实验平台
"""

import streamlit as st
import sys
import os
from datetime import datetime
import base64

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模块
from 算法模块.clustering import render_kmeans_page
from 算法模块.classification import render_classification_subpage
from 算法模块.association import render_apriori_page
from 算法模块.dimension_reduction import render_pca_page
from 算法模块.regression import render_regression_page
from 工具.coze_assistant import render_coze_sidebar, render_chat_interface, get_suggested_questions
from 工具.data_loader import DATASET_INFO, load_csv_file, get_dataset_info
from 工具.visualizer import download_plotly_image

# 页面配置
st.set_page_config(
    page_title="Mining-Lab - 数据挖掘实验平台",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 页面样式
st.markdown("""
<style>
    /* 主标题样式 */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* 指标卡片 */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 算法标签 */
    .algo-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: #e3f2fd;
        color: #1565c0;
        border-radius: 1rem;
        font-size: 0.875rem;
        margin: 0.25rem;
    }
    
    /* 成功提示 */
    .success-box {
        padding: 1rem;
        background: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    """渲染侧边栏导航"""
    
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <h2 style="color: #2E86AB; margin: 0;">🧪 Mining-Lab</h2>
            <p style="color: #666; font-size: 0.875rem;">数据挖掘算法实验平台</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 导航菜单
        st.markdown("### 📚 功能导航")
        
        menu_sections = {
            "算法实验": [
                ("K-Means 聚类", "clustering", "📊"),
                ("决策树", "decision_tree", "🌳"),
                ("KNN 分类", "knn", "📍"),
                ("朴素贝叶斯", "naive_bayes", "📊"),
                ("Apriori 关联", "apriori", "🔗"),
                ("PCA 降维", "pca", "🎯"),
                ("回归分析", "regression", "📈"),
            ],
            "数据管理": [
                ("数据总览", "data_overview", "📁"),
            ],
            "实验工具": [
                ("实验报告", "report", "📝"),
            ]
        }
        
        # 初始化页面状态
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'home'
        
        # 渲染菜单
        for section, items in menu_sections.items():
            st.markdown(f"**{section}**")
            
            for name, page_id, icon in items:
                # 检查是否有实验结果
                has_result = 'experiment_results' in st.session_state and st.session_state.experiment_results
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(icon)
                with col2:
                    if st.button(
                        name,
                        key=f"nav_{page_id}",
                        use_container_width=True,
                        type="primary" if st.session_state.current_page == page_id else "secondary"
                    ):
                        st.session_state.current_page = page_id
                        st.rerun()
            
            st.markdown("---")
        
        # AI 助手入口
        st.markdown("### 🤖 AI 实验导师")
        
        if st.button("💬 打开 AI 助手", use_container_width=True):
            st.session_state.current_page = 'ai_assistant'
            st.rerun()
        
        # Coze API 设置
        st.markdown("---")
        st.markdown("#### ⚙️ 设置")
        
        with st.expander("API 配置", expanded=False):
            api_key = st.text_input(
                "Coze API Key",
                type="password",
                key="api_key_input",
                help="输入您的 Coze API Key"
            )
            
            if api_key:
                st.session_state.coze_api_key = api_key
                st.success("✅ 已配置")
        
        # 版本信息
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #999; font-size: 0.75rem;">
            <p>Mining-Lab v1.0</p>
            <p>© 2024 数据挖掘课程组</p>
        </div>
        """, unsafe_allow_html=True)


def render_home_page():
    """渲染首页"""
    
    # 欢迎横幅
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    ">
        <h1 style="margin: 0; font-size: 2.5rem;">🧪 Mining-Lab</h1>
        <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0;">
            面向《数据仓库与数据挖掘》课程的交互式实验平台
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 快速开始
    st.markdown("### 🚀 快速开始")
    
    col1, col2, col3, col4 = st.columns(4)
    
    quick_actions = [
        ("📊", "K-Means 聚类", "clustering", "快速体验聚类算法"),
        ("🌳", "决策树", "decision_tree", "可视化决策过程"),
        ("🔗", "Apriori 关联", "apriori", "挖掘数据关联规则"),
        ("🎯", "PCA 降维", "pca", "主成分分析可视化"),
    ]
    
    for col, (icon, title, page_id, desc) in zip(
        [col1, col2, col3, col4],
        quick_actions
    ):
        with col:
            if st.button(f"{icon}\n\n**{title}**\n\n{desc}", use_container_width=True, key=f"quick_{page_id}"):
                st.session_state.current_page = page_id
                st.rerun()
    
    # 平台功能
    st.markdown("---")
    st.markdown("### 🎯 平台功能")
    
    features = [
        {
            "icon": "📊",
            "title": "算法实验",
            "description": "K-Means、决策树、朴素贝叶斯、KNN、Apriori、PCA、回归分析等经典算法",
            "color": "#4ECDC4"
        },
        {
            "icon": "🤖",
            "title": "AI 助手",
            "description": "Coze 智能体集成，24小时在线答疑，提供算法原理讲解和结果解读",
            "color": "#FF6B6B"
        },
        {
            "icon": "📁",
            "title": "数据管理",
            "description": "内置经典数据集，支持 CSV 上传，数据预览和统计描述",
            "color": "#45B7D1"
        },
        {
            "icon": "📝",
            "title": "实验报告",
            "description": "一键生成 Markdown 格式报告，包含参数、数据、图表和结论",
            "color": "#96CEB4"
        },
        {
            "icon": "📈",
            "title": "可视化",
            "description": "Plotly 交互式图表，支持缩放、旋转和导出",
            "color": "#FFEAA7"
        },
        {
            "icon": "🔧",
            "title": "参数调优",
            "description": "实时调节算法参数，即时查看效果变化",
            "color": "#DDA0DD"
        },
    ]
    
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div style="
                background: white;
                padding: 1.5rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                border-left: 4px solid {feature['color']};
                margin-bottom: 1rem;
            ">
                <h4 style="margin: 0 0 0.5rem 0;">{feature['icon']} {feature['title']}</h4>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 内置数据集
    st.markdown("---")
    st.markdown("### 📦 内置数据集")
    
    dataset_cols = st.columns(len(DATASET_INFO))
    
    for i, (ds_id, ds_info) in enumerate(DATASET_INFO.items()):
        with dataset_cols[i]:
            st.markdown(f"""
            <div style="
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                text-align: center;
            ">
                <h5 style="margin: 0;">{ds_info['name']}</h5>
                <p style="color: #666; font-size: 0.8rem; margin: 0.5rem 0;">
                    样本: {ds_info['n_samples']} | 特征: {ds_info['n_features']}
                </p>
                <p style="color: #999; font-size: 0.75rem; margin: 0;">
                    {', '.join(ds_info['recommended_algorithms'][:2])}
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # 使用统计
    if 'experiment_count' not in st.session_state:
        st.session_state.experiment_count = 0
    
    st.markdown("---")
    st.markdown("### 📊 使用统计")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("实验次数", st.session_state.experiment_count)
    col2.metric("数据集", len(DATASET_INFO))
    col3.metric("算法模块", 7)


def render_data_overview_page():
    """渲染数据总览页面"""
    
    st.markdown("## 📁 数据管理")
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📚 内置数据集", "📤 上传数据"])
    
    with tab1:
        st.markdown("### 内置经典数据集")
        
        # 数据集选择
        selected_dataset = st.selectbox(
            "选择数据集",
            options=list(DATASET_INFO.keys()),
            format_func=lambda x: DATASET_INFO[x]['name']
        )
        
        ds_info = DATASET_INFO[selected_dataset]
        
        # 数据集信息卡片
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("数据集名称", ds_info['name'])
        col2.metric("样本数", ds_info['n_samples'])
        col3.metric("特征数", ds_info['n_features'])
        col4.metric("类别数", ds_info['n_classes'] if ds_info['n_classes'] else "N/A")
        
        st.markdown(f"**描述**: {ds_info['description']}")
        st.markdown(f"**推荐算法**: {', '.join(ds_info['recommended_algorithms'])}")
        
        # 加载并显示数据
        from 工具.data_loader import load_builtin_dataset
        
        df = load_builtin_dataset(selected_dataset)
        
        st.markdown("#### 数据预览")
        st.dataframe(df.head(10))
        
        st.markdown("#### 数据统计")
        st.dataframe(df.describe())
        
        st.markdown("#### 数据类型")
        dtype_df = pd.DataFrame({
            '列名': df.columns,
            '数据类型': df.dtypes.astype(str),
            '非空数': df.count(),
            '缺失值': df.isnull().sum()
        })
        st.dataframe(dtype_df)
    
    with tab2:
        st.markdown("### 上传自定义数据")
        
        uploaded_file = st.file_uploader(
            "选择 CSV 文件",
            type=['csv'],
            help="支持 UTF-8、GBK 等编码的 CSV 文件"
        )
        
        if uploaded_file:
            df = load_csv_file(uploaded_file)
            
            if df is not None:
                st.success(f"✅ 成功加载数据: {df.shape[0]} 行 × {df.shape[1]} 列")
                
                st.markdown("#### 数据预览")
                st.dataframe(df.head(10))
                
                st.markdown("#### 数据统计")
                st.dataframe(df.describe())
                
                # 保存到 session
                st.session_state.custom_data = df
                st.session_state.current_dataset = {
                    'name': uploaded_file.name,
                    'shape': df.shape
                }
        
        st.markdown("---")
        st.markdown("#### 📋 CSV 文件格式要求")
        st.markdown("""
        1. **第一行**应为表头（列名）
        2. **编码格式**：UTF-8 或 GBK
        3. **文件大小**：建议小于 50MB
        4. **缺失值**：支持空值或特定标记（如 NA、NULL）
        
        **分类数据格式示例：**
        | feature1 | feature2 | target |
        |----------|----------|--------|
        | 5.1      | 3.5      | 0      |
        | 4.9      | 3.0      | 1      |
        """)


def render_report_page():
    """渲染实验报告页面"""
    
    st.markdown("## 📝 实验报告")
    st.markdown("---")
    
    if 'experiment_results' not in st.session_state or not st.session_state.experiment_results:
        st.info("📋 请先完成一个算法实验，系统将自动保存实验结果。")
        
        # 显示报告模板
        st.markdown("### 📄 报告模板预览")
        
        template = """
        # 实验报告
        
        ## 实验信息
        - **日期**: {date}
        - **算法**: -
        - **数据集**: -
        
        ## 参数配置
        ```
        # 算法参数
        ```
        
        ## 数据摘要
        - 样本数:
        - 特征数:
        - 类别数:
        
        ## 实验结果
        ### 评估指标
        | 指标 | 值 |
        |------|-----|
        |      |     |
        
        ### 可视化图表
        [图表将自动插入]
        
        ## 分析结论
        > 关键发现和结论...
        
        ## 参考代码
        ```python
        # 核心代码
        ```
        """
        
        st.markdown(template.format(date=datetime.now().strftime("%Y-%m-%d")))
        return
    
    # 生成报告
    results = st.session_state.experiment_results
    
    # 基本信息
    col1, col2 = st.columns(2)
    with col1:
        algorithm = results.get('algorithm', '未知')
        st.info(f"📋 当前报告: **{algorithm}** 算法实验报告")
    
    with col2:
        if st.button("📥 导出报告 (Markdown)"):
            report_content = generate_report_content()
            
            # 创建下载链接
            b64 = base64.b64encode(report_content.encode()).decode()
            href = f'<a href="data:file/markdown;base64,{b64}" download="实验报告_{algorithm}_{datetime.now().strftime("%Y%m%d")}.md">点击下载</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 报告内容
    report_html = generate_report_html()
    st.markdown(report_html, unsafe_allow_html=True)
    
    # 下载按钮
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Markdown 格式"):
            report_content = generate_report_content()
            st.download_button(
                label="下载 .md 文件",
                data=report_content,
                file_name=f"实验报告_{algorithm}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )


def generate_report_content() -> str:
    """生成 Markdown 格式报告"""
    
    results = st.session_state.experiment_results
    dataset = st.session_state.get('current_dataset', {})
    algorithm_info = st.session_state.get('current_algorithm', {})
    
    report = f"""# Mining-Lab 实验报告

## 实验信息
- **日期**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **算法**: {algorithm_info.get('name', '未知')}
- **数据集**: {dataset.get('name', '未知')}

## 参数配置
"""
    
    if 'params' in algorithm_info:
        for key, value in algorithm_info['params'].items():
            report += f"- **{key}**: {value}\n"
    
    report += f"""
## 数据摘要
- **样本数**: {dataset.get('shape', (0, 0))[0]}
- **特征数**: {dataset.get('shape', (0, 0))[1]}

## 实验结果

### 评估指标
"""
    
    if 'metrics' in results:
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                report += f"- **{key}**: {value:.4f}\n"
            else:
                report += f"- **{key}**: {value}\n"
    
    report += """
## 分析结论

"""
    
    if 'summary' in results:
        report += results['summary']
    
    report += """

---

*由 Mining-Lab 数据挖掘实验平台自动生成*
"""
    
    return report


def generate_report_html() -> str:
    """生成 HTML 格式报告预览"""
    
    results = st.session_state.experiment_results
    dataset = st.session_state.get('current_dataset', {})
    algorithm_info = st.session_state.get('current_algorithm', {})
    
    html = f"""
    <div style="background: white; padding: 2rem; border-radius: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <h2 style="color: #2E86AB; border-bottom: 2px solid #eee; padding-bottom: 0.5rem;">
            📋 实验报告
        </h2>
        
        <h3>📅 实验信息</h3>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #f8f9fa;">
                <td style="padding: 0.5rem; border: 1px solid #ddd;"><strong>日期</strong></td>
                <td style="padding: 0.5rem; border: 1px solid #ddd;">{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem; border: 1px solid #ddd;"><strong>算法</strong></td>
                <td style="padding: 0.5rem; border: 1px solid #ddd;">{algorithm_info.get('name', '未知')}</td>
            </tr>
            <tr style="background: #f8f9fa;">
                <td style="padding: 0.5rem; border: 1px solid #ddd;"><strong>数据集</strong></td>
                <td style="padding: 0.5rem; border: 1px solid #ddd;">{dataset.get('name', '未知')}</td>
            </tr>
        </table>
        
        <h3>⚙️ 参数配置</h3>
        <ul>
    """
    
    if 'params' in algorithm_info:
        for key, value in algorithm_info['params'].items():
            html += f"<li><strong>{key}</strong>: {value}</li>\n"
    
    html += """
        </ul>
        
        <h3>📊 评估指标</h3>
        <ul>
    """
    
    if 'metrics' in results:
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                html += f"<li><strong>{key}</strong>: {value:.4f}</li>\n"
            else:
                html += f"<li><strong>{key}</strong>: {value}</li>\n"
    
    html += """
        </ul>
        
        <h3>📝 分析结论</h3>
    """
    
    if 'summary' in results:
        html += f"<blockquote>{results['summary']}</blockquote>\n"
    
    html += """
    </div>
    """
    
    return html


def render_ai_assistant_page():
    """渲染 AI 助手页面"""
    
    st.markdown("## 🤖 AI 实验导师")
    st.markdown("---")
    
    # 渲染 Coze 设置
    api_key, assistant = render_coze_sidebar()
    
    if not api_key:
        st.warning("⚠️ 请先在侧边栏配置 Coze API Key 以启用 AI 助手功能")
        
        st.markdown("""
        ### 💡 如何获取 API Key？
        
        1. 访问 [Coze 控制台](https://console.coze.cn)
        2. 登录或注册账号
        3. 进入「开发者」→「API Key」
        4. 创建新的 API Key 并复制
        
        ### 🎓 导师能力
        
        - **算法讲解**：用通俗语言解释数据挖掘算法原理
        - **参数建议**：根据数据特征推荐最佳参数配置
        - **结果解读**：帮助理解混淆矩阵、聚类结果等
        - **代码指导**：提供 sklearn 等库的代码示例
        - **练习测验**：生成算法相关的练习题
        """)
        
        # 显示建议问题
        if 'current_algorithm' in st.session_state:
            algorithm = st.session_state.current_algorithm.get('name', '')
            suggestions = get_suggested_questions(algorithm)
            
            st.markdown("### 💬 常见问题")
            
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
    else:
        # 渲染聊天界面
        render_chat_interface()


def main():
    """主函数"""
    
    # 初始化 session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    # 渲染侧边栏
    render_sidebar()
    
    # 根据当前页面渲染内容
    page = st.session_state.current_page
    
    if page == 'home':
        render_home_page()
    
    elif page == 'clustering':
        render_kmeans_page()
    
    elif page in ['decision_tree', 'knn', 'naive_bayes']:
        render_classification_subpage(page)
    
    elif page == 'apriori':
        render_apriori_page()
    
    elif page == 'pca':
        render_pca_page()
    
    elif page == 'regression':
        render_regression_page()
    
    elif page == 'data_overview':
        render_data_overview_page()
    
    elif page == 'report':
        render_report_page()
    
    elif page == 'ai_assistant':
        render_ai_assistant_page()
    
    else:
        render_home_page()
    
    # 增加实验计数
    if 'experiment_count' not in st.session_state:
        st.session_state.experiment_count = 0


if __name__ == "__main__":
    import pandas as pd
    main()
