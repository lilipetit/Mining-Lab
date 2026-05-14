# -*- coding: utf-8 -*-
"""
Coze API 集成模块
提供与 Coze 智能体的对话功能
"""

import requests
import json
import streamlit as st
from datetime import datetime
from typing import Optional, Generator
import time

# Coze API 配置
COZE_API_BASE = "https://api.coze.cn"
COZE_API_VERSION = "v3"


class CozeAssistant:
    """Coze AI 助手类"""
    
    def __init__(self, api_key: str, bot_id: str = None):
        """
        初始化 Coze 助手
        
        Args:
            api_key: Coze API 密钥
            bot_id: 机器人 ID（可选）
        """
        self.api_key = api_key
        self.bot_id = bot_id
        self.session_history = []
    
    def _get_headers(self) -> dict:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def chat(self, message: str, user_id: str = "default_user", 
             stream: bool = False) -> dict:
        """
        发送聊天消息
        
        Args:
            message: 用户消息
            user_id: 用户ID
            stream: 是否流式响应
            
        Returns:
            dict: API 响应
        """
        url = f"{COZE_API_BASE}/{COZE_API_VERSION}/chat"
        
        payload = {
            "model": "moonshot-v1-8k",  # 默认模型
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ],
            "user_id": user_id
        }
        
        if self.bot_id:
            payload["bot_id"] = self.bot_id
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": True,
                    "message": f"API 请求失败: {response.status_code}",
                    "details": response.text
                }
                
        except requests.exceptions.Timeout:
            return {
                "error": True,
                "message": "请求超时，请稍后重试"
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"网络错误: {str(e)}"
            }
    
    def chat_with_conversation(self, message: str, user_id: str = "default_user") -> dict:
        """
        带对话历史的聊天
        
        Args:
            message: 用户消息
            user_id: 用户ID
            
        Returns:
            dict: API 响应
        """
        # 添加用户消息到历史
        self.session_history.append({
            "role": "user",
            "content": message
        })
        
        url = f"{COZE_API_BASE}/{COZE_API_VERSION}/chat"
        
        payload = {
            "model": "moonshot-v1-8k",
            "messages": self.session_history.copy(),
            "user_id": user_id
        }
        
        if self.bot_id:
            payload["bot_id"] = self.bot_id
        
        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 提取助手的回复并添加到历史
                if result.get("data") and result["data"].get("messages"):
                    for msg in result["data"]["messages"]:
                        if msg.get("role") == "assistant":
                            self.session_history.append({
                                "role": "assistant",
                                "content": msg.get("content", "")
                            })
                            break
                
                return result
            else:
                return {
                    "error": True,
                    "message": f"API 请求失败: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            return {
                "error": True,
                "message": f"错误: {str(e)}"
            }
    
    def get_response_content(self, chat_result: dict) -> str:
        """
        从聊天结果中提取回复内容
        
        Args:
            chat_result: chat() 返回的结果
            
        Returns:
            str: 回复内容
        """
        if chat_result.get("error"):
            return f"⚠️ {chat_result.get('message', '未知错误')}"
        
        # 轮询获取完整回复（简化处理）
        conversation_id = chat_result.get("data", {}).get("id")
        if not conversation_id:
            return "⚠️ 无法获取对话ID"
        
        # 获取消息
        return self._fetch_messages(conversation_id)
    
    def _fetch_messages(self, conversation_id: str, max_retries: int = 10) -> str:
        """
        获取对话消息
        
        Args:
            conversation_id: 对话ID
            max_retries: 最大重试次数
            
        Returns:
            str: 消息内容
        """
        url = f"{COZE_API_BASE}/{COZE_API_VERSION}/chat/retrieve"
        
        params = {"chat_id": conversation_id}
        
        for i in range(max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    params=params,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("data", {}).get("status")
                    
                    if status == "completed":
                        # 获取消息列表
                        return self._get_messages(conversation_id)
                    elif status == "failed":
                        return "⚠️ 对话生成失败"
                    
                time.sleep(2)
                
            except Exception:
                time.sleep(2)
        
        return "⏳ 等待回复中，请稍后..."
    
    def _get_messages(self, conversation_id: str) -> str:
        """获取对话消息列表"""
        url = f"{COZE_API_BASE}/{COZE_API_VERSION}/chat/message/list"
        
        params = {"chat_id": conversation_id}
        
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                messages = result.get("data", [])
                
                # 提取最后一条助手回复
                for msg in reversed(messages):
                    if msg.get("role") == "assistant" and msg.get("type") == "answer":
                        return msg.get("content", "")
                
                return "⚠️ 未找到有效回复"
            else:
                return "⚠️ 获取消息失败"
                
        except Exception:
            return "⚠️ 获取消息出错"
    
    def clear_history(self):
        """清空对话历史"""
        self.session_history = []


# Streamlit UI 组件
def render_coze_sidebar():
    """
    渲染 Coze 设置侧边栏
    
    Returns:
        tuple: (api_key, assistant实例)
    """
    with st.sidebar:
        st.markdown("### 🤖 AI 实验导师")
        st.markdown("---")
        
        # API Key 设置
        api_key = st.text_input(
            "🔑 Coze API Key",
            type="password",
            help="输入您的 Coze API 密钥",
            placeholder="输入 API Key..."
        )
        
        # 帮助信息
        with st.expander("📖 如何获取 API Key?"):
            st.markdown("""
            1. 访问 [Coze 控制台](https://console.coze.cn)
            2. 登录/注册账号
            3. 进入「开发者」->「API Key」
            4. 创建新的 API Key 并复制
            """)
        
        # Bot ID 设置（可选）
        bot_id = st.text_input(
            "🤖 Bot ID (可选)",
            help="指定使用的机器人ID",
            placeholder="留空使用默认"
        )
        
        # 清空历史按钮
        if st.button("🗑️ 清空对话历史"):
            if 'coze_assistant' in st.session_state:
                st.session_state.coze_assistant.clear_history()
                st.session_state.chat_history = []
                st.success("对话历史已清空")
        
        # 初始化助手
        assistant = None
        if api_key:
            try:
                assistant = CozeAssistant(api_key, bot_id if bot_id else None)
                st.session_state.coze_assistant = assistant
                st.success("✅ API 连接成功")
            except Exception as e:
                st.error(f"初始化失败: {str(e)}")
        
        return api_key, assistant


def render_chat_interface():
    """渲染聊天界面"""
    
    # 初始化聊天历史
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 系统提示词
    SYSTEM_PROMPT = """你是《数据仓库与数据挖掘》课程的智能实验导师。

**能力：**
1. 算法原理解释 - 用通俗语言解释经典数据挖掘算法
2. 参数调优建议 - 根据数据特征推荐最佳参数配置
3. 结果分析解读 - 帮助学生理解实验输出（混淆矩阵、聚类结果等）
4. 编程指导 - 提供 Python 代码示例（sklearn 用法）
5. 练习与测验 - 生成算法相关的练习题

**风格：**
- 耐心、鼓励、启发式教学
- 避免直接给答案，引导学生思考
- 适当使用类比和图示语言
- 结合当前实验上下文回答问题
"""
    
    # 显示聊天历史
    st.markdown("### 💬 对话区域")
    
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style="background-color: #0d47a1; color: white; padding: 15px; border-radius: 15px; margin: 10px 0; text-align: left;">
            <strong>👤 你：</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #1a1a2e; color: white; padding: 15px; border-radius: 15px; margin: 10px 0; text-align: left;">
            <strong>🤖 导师：</strong><br>{msg['content']}
            </div>
            """, unsafe_allow_html=True)
    
    # 快速提问按钮
    st.markdown("#### 快捷问题")
    
    quick_questions = [
        ("解释K-Means", "请用通俗易懂的语言解释K-Means聚类算法的工作原理"),
        ("参数建议", "对于鸢尾花数据集，K-Means聚类应该如何选择K值？"),
        ("结果解读", "如何解读聚类分析中的轮廓系数？"),
        ("代码示例", "请给出一个使用sklearn实现决策树的代码示例"),
        ("生成练习", "出一道关于Apriori关联规则算法的练习题")
    ]
    
    cols = st.columns(2)
    for i, (label, question) in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(label, key=f"quick_{i}"):
                handle_user_message(question)
    
    # 用户输入
    st.markdown("---")
    user_input = st.text_area(
        "输入您的问题：",
        placeholder="例如：决策树如何避免过拟合？",
        height=80,
        key="chat_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        send_button = st.button("📤 发送", type="primary")
    with col2:
        use_context = st.checkbox("包含当前实验上下文", value=True)
    
    if send_button and user_input:
        handle_user_message(user_input, include_context=use_context)


def handle_user_message(message: str, include_context: bool = True):
    """
    处理用户消息
    
    Args:
        message: 用户消息
        include_context: 是否包含当前实验上下文
    """
    # 添加到历史
    st.session_state.chat_history.append({
        "role": "user",
        "content": message
    })
    
    # 获取助手
    if 'coze_assistant' not in st.session_state or st.session_state.coze_assistant is None:
        st.warning("⚠️ 请先在侧边栏配置 Coze API Key")
        return
    
    # 构建上下文
    context = ""
    if include_context:
        context = build_experiment_context()
    
    full_message = f"{context}\n\n用户问题：{message}" if context else message
    
    # 调用 API
    with st.spinner("🤖 导师思考中..."):
        try:
            result = st.session_state.coze_assistant.chat(full_message)
            
            if result.get("error"):
                response = f"⚠️ {result.get('message', '未知错误')}"
            else:
                response = st.session_state.coze_assistant.get_response_content(result)
            
            # 添加到历史
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response
            })
            
            # 刷新页面显示新消息
            st.rerun()
            
        except Exception as e:
            st.error(f"发生错误: {str(e)}")


def build_experiment_context() -> str:
    """
    构建当前实验上下文
    
    Returns:
        str: 上下文字符串
    """
    context_parts = []
    
    # 数据集信息
    if 'current_dataset' in st.session_state:
        ds = st.session_state.current_dataset
        context_parts.append(f"**当前数据集**: {ds.get('name', '未知')}")
        if 'shape' in ds:
            context_parts.append(f"样本数: {ds['shape'][0]}, 特征数: {ds['shape'][1]}")
    
    # 算法信息
    if 'current_algorithm' in st.session_state:
        alg = st.session_state.current_algorithm
        context_parts.append(f"**当前算法**: {alg.get('name', '未知')}")
        if 'params' in alg:
            params_str = ", ".join([f"{k}={v}" for k, v in alg['params'].items()])
            context_parts.append(f"参数配置: {params_str}")
    
    # 实验结果
    if 'experiment_results' in st.session_state:
        results = st.session_state.experiment_results
        context_parts.append(f"**实验结果**: {results.get('summary', '')}")
    
    if context_parts:
        return "【当前实验上下文】\n" + "\n".join(context_parts) + "\n"
    
    return ""


def get_suggested_questions(algorithm: str) -> list:
    """
    根据当前算法返回建议问题
    
    Args:
        algorithm: 算法名称
        
    Returns:
        list: 问题列表
    """
    suggestions = {
        'K-Means': [
            "K-Means如何处理噪声和异常值？",
            "肘部法则的原理是什么？",
            "K-Means与DBSCAN有什么区别？"
        ],
        '决策树': [
            "如何避免决策树过拟合？",
            "剪枝策略有哪些优缺点？",
            "特征重要性是如何计算的？"
        ],
        '朴素贝叶斯': [
            "朴素贝叶斯的'朴素'是什么意思？",
            "GaussianNB和MultinomialNB的区别是什么？",
            "拉普拉斯平滑的作用是什么？"
        ],
        'Apriori': [
            "支持度和置信度有什么区别？",
            "如何从关联规则中提取有价值的商业洞察？",
            "Apriori算法的优缺点是什么？"
        ],
        'KNN': [
            "K值选择对结果有什么影响？",
            "KNN如何处理特征尺度不一致的问题？",
            "KNN的时间复杂度是多少？"
        ],
        'PCA': [
            "PCA降维后如何解释主成分的含义？",
            "保留多少主成分比较合适？",
            "PCA和t-SNE有什么区别？"
        ],
        '回归': [
            "如何判断回归模型是否合适？",
            "线性回归的假设条件有哪些？",
            "过拟合和欠拟合如何处理？"
        ]
    }
    
    return suggestions.get(algorithm, [
        "这个算法的原理是什么？",
        "如何选择合适的参数？",
        "如何评估实验结果？"
    ])
