# -*- coding: utf-8 -*-
"""
数据加载模块
提供内置数据集加载和CSV上传功能
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
import streamlit as st
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


@st.cache_data
def load_builtin_dataset(dataset_name: str) -> pd.DataFrame:
    """
    加载内置数据集
    
    Args:
        dataset_name: 数据集名称
        
    Returns:
        pd.DataFrame: 数据集
    """
    datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
        'diabetes': load_diabetes
    }
    
    if dataset_name not in datasets:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    loader = datasets[dataset_name]
    data = loader()
    
    # 转换为DataFrame
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    
    # 添加目标类别名称
    if dataset_name == 'iris':
        df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    elif dataset_name == 'wine':
        df['target_name'] = df['target'].map({i: f'class_{i}' for i in range(data.target_names.shape[0])})
    elif dataset_name == 'breast_cancer':
        df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})
    
    return df


def load_csv_file(uploaded_file) -> pd.DataFrame:
    """
    加载用户上传的CSV文件
    
    Args:
        uploaded_file: Streamlit上传的文件对象
        
    Returns:
        pd.DataFrame: 数据集
    """
    try:
        # 尝试不同编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin1']:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding)
                return df
            except UnicodeDecodeError:
                continue
        
        # 如果都失败，尝试二进制读取
        uploaded_file.seek(0)
        df = pd.read_csv(BytesIO(uploaded_file.read()))
        return df
        
    except Exception as e:
        st.error(f"文件读取失败: {str(e)}")
        return None


def get_dataset_info(df: pd.DataFrame) -> dict:
    """
    获取数据集基本信息
    
    Args:
        df: 数据集
        
    Returns:
        dict: 数据集信息
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    # 数值统计
    info['describe'] = df.describe().to_dict()
    
    return info


def generate_shopping_basket_data(n_transactions: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    生成模拟购物篮数据（用于Apriori算法演示）
    
    Args:
        n_transactions: 事务数量
        seed: 随机种子
        
    Returns:
        pd.DataFrame: 购物篮数据（布尔型矩阵）
    """
    np.random.seed(seed)
    
    # 定义商品类别和常见商品
    items = [
        '面包', '牛奶', '鸡蛋', '奶酪', '苹果', '香蕉', '橙汁', '咖啡',
        '薯片', '饼干', '巧克力', '可乐', '啤酒', '葡萄酒', '牛肉', '鸡肉',
        '鱼', '大米', '面条', '番茄酱', '沙拉', '酸奶', '黄油', '果酱',
        '洗衣粉', '洗发水', '牙膏', '肥皂', '纸巾', '牛奶'
    ]
    
    # 定义商品共现概率（模拟真实购物行为）
    # 相关商品放在一起增加共现概率
    related_groups = [
        ['面包', '黄油', '果酱'],  # 早餐类
        ['牛奶', '鸡蛋', '奶酪'],  # 乳制品
        ['牛肉', '番茄酱', '面条'],  # 晚餐
        ['薯片', '可乐', '啤酒'],  # 零食饮料
        ['洗发水', '肥皂', '牙膏'],  # 日用品
    ]
    
    # 生成事务
    transactions = []
    
    for _ in range(n_transactions):
        transaction = np.random.rand(len(items)) < 0.08  # 基础购买概率
        
        # 确保每个相关组至少有一件商品被购买（增加共现）
        for group in related_groups:
            if np.random.rand() < 0.4:  # 40%概率购买某类商品
                group_indices = [items.index(item) for item in group if item in items]
                for idx in group_indices:
                    if np.random.rand() < 0.7:  # 该组内70%购买率
                        transaction[idx] = True
        
        transactions.append(transaction)
    
    # 转换为DataFrame
    df = pd.DataFrame(transactions, columns=items)
    
    return df


def prepare_classification_data(df: pd.DataFrame, target_col: str) -> tuple:
    """
    准备分类数据
    
    Args:
        df: 原始数据
        target_col: 目标列名
        
    Returns:
        tuple: (特征, 目标, 特征名称)
    """
    # 分离特征和目标
    X = df.drop(columns=[target_col, 'target_name'] if 'target_name' in df.columns else [target_col])
    y = df[target_col]
    
    # 只保留数值型特征
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    
    return X.values, y.values, list(X.columns)


def prepare_clustering_data(df: pd.DataFrame, n_features: int = 4) -> np.ndarray:
    """
    准备聚类数据（标准化）
    
    Args:
        df: 数据集
        n_features: 使用的特征数量
        
    Returns:
        np.ndarray: 标准化后的数据
    """
    from sklearn.preprocessing import StandardScaler
    
    # 选择数值特征
    numeric_df = df.select_dtypes(include=[np.number])
    
    # 排除target列
    if 'target' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['target'])
    
    # 选择前n_features个特征
    X = numeric_df.iloc[:, :n_features].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled


def sample_data(df: pd.DataFrame, n_samples: int = None, method: str = 'random') -> pd.DataFrame:
    """
    数据采样
    
    Args:
        df: 原始数据
        n_samples: 采样数量
        method: 采样方法 ('random' 或 'stratified')
        
    Returns:
        pd.DataFrame: 采样后的数据
    """
    if n_samples is None or n_samples >= len(df):
        return df
    
    if method == 'stratified' and 'target' in df.columns:
        # 分层采样
        sampled_dfs = []
        for label in df['target'].unique():
            subset = df[df['target'] == label]
            n = min(n_samples // len(df['target'].unique()), len(subset))
            sampled_dfs.append(subset.sample(n=n, random_state=42))
        return pd.concat(sampled_dfs, ignore_index=True)
    else:
        # 随机采样
        return df.sample(n=n_samples, random_state=42)


# 数据集元信息
DATASET_INFO = {
    'iris': {
        'name': '鸢尾花数据集',
        'name_en': 'Iris Dataset',
        'n_samples': 150,
        'n_features': 4,
        'n_classes': 3,
        'description': '经典的植物分类数据集，包含3种鸢尾花的4个测量特征。',
        'recommended_algorithms': ['K-Means', '决策树', 'KNN', 'PCA']
    },
    'wine': {
        'name': '酒馆数据集',
        'name_en': 'Wine Dataset',
        'n_samples': 178,
        'n_features': 13,
        'n_classes': 3,
        'description': '意大利葡萄酒的化学分析数据，用于葡萄酒分类。',
        'recommended_algorithms': ['K-Means', 'PCA', '决策树']
    },
    'breast_cancer': {
        'name': '乳腺癌数据集',
        'name_en': 'Breast Cancer Dataset',
        'n_samples': 569,
        'n_features': 30,
        'n_classes': 2,
        'description': '乳腺肿瘤细胞核特征的医学数据集，用于良/恶性分类。',
        'recommended_algorithms': ['决策树', 'KNN', '朴素贝叶斯']
    },
    'diabetes': {
        'name': '糖尿病数据集',
        'name_en': 'Diabetes Dataset',
        'n_samples': 768,
        'n_features': 8,
        'n_classes': 2,
        'description': '印第安人糖尿病数据集，包含患者健康指标。',
        'recommended_algorithms': ['逻辑回归', 'KNN', '决策树']
    },
    'shopping_basket': {
        'name': '购物篮数据',
        'name_en': 'Shopping Basket Data',
        'n_samples': 1000,
        'n_features': 30,
        'n_classes': None,
        'description': '模拟超市购物交易数据，用于关联规则挖掘。',
        'recommended_algorithms': ['Apriori']
    }
}
