# Mining-Lab 部署指南

本文档详细介绍如何将 Mining-Lab 部署到 Streamlit Cloud，实现永久可访问的在线实验平台。

## 部署方式

### 方式一：Streamlit Cloud（一键部署）

#### 前置条件
1. GitHub 账号
2. 代码已推送到 GitHub 仓库
3. GitHub 账号已绑定 Streamlit Cloud

#### 部署步骤

1. **准备 GitHub 仓库**
   ```bash
   cd Mining-Lab
   git init
   git add .
   git commit -m "Initial commit: Mining-Lab v1.0"
   git branch -M main
   git remote add origin https://github.com/yourusername/Mining-Lab.git
   git push -u origin main
   ```

2. **登录 Streamlit Cloud**
   - 访问 [share.streamlit.io](https://share.streamlit.io)
   - 使用 GitHub 账号登录

3. **创建新应用**
   - 点击 "New app"
   - 选择您的 GitHub 仓库
   - 选择分支（main）
   - 设置主文件路径（app.py）
   - 点击 "Deploy!"

4. **自定义域名（可选）**
   - 在 Settings 中配置自定义域名
   - 支持 HTTPS 自动配置

#### 注意事项
- Streamlit Cloud 免费版限制：1个活跃应用、CPU 1核、800MB 内存
- 每月100小时使用限制
- 应用休眠：24小时无访问后休眠，首次访问会冷启动

### 方式二：Docker 部署

适用于有服务器的用户。

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

构建和运行：
```bash
docker build -t mining-lab .
docker run -p 8501:8501 mining-lab
```

### 方式三：传统服务器部署

```bash
# Ubuntu 20.04 示例
sudo apt update
sudo apt install python3.10-venv python3-pip

# 克隆代码
git clone https://github.com/yourusername/Mining-Lab.git
cd Mining-Lab

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 使用 systemd 运行
sudo nano /etc/systemd/system/mining-lab.service
```

服务配置：
```ini
[Unit]
Description=Mining-Lab Streamlit App
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/mining-lab
ExecStart=/opt/mining-lab/venv/bin/streamlit run app.py --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable mining-lab
sudo systemctl start mining-lab
```

## 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| COZE_API_KEY | Coze API 密钥 | 用户输入 |
| STREAMLIT_SERVER_PORT | 服务端口 | 8501 |
| STREAMLIT_SERVER_HEADLESS | 无头模式 | true |

## 性能优化

### 1. 缓存策略
使用 `@st.cache_data` 装饰器缓存数据加载：
```python
@st.cache_data
def load_data(dataset_name):
    # 耗时操作
    return data
```

### 2. 大数据处理
- 限制最大数据行数
- 使用分页加载
- 启用实验模式跳过耗时计算

### 3. 并行计算
```python
import concurrent.futures

def parallel_compute(data, func):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(func, data)
    return list(results)
```

## 故障排查

### 常见问题

1. **应用无法启动**
   - 检查 requirements.txt 语法
   - 查看 Streamlit Cloud 日志
   - 确认 Python 版本兼容性

2. **内存超限**
   - 减少内置数据集大小
   - 优化图片缓存
   - 使用下采样

3. **Coze API 连接失败**
   - 确认 API Key 正确
   - 检查网络策略
   - 查看 API 额度

### 日志查看
```bash
# 本地日志
streamlit run app.py --logger.level=debug

# Docker 日志
docker logs -f mining-lab
```

## 监控与维护

### 健康检查
```bash
curl http://localhost:8501/_stcore/health
```

### 自动备份
建议使用 GitHub Actions 定时备份数据。

## 联系方式

如部署过程中遇到问题，请提交 GitHub Issue。
