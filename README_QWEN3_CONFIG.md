# 使用通义千问qwen-plus模型配置指南

## 环境配置

### 1. 设置API密钥环境变量

```bash
# 设置通义千问API密钥
export DASHSCOPE_API_KEY="sk-your-api-key-here"

# 或者在.bashrc/.zshrc中添加
echo 'export DASHSCOPE_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 安装依赖

确保已安装必要的Python包：

```bash
pip install openai
```

## 使用方法

### 运行多智能体交通控制系统

```bash
# 使用qwen-plus模型运行多智能体系统
python main.py \
    --llm-path-or-name qwen-plus \
    --batch-size 8 \
    --location Manhattan \
    --step-size 180.0 \
    --max-steps 43200 \
    --multi-agent

# 或者使用其他qwen模型
python main.py \
    --llm-path-or-name qwen-plus \
    --batch-size 16 \
    --location Manhattan \
    --multi-agent

# 关闭反思功能
python main.py \
    --llm-path-or-name qwen-plus \
    --batch-size 8 \
    --no-reflection \
    --multi-agent
```

### 单智能体模式

```bash
python main.py \
    --llm-path-or-name qwen-plus \
    --batch-size 8 \
    --location Manhattan \
    --single-agent
```

## 支持的通义千问模型

- `qwen-plus` - 最新的轻量级快速模型
- `qwen-plus` - 增强版模型
- `qwen-turbo` - 高速模型
- `qwen-max` - 最强性能模型
- `qwen3-coder` - 代码专用模型

## 模型配置参数

系统自动使用以下配置参数：

```python
generation_kwargs = {
    "temperature": 0.1,      # 生成多样性控制
    "top_p": 1.0,            # 核采样参数
    "max_tokens": 8192,      # 最大token数
}
```

## API调用示例

系统内部使用OpenAI兼容接口：

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "You are a helpful traffic coordinator."},
        {"role": "user", "content": "Plan optimal routes for traffic coordination."}
    ],
    temperature=0.1,
    max_tokens=8192
)
```

## 系统架构特点

使用qwen-plus的多智能体交通控制系统包含：

### 🤖 LLM驱动的智能决策
- **区域智能体**: 使用`regional_coordination_decision()`进行区域内路径规划
- **交通智能体**: 使用`macro_route_planning()`进行跨区域宏观规划
- **智能体通信**: 使用`inter_agent_communication()`进行协调

### 🌐 多层级协调
- **20个区域**并行处理
- **465个边界边**智能负载均衡
- **多时间窗口预测**（180s-720s）
- **异步决策机制**

### 📊 性能监控
- 实时交通状态监控
- LLM调用日志记录
- 系统性能指标追踪
- WandB实验跟踪

## 故障排除

### 1. API密钥错误
```bash
# 检查环境变量
echo $DASHSCOPE_API_KEY

# 确保密钥格式正确 (sk-开头)
```

### 2. 网络连接问题
```bash
# 测试API连接
curl -H "Authorization: Bearer $DASHSCOPE_API_KEY" \
     -H "Content-Type: application/json" \
     https://dashscope.aliyuncs.com/compatible-mode/v1/models
```

### 3. 模型不可用
- 检查模型名称是否正确
- 确认账户是否有访问权限
- 查看阿里云控制台配额

## 性能优化建议

1. **批次大小调整**: 根据内存和API限制调整`--batch-size`
2. **并发控制**: 系统使用4个线程池处理区域决策
3. **缓存策略**: LLM响应自动缓存以减少重复调用
4. **错误重试**: 内置3次重试机制保证鲁棒性

## 日志和监控

系统日志保存在 `logs/` 目录下：
```
logs/
├── Manhattan_qwen-plus/
│   ├── agent_logs.json
│   ├── llm_calls.json
│   ├── system_performance.json
│   └── vehicle_tracking.json
```

通过WandB可视化查看实验结果和性能指标。