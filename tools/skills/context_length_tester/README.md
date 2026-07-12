# Context Length Tester Skill

## 概述

测试 LLM 代理服务（OpenAI 兼容 API）的实际上下文长度。

支持两种测试模式：
1. **二分查找法 (probe)**：快速定位 API 能接受的最大 token 数
2. **大海捞针法 (needle)**：通过 Needle In A Haystack 测试模型实际能处理的上下文长度

## 使用方法

### 快速开始

```bash
# 大海捞针测试（推荐）
uv run tools/skills/context_length_tester/context_length_tester.py --model deepseek-v3

# 二分查找 API 上限
uv run tools/skills/context_length_tester/context_length_tester.py --model deepseek-v3 --mode probe

# 自定义参数
uv run tools/skills/context_length_tester/context_length_tester.py --model deepseek-v3 \
  --start 100000 --step 20000 --depths 0.0,0.25,0.5,0.75,1.0
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | 模型名称，如 deepseek-v3, gpt-4.1 |
| `--mode` | needle | 测试模式: probe 或 needle |
| `--api-url` | http://localhost:6001/v1/chat/completions | API 端点 |
| `--api-key` | sk-test | API 密钥 |
| `--start` | 100000 | needle 模式起始 token 数 |
| `--max` | 200000 | 最大测试 token 数 |
| `--step` | 20000 | needle 模式步长 |
| `--depths` | 0.0,0.5,1.0 | 测试深度（逗号分隔） |

### 环境变量

```bash
export OPENAI_API_KEY="sk-test"
export OPENAI_API_BASE="http://localhost:6001/v1"
```

## 测试模式详解

### 1. Probe 模式（二分查找）

快速测试 API 能接受的最大请求长度：

```bash
uv run context_length_tester.py --model deepseek-v3 --mode probe --max 200000
```

**特点**：
- 速度快（约 2-3 分钟）
- 只能测出 API 的上限，不能验证模型是否真读到了内容
- 适合快速了解服务配置

### 2. Needle 模式（大海捞针）

实际测试模型能处理的上下文长度：

```bash
uv run context_length_tester.py --model deepseek-v3 --mode needle
```

**原理**：
1. 构建一个长文本（haystack）
2. 在文本不同深度插入关键信息（needle）
3. 询问模型关于 needle 的问题
4. 如果模型能准确回答，说明它确实读到了那个位置

**特点**：
- 更准确但耗时较长（约 10-30 分钟）
- 使用随机密钥避免缓存干扰
- 支持多深度测试（开头、中间、结尾）
- 自动精细测试定位精确上限

## 示例输出

```
================================================================================
大海捞针测试 - 模型: deepseek-v3
范围: 100000 ~ 200000 tokens, 步长: 20000
测试深度: (0.0, 0.5, 1.0)
================================================================================

============================================================
当前测试: 100000 tokens | 密钥: KEY-37103-9978
============================================================
实际长度: 100014 tokens

  深度   0.0%... ✓ PASS (16.8s) - KEY-37103-9978
  深度  50.0%... ✓ PASS (16.8s) - KEY-37103-9978
  深度 100.0%... ✓ PASS (17.0s) - KEY-37103-9978

============================================================
当前测试: 120000 tokens | 密钥: KEY-17865-4421
============================================================
实际长度: 120034 tokens

  深度   0.0%... ✓ PASS (21.3s) - KEY-17865-4421
  深度  50.0%... ✓ PASS (21.2s) - KEY-17865-4421
  深度 100.0%... ✗ FAIL (22.7s) - 

============================================================
模型在 120034 tokens 时开始丢失信息
============================================================

================================================================================
模型 deepseek-v3 的实测上下文长度约为: 100014 tokens
================================================================================
```

## 注意事项

1. **随机密钥**：每次测试使用随机密钥，避免模型缓存
2. **超时设置**：needle 测试使用 180 秒超时，大文本推理可能较慢
3. **深度选择**：建议至少测试 0%（开头）、50%（中间）、100%（末尾）三个深度
4. **网络稳定**：确保 API 服务稳定，避免因网络问题导致误判

## 在代码中使用

```python
from context_length_tester import ContextLengthTester

tester = ContextLengthTester(
    api_url="http://localhost:6001/v1/chat/completions",
    api_key="sk-test",
    model="deepseek-v3"
)

# 二分查找
max_tokens = tester.probe_context_size(low=1000, high=200000)

# 大海捞针
max_tokens = tester.needle_test(
    start_tokens=100000,
    max_tokens=200000,
    step_tokens=20000,
    depth_ratios=(0.0, 0.5, 1.0)
)
```

## 依赖

- requests
- tiktoken

安装：
```bash
uv add requests tiktoken
```
