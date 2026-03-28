# --tool-call-parser 参数测试

本目录包含针对 SGLang `--tool-call-parser` 参数的分析文档和测试脚本。

## 文件说明

### 1. tool_call_parser_analysis.md
详细的分析文档，包含：
- 使用场景分析
- 实现原理分析
- 数据流和处理逻辑
- 配置参数说明
- 测试用例设计

### 2. test_tool_call_parser.py
完整的测试脚本，包含 5 个测试用例：
- Llama3 Parser 基础功能测试
- Pythonic Parser 流式输出测试
- Tool Choice Required 约束测试
- 多轮对话工具调用测试
- 无效 Parser 类型错误处理测试

## 使用方法

### 环境要求
- SGLang 框架已安装
- OpenAI Python SDK: `pip install openai`
- NPU 硬件支持
- Llama-3.2-1B-Instruct 模型已下载

### 运行测试

#### 方法 1：使用默认模型路径
```bash
python test_tool_call_parser.py
```

#### 方法 2：指定模型路径
```bash
python test_tool_call_parser.py /path/to/your/model
```

#### 方法 3：通过环境变量设置
```bash
export SGLANG_TEST_MODEL=/path/to/your/model
python test_tool_call_parser.py
```

### 默认模型路径
```
/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct
```

## 测试覆盖范围

| 测试用例 | 测试目的 | 测试环境 |
|---------|---------|---------|
| Llama3 Parser 基础功能 | 验证 llama3 parser 能够正确解析工具调用输出) | 单张 NPU 卡 |
| Pythonic Parser 流式输出 | 验证 pythonic parser 在流式输出模式下的解析能力 | 单张 NPU 卡 |
| Tool Choice Required 约束 | 验证 tool_choice=required 时强制工具调用的约束生效 | 单张 NPU 卡 |
| 多轮对话工具调用 | 验证在多轮对话中工具调用的正确性 | 单张 NPU 卡 |
| 无效 Parser 类型处理 | 验证传入无效 parser 类型时的错误处理 | 单张 NPU 卡 |

## 测试结果示例

```
================================================================================
开始 --tool-call-parser 参数测试
================================================================================
模型路径: /root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct
服务地址: http://127.0.0.1:30000

================================================================================
测试用例 1：Llama3 Parser 基础功能测试
================================================================================
启动服务，参数: ...
服务启动成功 (耗时 15 秒)
✓ 成功：工具调用解析正确
  - 工具名称: add
  - 参数: {"a": 3, "b": 5}

================================================================================
测试结果汇总
================================================================================
✓ 通过 - Llama3 Parser 基础功能
✓ 通过 - Pythonic Parser 流式输出
✓ 通过 - Tool Choice Required 约束
✓ 通过 - 多轮对话工具调用
✓ 通过 - 无效 Parser 类型处理

--------------------------------------------------------------------------------
总计: 5/5 通过
================================================================================
```

## 注意事项

1. **端口占用**：测试使用端口 30000 和 30001，确保这些端口未被占用
2. **模型路径**：确保模型路径正确且可访问
3. **NPU 环境**：测试必须在 NPU 环境下运行
4. **测试时长**：完整测试可能需要 10-20 分钟
5. **内存要求**：确保有足够的内存加载模型

## 故障排查

### 问题：服务启动超时
- 检查模型路径是否正确
- 检查 NPU 驱动是否正常
- 检查端口是否被占用
- 查看日志输出

### 问题：测试失败
- 确认 SGLang 版本是否支持 tool-call-parser
- 检查 OpenAI SDK 版本
- 查看详细的错误信息

### 问题：模型未生成工具调用
- 这是正常现象，小模型可能不总是生成工具调用
- 测试脚本会处理这种情况

## 扩展测试

如需添加更多测试用例，可以参考现有测试用例的结构：

```python
def test_your_custom_test(self) -> bool:
    """你的测试用例"""
    print("\n" + "="*80)
    print("测试用例：你的测试名称")
    print("="*80)

    if not self.start_server("parser_type"):
        return False

    try:
        # 你的测试逻辑
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        self.stop_server()
```

然后在 `run_all_tests` 方法中添加你的测试：

```python
tests = [
    # ... 现有测试
    ("你的测试名称", self.test_your_custom_test),
]
```

## 相关文档

- [SGLang 文档](https://docs.sglang.io)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [NPU 支持特性](../../docs/platforms/ascend_npu_support_features.md)
