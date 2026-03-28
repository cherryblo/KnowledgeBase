# SGLang --enable-cache-report 参数分析

## 1. 参数作用

`--enable-cache-report`是SGLang中的一个布尔类型参数，用于在OpenAI兼容API响应中返回缓存token的统计信息。

- **默认值**：`False`
- **启用方式**：`--enable-cache-report`（无需值，直接添加该参数）
- **作用**：在API响应的`usage`字段中添加`prompt_tokens_details`对象，包含`cached_tokens`字段，显示本次请求中从缓存中命中的token数量。

## 2. 实现原理

### 核心实现组件

1. **`UsageProcessor`类**：位于代码库中，负责将原始token计数转换为`UsageInfo`对象
2. **`PromptTokensDetails`类**：定义在`protocol.py`中，包含`cached_tokens`字段
3. **`UsageInfo`类**：定义在`protocol.py`中，包含`prompt_tokens_details`可选字段

### 工作流程

1. **参数传递**：用户通过命令行启用`--enable-cache-report`参数
2. **请求处理**：系统在处理请求时跟踪哪些token是从缓存中获取的
3. **缓存统计**：当请求完成后，`UsageProcessor`类会计算缓存的token数量
4. **响应生成**：如果启用了缓存报告，系统会将缓存信息添加到`UsageInfo`对象的`prompt_tokens_details`字段中
5. **结果返回**：API响应中包含带有缓存信息的`usage`字段

### 关键代码解析

```python
# UsageProcessor类中的核心方法
@staticmethod
def calculate_response_usage(responses, n_choices=1, enable_cache_report=False):
    # 计算总prompt和completion tokens
    prompt_tokens = sum(responses[i]["meta_info"].get("prompt_tokens", 0) 
                       for i in range(0, len(responses), n_choices))
    completion_tokens = sum(r["meta_info"].get("completion_tokens", 0) for r in responses)
    
    # 仅当enable_cache_report为True时计算缓存信息
    cached_details = None
    if enable_cache_report:
        cached_total = sum(responses[i]["meta_info"].get("cached_tokens", 0) 
                          for i in range(0, len(responses), n_choices))
        cached_details = UsageProcessor._details_if_cached(cached_total)
    
    return UsageProcessor.calculate_token_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_details,
    )
```

## 3. 开启参数后的输出信息

### 未开启时的响应
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 7,
    "total_tokens": 17
  }
}
```

### 开启后的响应（新增部分）
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "model-name",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 7,
    "total_tokens": 17,
    "prompt_tokens_details": {
      "cached_tokens": 5  # 新增：本次请求中从缓存命中的token数量
    }
  }
}
```

## 4. 使用场景

- **性能监控**：了解缓存命中率，评估缓存策略的有效性
- **成本优化**：缓存的token通常计算成本更低，可用于成本分析
- **系统调优**：根据缓存命中情况调整缓存配置和模型参数
- **API计费**：在需要基于实际计算资源使用量计费的场景中提供依据

## 5. 注意事项

- 只有在使用OpenAI兼容API时，该参数才会生效
- 缓存报告的准确性取决于底层缓存系统的实现
- 启用缓存报告可能会带来轻微的性能开销（计算缓存token数量）
- 缓存token数量仅反映prompt部分的缓存情况，不包括completion部分

通过这个参数，用户可以更好地了解SGLang的缓存机制工作情况，从而进行系统调优和性能优化。