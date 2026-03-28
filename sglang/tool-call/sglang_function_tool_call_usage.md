# SGLang Function 和 Tool Call 使用方法

## 1. 函数定义

在SGLang中，函数(tool)通过JSON格式定义，包含名称、描述和参数规范：

```python
ADD_TOOL = {
    "type": "function",
    "function": {
        "name": "add",  # 函数名称
        "description": "Compute the sum of two integers",  # 函数描述
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First integer"},  # 参数a
                "b": {"type": "integer", "description": "Second integer"},  # 参数b
            },
            "required": ["a", "b"],  # 必需参数
        },
    },
}

# 带strict模式的函数定义（强制参数校验）
ADD_TOOL_STRICT = {
    "type": "function",
    "function": {**ADD_TOOL["function"], "strict": True},
}
```

## 2. 工具调用模式

### 2.1 基本工具调用

指定`tool_choice="required"`强制模型调用工具：

```python
import openai
import json

client = openai.Client(api_key="sk-test", base_url="http://localhost:3000/v1")

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Compute 3 + 5"}],
    tools=[ADD_TOOL_STRICT],
    tool_choice="required",
    temperature=0.1,
)

# 处理响应
msg = response.choices[0].message
if msg.tool_calls and len(msg.tool_calls) > 0:
    tc = msg.tool_calls[0]
    function_name = tc.function.name  # 获取函数名
    arguments = json.loads(tc.function.arguments)  # 获取参数
    print(f"函数名: {function_name}")
    print(f"参数: {arguments}")
```

### 2.2 自动工具调用

设置`tool_choice="auto"`让模型决定是否调用工具：

```python
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Compute 3 + 5"}],
    tools=[ADD_TOOL_STRICT],
    tool_choice="auto",
    temperature=0.1,
)
```

### 2.3 流式工具调用

使用`stream=True`获取流式响应：

```python
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Compute 5 + 7"}],
    tools=[ADD_TOOL_STRICT],
    tool_choice="required",
    stream=True,
    temperature=0.1,
)

# 处理流式响应
arg_fragments = []
function_name = None
for chunk in response:
    if chunk.choices[0].delta.tool_calls:
        tc = chunk.choices[0].delta.tool_calls[0]
        function_name = tc.function.name or function_name
        if tc.function.arguments:
            arg_fragments.append(tc.function.arguments)

if function_name and arg_fragments:
    arguments = json.loads("" .join(arg_fragments))
    print(f"函数名: {function_name}")
    print(f"参数: {arguments}")
```

### 2.4 多轮工具调用

将工具调用结果返回给模型进行多轮对话：

```python
# 第一轮：请求工具调用
messages = [{"role": "user", "content": "What is 3 + 5?"}]
r1 = client.chat.completions.create(
    model="your-model",
    messages=messages,
    tools=[ADD_TOOL_STRICT],
    tool_choice="required",
    temperature=0.1,
)

tc = r1.choices[0].message.tool_calls[0]

# 模拟工具执行结果
result = 8

# 第二轮：将工具结果返回给模型
messages.append(r1.choices[0].message)
messages.append(
    {
        "role": "tool",
        "tool_call_id": tc.id,  # 必须与工具调用的ID匹配
        "content": str(result),  # 工具执行结果
        "name": tc.function.name,  # 必须与调用的函数名匹配
    }
)

r2 = client.chat.completions.create(
    model="your-model",
    messages=messages,
    tools=[ADD_TOOL],
    temperature=0.1,
)

# 模型基于工具结果的回复
print(r2.choices[0].message.content)
```

### 2.5 并行工具调用

单次请求调用多个工具：

```python
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "Please call both functions: use add to compute 3+5, and use get_weather to check the weather in Tokyo."}],
    tools=[ADD_TOOL_STRICT, WEATHER_TOOL_STRICT],
    tool_choice="auto",
    temperature=0,
)

# 处理多个工具调用结果
tool_calls = response.choices[0].message.tool_calls
for tc in tool_calls:
    function_name = tc.function.name
    arguments = json.loads(tc.function.arguments)
    print(f"函数名: {function_name}")
    print(f"参数: {arguments}")
```

### 2.6 流式并行工具调用

流式处理多个工具调用：

```python
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "What is 3+5 and what is the weather in Tokyo?"}],
    tools=[ADD_TOOL, WEATHER_TOOL],
    tool_choice="auto",
    stream=True,
)

# 收集所有工具调用
tool_calls = {}
for chunk in response:
    if not chunk.choices[0].delta.tool_calls:
        continue
    for tc in chunk.choices[0].delta.tool_calls:
        idx = tc.index
        if idx not in tool_calls:
            tool_calls[idx] = {"name": "", "arguments": ""}
        if tc.function.name:
            tool_calls[idx]["name"] = tc.function.name
        if tc.function.arguments:
            tool_calls[idx]["arguments"] += tc.function.arguments

# 处理收集到的工具调用
for idx, tc in tool_calls.items():
    print(f"工具调用 {idx}:")
    print(f"  函数名: {tc['name']}")
    print(f"  参数: {json.loads(tc['arguments'])}")
```

### 2.7 指定特定工具

强制模型调用指定的工具：

```python
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    tools=[ADD_TOOL, WEATHER_TOOL],
    tool_choice={"type": "function", "function": {"name": "get_weather"}},  # 指定调用get_weather工具
    temperature=0.1,
)
```

### 2.8 不使用工具

设置`tool_choice="none"`禁止模型调用任何工具：

```python
response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "What is 1+1?"}],
    tools=[ADD_TOOL_STRICT],
    tool_choice="none",  # 禁止调用工具
    temperature=0.1,
)
```

## 3. 响应解析

工具调用的响应结构：

```python
# 完整响应
response = client.chat.completions.create(...)
message = response.choices[0].message

# 工具调用信息
tool_calls = message.tool_calls  # 工具调用列表
for tc in tool_calls:
    tc.id  # 工具调用ID（用于多轮对话）
    tc.function.name  # 调用的函数名
    tc.function.arguments  # 函数参数（JSON字符串）
    json.loads(tc.function.arguments)  # 解析为Python对象

# 完成原因
response.choices[0].finish_reason  # "tool_calls" 或 "stop"
```

## 4. 关键参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `tools` | 可用工具列表 | 函数定义JSON列表 |
| `tool_choice` | 工具选择策略 | "required"（必须调用）、"auto"（自动决定）、"none"（不调用）、或特定函数对象 |
| `stream` | 是否使用流式响应 | `True`/`False` |
| `temperature` | 采样温度，影响输出随机性 | 0.0-2.0 |

## 5. 注意事项

1. 使用`strict=True`可以强制模型遵循参数规范
2. 多轮对话中，`tool_call_id`必须与之前的工具调用ID匹配
3. 流式响应需要逐块收集参数片段并组合
4. 并行工具调用会在单个响应中返回多个`tool_calls`
5. 工具调用完成后，`finish_reason`会是`"tool_calls"`

以上是从SGLang测试用例中提取的function和tool call的完整使用方法，涵盖了各种常见的使用场景。