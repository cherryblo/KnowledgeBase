# --tool-call-parser 参数分析与测试用例

## 一、使用场景分析

### 1.1 使用场景
`--tool-call-parser` 参数用于在 SGLang 推理服务中启用和配置工具调用（Function Calling/Tool Use）解析功能。主要使用场景包括：

- **智能体应用**：当模型需要调用外部工具或函数来完成任务时
- **API 集成**：将大语言模型与外部系统（如数据库、API 服务）集成
- **多步骤任务**：需要模型执行一系列工具调用来完成复杂任务
- **结构化输出**：需要模型按照特定格式输出函数调用参数

### 1.2 作用说明
该参数的作用包括：

1. **解析模型输出**：识别模型生成的工具调用内容，将其从原始文本中提取出来
2. **格式转换**：将不同模型的工具调用格式转换为统一的 OpenAI 兼容格式
3. **流式支持**：支持在流式输出中实时解析工具调用
4. **参数验证**：验证工具调用的参数是否符合定义的 JSON Schema
5. **约束生成**：为模型生成提供结构化约束，确保输出格式正确

### 1.3 使用约束
- 需要模型本身具备工具调用能力
- 不同模型需要使用对应的 parser 类型
- 需要在请求中提供 tools 定义
- 某些 parser 可能需要特定的 chat template

### 1.4 带来的增益
- **提升准确性**：通过结构化约束提高工具调用的准确率
- **改善用户体验**：流式解析提供更实时的反馈
- **简化集成**：统一的接口简化了与不同模型的集成
- **降低错误率**：参数验证减少无效的工具调用

## 二、实现原理分析

### 2.1 数据流

```
用户请求 (包含 tools 定义)
    ↓
OpenAI API 接口 (/v1/chat/completions)
    ↓
ServingChat 处理
    ↓
创建 FunctionCallParser 实例
    ↓
生成结构化约束 (tool_call_constraint)
    ↓
模型推理 (带约束)
    ↓
DetokenizerManager 处理输出
    ↓
FunctionCallParser 解析
    ↓
返回工具调用结果
```

### 2.2 内部处理逻辑

#### 2.2.1 初始化阶段
1. **参数解析**：从 server_args 读取 tool_call_parser 配置
2. **Parser 选择**：根据 parser 类型选择对应的 Detector 类
3. **实例化**：创建 FunctionCallParser 对象，包含对应的 Detector

#### 2.2.2 请求处理阶段
1. **Tools 验证**：验证请求中的 tools 定义是否有效
2. **约束生成**：调用 `get_structure_constraint()` 生成约束
   - 如果支持 structural_tag 且 tool_choice=auto，生成 structural_tag 约束
   - 如果 tool_choice=required 或指定工具，生成 json_schema 约束
3. **约束应用**：将约束添加到采样参数中

#### 2.2.3 输出解析阶段
1. **流式解析**：调用 `parse_streaming_increment()` 逐块解析
   - 维护缓冲区处理不完整的输出
   - 识别工具调用标记
   - 提取工具名称和参数
   - 返回解析结果
2. **非流式解析**：调用 `parse_non_streaming()` 一次性解析
   - 查找所有工具调用块
   - 解析 JSON 内容
   - 返回完整的工具调用列表

#### 2.2.4 Detector 实现
每个 Parser 对应一个 Detector 类，实现以下方法：
- `has_tool_call(text)`: 检测文本是否包含工具调用
- `detect_and_parse(text, tools)`: 一次性解析
- `parse_streaming_increment(new_text, tools)`: 流式增量解析
- `structure_info()`: 返回结构化信息用于约束生成

### 2.3 配置参数

#### 2.3.1 --tool-call-parser
- **作用**：指定工具调用解析器类型
- **类型**：字符串
- **有效值**：
  - `deepseekv3`: DeepSeek V3 格式
  - `deepseekv31`: DeepSeek V3.1 格式
  - `deepseekv32`: DeepSeek V3.2 格式
  - `glm`: GLM-4 格式
  - `glm45`: GLM-4.5 格式（已废弃，使用 glm）
  - `glm47`: GLM-4.7 格式
  - `gpt-oss`: GPT-OSS 格式
  - `kimi_k2`: Kimi K2 格式
  - `llama3`: Llama-3 格式
  - `mistral`: Mistral 格式
  - `pythonic`: Pythonic 格式
  - `qwen`: Qwen 格式（推荐）
  - `qwen25`: Qwen 2.5 格式（已废弃，使用 qwen）
  - `qwen3_coder`: Qwen3 Coder 格式
  - `step3`: Step-3 格式
  - `gigachat3`: GigaChat3 格式

#### 2.3.2 相关参数
- `--tool-choice`: 工具选择策略（auto/required/none/指定工具）
- `tools`: 请求参数，定义可用的工具列表

### 2.4 参数生效点和影响

#### 2.4.1 生效点
1. **服务启动时**：解析命令行参数，存储 parser 类型
2. **请求处理时**：在 ServingChat.create_request_completion() 中创建 FunctionCallParser
3. **约束生成时**：在 get_structure_constraint() 中根据 tool_choice 生成约束
4. **输出解析时**：在 DetokenizerManager 中调用解析器处理输出

#### 2.4.2 影响
- **约束生成**：影响模型输出的格式和内容
- **解析准确性**：不同 parser 对应不同的解析逻辑
- **性能**：流式解析的实时性影响用户体验
- **兼容性**：支持不同模型的工具调用格式

## 三、测试用例设计

### 3.1 测试用例 1：Llama3 Parser 基础功能测试

**测试目的**：验证 llama3 parser 能够正确解析 Llama-3.2 模型的工具调用输出

**测试步骤**：
1. 启动 SGLang 服务，指定 `--tool-call-parser llama3`
2. 发送包含 tools 定义的 chat completion 请求
3. 验证返回的 tool_calls 格式正确
4. 验证工具名称和参数解析正确

**测试环境**：单张 NPU 卡

**预期结果**：
- 请求成功返回
- tool_calls 字段不为空
- 工具名称为 "add"
- 参数包含 a 和 b 字段

### 3.2 测试用例 2：Pythonic Parser 流式输出测试

**测试目的**：验证 pythonic parser 在流式输出模式下能够正确解析工具调用

**测试步骤**：
1. 启动 SGLang 服务，指定 `--tool-call-parser pythonic`
2. 发送流式请求，包含多个工具定义
3. 逐块接收输出，验证工具调用的实时解析
4. 验证最终解析结果完整

**测试环境**：单张 NPU 卡

**预期结果**：
- 流式输出正常
- 能够实时识别工具调用开始
- 工具参数增量解析正确
- 最终结果包含所有工具调用

### 3.3 测试用例 3：Qwen Parser 多工具调用测试

**测试目的**：验证 qwen parser 能够处理并行的多工具调用

**测试步骤**：
1. 启动 SGLang 服务，指定 `--tool-call-parser qwen`
2. 定义多个工具（get_weather, get_tourist_attractions）
3. 发送可能触发多个工具调用的请求
4. 验证返回多个工具调用

**测试环境**：单张 NPU 卡

**预期结果**：
- 返回多个 tool_calls
- 每个工具调用格式正确
- 参数解析准确

### 3.4 测试用例 4：Tool Choice Required 测试

**测试目的**：验证 tool_choice=required 时强制工具调用的约束生效

**测试步骤**：
1. 启动 SGLang 服务，指定 `--tool-call-parser llama3`
2. 发送请求，设置 tool_choice="required"
3. 验证模型必须返回工具调用

**测试环境**：单张 NPU 卡

**预期结果**：
- 模型必须返回工具调用
- 不会返回纯文本回复

### 3.5 测试用例 5：多轮对话工具调用测试

**测试目的**：验证在多轮对话中工具调用的正确性

**测试步骤**：
1. 启动 SGLang 服务，指定 `--tool-call-parser llama3`
2. 第一轮：发送用户消息，获取工具调用
3. 第二轮：发送工具响应，获取最终答案
4. 验证对话历史正确传递

**测试环境**：单张 NPU 卡

**预期结果**：
- 第一轮返回工具调用
- 第二轮基于工具响应返回正确答案
- 对话上下文正确

### 3.6 测试用例 6：无效 Parser 类型测试

**测试目的**：验证传入无效 parser 类型时的错误处理

**测试步骤**：
1. 尝试启动服务，指定无效的 parser 类型
2. 观察服务启动行为

**测试环境**：单张 NPU 卡

**预期结果**：
- 服务启动失败或给出明确错误提示
- 不会使用默认值继续运行

### 3.7 测试用例 7：不同 Parser 格式对比测试

**测试目的**：验证不同 parser 类型对应不同的输出格式

**测试步骤**：
1. 分别使用 llama3、pythonic、qwen parser 启动服务
2. 发送相同的请求
3. 观察不同 parser 的解析行为

**测试环境**：单张 NPU 卡

**预期结果**：
- 不同 parser 正确处理各自对应的格式
- 最终都返回统一的 OpenAI 格式

### 3.8 测试用例 8：参数验证测试

**测试目的**：验证工具调用参数的 JSON Schema 验证功能

**测试步骤**：
1. 启动服务，指定 parser
2. 定义带严格参数验证的工具
3. 发送请求，观察参数验证效果

**测试环境**：单张 NPU 卡

**预期结果**：
- 参数符合 schema 定义
- 必填参数不能缺失
- 参数类型正确

## 四、测试说明

### 4.1 测试覆盖范围
- ✅ 基础功能测试
- ✅ 流式输出测试
- ✅ 多工具调用测试
- ✅ 约束生成测试
- ✅ 多轮对话测试
- ✅ 错误处理测试
- ✅ 格式兼容性测试
- ✅ 参数验证测试

### 4.2 测试注意事项
1. 测试使用 Llama-3.2-1B-Instruct 模型
2. 所有测试在 NPU 环境下执行
3. 测试使用 ascend attention backend
4. 测试端口使用默认的 30000
5. 每个测试用例独立启动和关闭服务

### 4.3 测试依赖
- SGLang 框架
- OpenAI Python SDK
- Llama-3.2-1B-Instruct 模型权重
- NPU 硬件支持
