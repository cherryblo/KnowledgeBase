# --completion-template 参数分析与测试用例

## 一、使用场景分析

### 1.1 使用场景
`--completion-template` 参数用于在 SGLang 推理服务中配置代码补全（FIM - Fill-in-the-Middle）功能。主要使用场景包括：

- **代码编辑器集成**：为 VS Code、Cursor 等 IDE 提供智能代码补全
- **代码生成工具**：在代码生成场景中自动补全函数或类定义
- **代码审查助手**：帮助开发者快速完成代码片段
- **AI 编程助手**：为开发者提供实时的代码建议

### 1.2 作用说明
该参数的作用包括：

1. **模板选择**：选择使用哪种 FIM 格式模板
2. **提示格式化**：将用户输入（前缀和后缀）格式化为模型可理解的提示
3. **标记管理**：定义 FIM 开始、中间、结束标记
4. **位置控制**：控制补全内容在提示中的位置

### 1.3 使用约束
- 只用于 OpenAI 兼容 API 服务
- 只支持代码补全功能，不用于聊天或文本生成
- 需要使用支持 FIM 的模型
- 必须提供 prompt 和 suffix 参数

### 1.4 带来的增益
- **提升开发效率**：减少手动输入代码的时间
- **改善准确性**：通过专门的模板提高补全准确率
- **标准化接口**：提供统一的 OpenAI 兼容接口
- **灵活性**：支持自定义模板以适应不同需求

## 二、实现原理分析

### 2.1 数据流

```
服务启动
    ↓
解析 --completion-template 参数
    ↓
TemplateManager.load_completion_template()
    ↓
加载模板（内置或 JSON 文件）
    ↓
注册 CompletionTemplate
    ↓
接收 /v1/completions 请求
    ↓
generate_completion_prompt_from_request()
    ↓
使用模板格式化 prompt 和 suffix
    ↓
发送到模型推理
    ↓
返回补全结果
```

### 2.2 内部处理逻辑

#### 2.2.1 初始化阶段
1. **参数解析**：从命令行读取 `--completion-template` 参数
2. **模板加载**：`TemplateManager.load_completion_template()` 处理模板
3. **模板验证**：检查模板是否为内置名称或有效文件路径
4. **模板注册**：将模板注册到全局 `completion_templates` 字典

#### 2.2.2 模板结构
每个 CompletionTemplate 包含以下字段：
- `name`: 模板名称
- `fim_begin_token`: FIM 开始标记（如 `<｜fim▁begin｜>`）
- `fim_middle_token`: FIM 中间标记（如 `<｜fim▁hole｜>`）
- `fim_end_token`: FIM 结束标记（如 `<｜fim▁end｜>`）
- `fim_position`: FimPosition 枚举（MIDDLE 或 END）

#### 2.2.3 请求处理阶段
1. **参数接收**：接收 prompt（代码前缀）和 suffix（代码后缀）
2. **提示生成**：调用 `generate_completion_prompt_from_request()`
3. **模板应用**：根据模板格式生成完整提示
4. **标记插入**：在适当位置插入 FIM 标记

#### 2.2.4 提示生成逻辑
根据 `fim_position` 的值：
- **FimPosition.MIDDLE**：`{fim_begin_token}{prompt}{fim_middle_token}{suffix}{fim_end_token}`
- **FimPosition.END**：`{fim_begin_token}{prompt}{fim_end_token}{suffix}{fim_middle_token}`

### 2.3 配置参数

#### 2.3.1 --completion-template
- **参数名**：`--completion-template`
- **参数类型**：字符串
- **有效值**：
  - 内置模板名称：
    - `deepseek_coder`: DeepSeek Coder 格式
    - `star_coder`: Star Coder 格式
    - `qwen_coder`: Qwen Coder 格式
  - 自定义模板文件路径：JSON 格式的模板文件
- **默认值**：`None`

#### 2.3.2 内置模板格式

**deepseek_coder**:
- fim_begin_token: `<｜fim�▁begin｜>`
- fim_middle_token: `<｜fim▁hole｜>`
- fim_end_token: `<｜fim▁end｜>`
- fim_position: MIDDLE

**star_coder**:
- fim_begin_token: `<fim_prefix>`
- fim_middle_token: `<fim_middle>`
- fim_end_token: `<fim_suffix>`
- fim_position: END

**qwen_coder**:
- fim_begin_token: `<|fim_prefix|>`
- fim_middle_token: `<|fim_middle|>`
- fim_end_token: `<|fim_suffix|>`
- fim_position: END

#### 2.3.3 自定义 JSON 模板格式
```json
{
    "name": "custom_template",
    "fim_begin_token": "<custom_begin>",
    "fim_middle_token": "<custom_middle>",
    "fim_end_token": "<custom_end>",
    "fim_position": "middle"  // 或 "end"
}
```

### 2.4 参数生效点和影响

#### 2.4.1 生效点
1. **服务启动时**：`server_args.py:4384-4388` 添加命令行参数
2. **模板加载时**：`template_manager.py:183-203` 加载并注册模板
3. **请求处理时**：`code_completion_parser.py:83-106` 生成补全提示
4. **提示生成时**：根据模板格式化输入

#### 2.4.2 影响
- **提示格式**：决定了发送给模型的提示格式
- **补全位置**：影响模型理解补全内容的位置
- **兼容性**：确保与特定模型的 FIM 格式兼容
- **性能**：适当的模板可以提高补全质量

## 三、测试用例设计

### 3.1 测试用例 1：内置 deepseek_coder 模板测试

**测试目的**：验证使用内置 deepseek_coder 模板时，FIM 补全功能正常工作

**测试步骤**：
1. 启动 SGLang 服务，指定 `--completion-template deepseek_coder`
2. 发送代码补全请求，提供 prompt 和 suffix
3. 验证返回的补全内容正确
4. 验证 usage 统计信息准确

**测试环境**：单张 NPU 卡

**预期结果**：
- 服务成功启动
- 返回有效的补全内容
- prompt_tokens 和 completion_tokens 统计正确
- 补全内容连接 prompt 和 suffix 形成完整代码

### 3.2 测试用例 2：内置 star_coder 模板测试

**测试目的**：验证使用内置 star_coder 模板时，FIM 补全功能正常工作

**测试步骤**：
1. 启动 SGLang 服务，指定 `--completion-template star_coder`
2. 发送代码补全请求
3. 验证返回的补全内容符合 star_coder 格式
4. 验证补全内容在正确的位置

**测试环境**：单张 NPU 卡

**预期结果**：
- 服务成功启动
- 补全内容格式正确
- 补全内容能够正确连接前后缀

### 3.312 测试用例 3：内置 qwen_coder 模板测试

**测试目的**：验证使用内置 qwen_coder 模板时，FIM 补全功能正常工作

**测试步骤**：
1. 启动 SGLang 服务，指定 `--completion-template qwen_coder`
2. 发送代码补全请求
3. 验证返回的补全内容符合 qwen_coder 格式
4. 验证多个补全请求都能正常工作

**测试环境**：单张 NPU 卡

**预期结果**：
- 服务成功启动
- 补全内容格式正确
- 多次请求都能成功返回

### 3.4 测试用例 4：自定义 JSON 模板测试

**测试目的**：验证使用自定义 JSON 模板文件时，FIM 补全功能正常工作

**测试步骤**：
1. 创建自定义 JSON 模板文件
2. 启动 SGLang 服务，指定 `--completion-template /path/to/template.json`
3. 发送代码补全请求
4. 验证自定义模板被正确加载和应用

**测试环境**：单张 NPU 卡

**预期结果**：
- 服务成功启动
- 自定义模板被正确加载
- 补全内容符合自定义模板格式

### 3.5 测试用例 5：多补全请求测试

**测试目的**：验证在单个请求中生成多个补全候选的功能

**测试步骤**：
1. 启动 SGLang 服务，指定 `--completion-template deepseek_coder`
2. 发送代码补全请求，设置 `n=3` 生成 3 个候选
3. 验证返回 3 个不同的补全结果
4. 验证每个结果都是有效的

**测试环境**：单张 NPU 卡

**预期结果**：
- 返回 3 个补全候选
- 每个候选都是有效的代码片段
- 所有候选都能正确连接前后缀

### 3.6 测试用例 6：流式补全测试

**测试目的**：验证流式代码补全功能

**测试步骤**：
1. 启动 SGLang 服务，指定 `--completion-template deepseek_coder`
2. 发送流式代码补全请求
3. 逐块接收补全内容
4. 验证流式输出的完整性

**测试环境**：单张 NPU 卡

**预期结果**：
- 流式输出正常
- 补全内容增量返回
- 最终结果完整

### 3.7 测试用例 7：无效模板名称错误处理测试

**测试目的**：验证传入无效模板名称时的错误处理

**测试步骤**：
1. 尝试启动服务，指定无效的模板名称
2. 观察错误信息
3. 验证服务启动失败或给出明确错误

**测试环境**：单张 NPU 卡

**预期结果**：
- 服务启动失败
- 给出明确的错误提示
- 指出有效的模板选项

### 3.8 测试用例 8：无效模板文件路径错误处理测试

**测试目的**：验证传入无效模板文件路径时的错误处理

**测试步骤**：
1. 尝试启动服务，指定不存在的模板文件路径
2. 观察错误信息
3. 验证服务启动失败

**测试环境**：单张 NPU 卡

**预期结果**：
- 服务启动失败
- 给出文件不存在的错误提示

## 四、测试说明

### 4.1 测试覆盖范围
- ✅ 内置 deepseek_coder 模板
- ✅ 内置 star_coder 模板
- ✅ 内置 qwen_coder 模板
- ✅ 自定义 JSON 模板
- ✅ 多补全请求
- ✅ 流式补全
- ✅ 无效模板名称错误处理
- ✅ 无效模板文件路径错误处理

### 4.2 测试注意事项
1. 测试使用支持 FIM 的模型（如 DeepSeek-Coder）
2. 所有测试在 NPU 环境下执行
3. 测试使用 ascend attention backend
4. 测试使用 /v1/completions 端点
5. 测试端口使用默认的 30000
6. 每个测试用例独立启动和关闭服务

### 4.3 测试依赖
- SGLang 框架
- OpenAI Python SDK
- 支持 FIM 的模型权重（如 DeepSeek-Coder）
- NPU 硬件支持

### 4.4 测试模型
推荐使用以下模型进行模型：
- DeepSeek-Coder 系列
- Qwen-Coder 系列
- 其他支持 FIM 的代码模型
