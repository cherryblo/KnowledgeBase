# Checkpoint Decryption功能测试用例

## 一、使用场景分析

### 1.1 使用场景
Checkpoint Decryption功能主要用于以下场景：

1. **加密模型权重加载**：当模型checkpoint经过加密处理（如使用PBKDF2加密）时，需要使用解密后的配置文件来正确加载模型
2. **Speculative Decoding场景**：在使用speculative decoding时，主模型和draft模型可能分别使用不同的解密配置
3. **安全模型部署**：企业环境中，模型权重可能经过加密保护，需要解密配置才能正确加载
4. **多模型部署**：同时部署多个加密模型时，每个模型可能需要独立的解密配置

### 1.2 作用
- 提供加密模型checkpoint的正确配置信息
- 支持主模型和draft模型分别配置
- 确保模型加载时使用正确的解密参数

### 1.3 使用约束
- 解密配置文件必须是有效的JSON格式
- 配置文件路径必须存在且可读
- 必须与加密的模型checkpoint匹配
- 仅在模型checkpoint被加密时才需要使用

### 1.4 增益
- 支持加载加密的模型权重，增强安全性
- 灵活配置主模型和draft模型的解密参数
- 与HuggingFace transformers库无缝集成

---

## 二、实现原理分析

### 2.1 数据流

```
启动SGLang服务器
    ↓
解析命令行参数
    ↓
ServerArgs.decrypted_config_file (主模型解密配置)
ServerArgs.decrypted_draft_config_file (draft模型解密配置)
    ↓
ModelConfig初始化
    ↓
根据is_draft_model选择配置文件
    ↓
设置override_config_file
    ↓
调用get_config()加载HuggingFace配置
    ↓
传递_configuration_file参数给transformers
    ↓
PretrainedConfig.get_config_dict(_configuration_file=...)
    ↓
加载模型权重
    ↓
服务器就绪
```

### 2.2 内部处理逻辑

#### 2.2.1 核心组件

1. **ServerArgs**（服务器参数）
   - 位置：`python/sglang/srt/server_args.py`
   - 相关字段：
     - `decrypted_config_file`: 主模型的解密配置文件路径
     - `decrypted_draft_config_file`: draft模型的解密配置文件路径

2. **ModelConfig**（模型配置）
   - 位置：`python/sglang/srt/configs/model_config.py`
   - 作用：管理模型配置和加载
   - 核心逻辑：
     - 在`_get_hf_config()`方法中处理解密配置
     - 根据是否为draft模型选择相应的配置文件
     - 通过`_configuration_file`参数传递给transformers

3. **get_config()**（配置加载）
   - 位置：`python/sglang/srt/utils/hf_transformers_utils.py`
   - 作用：从HuggingFace加载模型配置
   - 接受`_configuration_file`参数用于解密配置

#### 2.2.2 处理流程

1. **参数解析阶段**
   - 解析`--decrypted-config-file`参数
   - 解析`--decrypted-draft-config-file`参数
   - 存储到`ServerArgs`对象中

2. **模型配置构建阶段**
   - 在`ModelConfig._get_hf_config()`中：
     ```python
     override_config_file = (
         server_args.decrypted_draft_config_file
         if is_draft_model
         else server_args.decrypted_config_file
     )
     ```

3. **配置加载阶段**
   - 如果设置了`override_config_file`：
     ```python
     if override_config_file and override_config_file.strip():
         kwargs["_configuration_file"] = override_config_file.strip()
     ```
   - 调用`get_config()`时传递`_configuration_file`参数

4. **Transformers集成**
   - `PretrainedConfig.get_config_dict()`接收`_configuration_file`参数
   - 用于解密和加载模型配置

### 2.3 配置参数详解

#### 2.3.1 --decrypted-config-file
- **作用**：指定主模型的解密配置文件路径
- **类型**：str
- **默认值**：None
- **有效范围**：有效的文件路径字符串
- **生效点**：`ModelConfig._get_hf_config()`方法中
- **影响**：
  - 当`is_draft_model=False`时使用
  - 传递给transformers的`get_config_dict()`作为`_configuration_file`参数
  - 影响主模型权重的解密和加载

#### 2.3.2 --decrypted-draft-config-file
- **作用**：指定draft模型的解密配置文件路径
- **类型**：str
- **默认值**：None
- **有效范围**：有效的文件路径字符串
- **生效点**：`ModelConfig._get_hf_config()`方法中
- **影响**：
  - 当`is_draft_model=True`时使用
  - 传递给transformers的`get_config_dict()`作为`_configuration_file`参数
  - 影响draft模型权重的解密和加载

---

## 三、测试用例设计

### 用例1：基础解密配置文件测试

#### 测试目的
验证`--decrypted-config-file`参数在NPU环境下生效，能够正确指定主模型的解密配置

#### 测试步骤
1. 创建一个模拟的解密配置文件（JSON格式）
2. 启动SGLang服务器，指定`--decrypted-config-file`参数
3. 验证服务器成功启动
4. 发送推理请求验证模型功能正常

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器成功启动
- 解密配置文件被正确加载
- 模型推理功能正常

---

### 用例2：Draft模型解密配置测试

#### 测试目的
验证`--decrypted-draft-config-file`参数在NPU环境下生效，能够正确指定draft模型的解密配置

#### 测试步骤
1. 创建主模型和draft模型的解密配置文件
2. 启动SGLang服务器，启用speculative decoding
3. 同时指定`--decrypted-config-file`和`--decrypted-draft-config-file`
4. 验证服务器成功启动
5. 发送推理请求验证speculative decoding功能正常

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器成功启动
- 主模型和draft模型的解密配置都被正确加载
- Speculative decoding功能正常

---

### 用例3：仅主模型解密配置测试

#### 测试目的
验证只设置主模型解密配置而不设置draft模型配置时的行为

#### 测试步骤
1. 创建主模型的解密配置文件
2. 启动SGLang服务器，仅指定`--decrypted-config-file`
3. 启用speculative decoding但不设置draft解密配置
4. 验证服务器启动行为

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器成功启动
- 主模型使用解密配置
- Draft模型使用默认配置或正常加载

---

### 用例4：无效配置文件路径测试

#### 测试目的
验证当指定不存在的解密配置文件时的错误处理

#### 测试步骤
1. 启动SGLang服务器，指定不存在的配置文件路径
2. 观察服务器启动行为

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器启动失败或给出明确的错误信息
- 错误信息指出配置文件不存在

---

### 用例5：无效JSON格式测试

#### 测试目的
验证当解密配置文件包含无效JSON格式时的错误处理

#### 测试步骤
1. 创建包含无效JSON的配置文件
2. 启动SGLang服务器，指定该配置文件
3. 观察服务器启动行为

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器启动失败或给出明确的错误信息
- 错误信息指出JSON格式错误

---

### 用例6：配置文件权限测试

#### 测试目的
验证当解密配置文件没有读取权限时的错误处理

#### 测试步骤
1. 创建解密配置文件并设置为不可读
2. 启动SGLang服务器，指定该配置文件
3. 观察服务器启动行为

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器启动失败或给出明确的错误信息
- 错误信息指出文件权限问题

---

### 用例7：不使用解密配置测试

#### 测试目的
验证不指定解密配置文件时，服务器能正常启动

#### 测试步骤
1. 启动SGLang服务器，不指定任何解密配置参数
2. 验证服务器成功启动
3. 发送推理请求验证功能正常

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器成功启动
- 模型使用默认配置加载
- 推理功能正常

---

### 用例8：多模型部署测试

#### 测试目的
验证在多模型部署场景下，每个模型使用独立的解密配置

#### 测试步骤
1. 创建多个模型的解密配置文件
2. 分别启动多个SGLang服务器实例
3. 每个实例指定不同的解密配置
4. 验证所有服务器都成功启动

#### 测试环境
- 使用1张NPU卡
- 单机部署（多端口）

#### 预期结果
- 所有服务器实例都成功启动
- 每个实例使用正确的解密配置
- 所有实例推理功能正常

---

### 用例9：配置文件内容验证测试

#### 测试目的目的
验证解密配置文件的内容被正确传递给模型加载过程

#### 测试步骤
1. 创建包含特定配置项的解密配置文件
2. 启动SGLang服务器，指定该配置文件
3. 通过服务器API获取模型信息
4. 验证配置项被正确应用

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器成功启动
- 配置文件中的配置项被正确应用
- 模型行为符合配置

---

### 用例10：并发加载测试

#### 测试目的
验证在并发场景下，解密配置的正确加载

#### 测试步骤
1. 创建解密配置文件
2. 启动SGLang服务器，指定解密配置
3. 同时发送多个并发推理请求
4. 验证所有请求都成功处理

#### 测试环境
- 使用1张NPU卡
- 单机部署

#### 预期结果
- 服务器成功启动
- 所有并发请求都成功处理
- 无配置相关的错误

---

## 四、已有测试用例分析

根据代码分析，SGLang目前没有专门的checkpoint decryption测试用例。相关功能主要通过集成测试间接验证：
- 模型加载测试
- Speculative decoding测试

**参考价值**：
- 现有测试验证了模型加载的基本流程
- 但缺少对解密配置参数的专门测试
- 需要补充专门的解密配置测试用例

---

## 五、测试用例与脚本对应关系

| 用例编号 | 用例名称 | 脚本文件名 |
|---------|---------|-----------|
| 用例1 | 基础解密配置文件测试 | test_01_basic_decryption_config.py |
| 用例2 | Draft模型解密配置测试 | test_02_draft_decryption_config.py |
| 用例3 | 仅主模型解密配置测试 | test_03_main_only_decryption.py |
| 用例4 | 无效配置文件路径测试 | test_04_invalid_path.py |
| 用例5 | 无效JSON格式测试 | test_05_invalid_json.py |
| 用例6 | 配置文件权限测试 | test_06_file_permission.py |
| 用例7 | 不使用解密配置测试 | test_07_no_decryption.py |
| 用例8 | 多模型部署测试 | test_08_multi_model.py |
| 用例9 | 配置文件内容验证测试 | test_09_config_content.py |
| 用例10 | 并发加载测试 | test_10_concurrent_loading.py |
