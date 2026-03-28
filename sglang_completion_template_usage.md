# SGLang Completion Template 使用方法

## 1. 概述

Completion Template是SGLang中用于代码补全（特别是fill-in-the-middle, FIM）功能的模板系统。它定义了代码补全时使用的特殊标记（token）和格式，目前主要用于OpenAI兼容API服务器。

## 2. CompletionTemplate 类结构

```python
@dataclasses.dataclass
class CompletionTemplate:
    """A class that manages completion prompt templates. only for code completion currently."""

    # 模板名称
    name: str
    
    # FIM开始标记
    fim_begin_token: str
    
    # FIM中间标记
    fim_middle_token: str
    
    # FIM结束标记
    fim_end_token: str
    
    # FIM中间标记的位置
    fim_position: FimPosition  # MIDDLE 或 END
```

## 3. 内置模板

SGLang默认提供了几个常用的代码补全模板：

### 3.1 DeepSeek Coder
```python
CompletionTemplate(
    name="deepseek_coder",
    fim_begin_token="<｜fim▁begin｜>",
    fim_middle_token="<｜fim▁hole｜>",
    fim_end_token="<｜fim▁end｜>",
    fim_position=FimPosition.MIDDLE,
)
```

### 3.2 Star Coder
```python
CompletionTemplate(
    name="star_coder",
    fim_begin_token="<fim_prefix>",
    fim_middle_token="<fim_middle>",
    fim_end_token="<fim_suffix>",
    fim_position=FimPosition.END,
)
```

### 3.3 Qwen Coder
```python
CompletionTemplate(
    name="qwen_coder",
    fim_begin_token="<|fim_prefix|>",
    fim_middle_token="<|fim_middle|>",
    fim_end_token="<|fim_suffix|>",
    fim_position=FimPosition.END,
)
```

## 4. 使用方法

### 4.1 通过命令行参数使用

在启动SGLang服务器时，可以通过`--completion-template`参数指定使用的模板：

```bash
# 使用内置模板
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --completion-template deepseek_coder

# 使用自定义模板文件
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --completion-template /path/to/my_completion_template.json
```

### 4.2 在代码中使用

```python
from sglang.srt.parser.code_completion_parser import (
    CompletionTemplate,
    FimPosition,
    register_completion_template,
    set_completion_template,
    generate_completion_prompt,
    completion_template_exists
)

# 检查模板是否存在
if completion_template_exists("deepseek_coder"):
    print("DeepSeek Coder template exists")

# 设置要使用的模板
set_completion_template("star_coder")

# 生成补全提示
prompt = "def hello("
suffix = "): return 'Hello world'"
generated_prompt = generate_completion_prompt(prompt, suffix, "star_coder")
print(generated_prompt)  # <fim_prefix>def hello(<fim_suffix>): return 'Hello world'<fim_middle>

# 注册自定义模板
custom_template = CompletionTemplate(
    name="my_custom_coder",
    fim_begin_token="<START>",
    fim_middle_token="<MIDDLE>",
    fim_end_token="<END>",
    fim_position=FimPosition.MIDDLE,
)
register_completion_template(custom_template)

# 使用自定义模板
generated_prompt = generate_completion_prompt(prompt, suffix, "my_custom_coder")
print(generated_prompt)  # <START>def hello(<MIDDLE>): return 'Hello world'<END>
```

### 4.3 通过OpenAI API使用

当使用OpenAI兼容API进行代码补全时，SGLang会自动应用配置的completion template：

```python
import openai

client = openai.Client(api_key="sk-test", base_url="http://localhost:3000/v1")

response = client.completions.create(
    model="deepseek-coder",
    prompt="def hello(",  # 前缀
    suffix="): return 'Hello world'",  # 后缀
    max_tokens=50,
    temperature=0.1,
)

print(response.choices[0].text)  # 补全的中间部分
```

## 5. FIM位置说明

FIM（Fill-in-the-Middle）有两种不同的位置模式：

### 5.1 MIDDLE模式
```
<fim_begin>前缀<fim_middle>后缀<fim_end>
```
例如：`<｜fim▁begin｜>def hello(<｜fim▁hole｜>): return 'Hello world'<｜fim▁end｜>`

### 5.2 END模式
```
<fim_begin>前缀<fim_end>后缀<fim_middle>
```
例如：`<fim_prefix>def hello(<fim_suffix>): return 'Hello world'<fim_middle>`

不同的模型可能支持不同的FIM格式，需要根据模型要求选择合适的模板。

## 6. 自定义模板

### 6.1 通过代码注册

如前所述，可以通过`register_completion_template`函数注册自定义模板。

### 6.2 通过模板文件

可以创建JSON格式的模板文件，然后通过`--completion-template`参数指定：

```json
{
    "name": "my_template",
    "fim_begin_token": "<START>",
    "fim_middle_token": "<MIDDLE>",
    "fim_end_token": "<END>",
    "fim_position": "MIDDLE"
}
```

## 7. 使用场景

- **代码补全服务**：为IDE、编辑器等提供代码补全功能
- **API服务**：通过OpenAI兼容API提供代码补全能力
- **模型适配**：适配不同代码模型的FIM格式要求
- **自定义代码生成**：根据特定需求定制代码生成格式

## 8. 注意事项

1. Completion Template目前仅用于代码补全功能
2. 不同的代码模型可能需要使用不同的模板
3. 仅在OpenAI兼容API服务器中生效
4. FIM位置（MIDDLE/END）对补全结果有影响，需要根据模型要求选择
5. 自定义模板需要确保与模型训练时使用的格式一致

通过合理使用Completion Template，可以提高代码补全的准确性和效率，特别是对于支持FIM功能的代码模型。