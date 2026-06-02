# gen_dataset_fixed_len.py 逐行分析

## 文件概述

- **路径**: `python/sglang/test/ascend/e2e/gen_dataset_fixed_len.py`
- **被调用方**: `test_npu_performance_utils.py` 中的 `run_aisbench()` 和 `run_bench_serving()`
- **作用**: 为 NPU 性能测试生成固定 token 长度的压测数据集，支持三种数据集类型

---

# 一、辅助函数

## 第 11-19 行：load_jsonl

```python
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data
```

按行读取 JSONL 文件，跳过空行。每行是一个独立的 JSON 对象。

---

## 第 22-29 行：save_jsonl

```python
def save_jsonl(data, file_path):
    file_dir = os.path.dirname(file_path)
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

将 `list[dict]` 写入 JSONL 文件，自动创建父目录，`ensure_ascii=False` 保留中文等非 ASCII 字符。

---

## 第 32-36 行：format_qa

```python
def format_qa(item):
    question = item["question"]
    answer = item["answer"]
    return f"Question: {question}\nLet's think step by step\nAnswer:\n{answer}\n\n"
```

将 GSM8K 数据条目格式化为 few-shot 示例文本。格式是经典的 Chain-of-Thought prompt 模板。

**示例输入**:
```json
{"question": "James buys 3 apples...", "answer": "He spent 6 dollars."}
```

**输出**:
```
Question: James buys 3 apples...
Let's think step by step
Answer:
He spent 6 dollars.

```

---

## 第 39-92 行：pad_to_target_tokens — 核心填充算法

```python
def pad_to_target_tokens(
    question,
    few_shot_pool_token_ids,
    tokenizer,
    target_tokens,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
):
```

### 算法流程

```
输入: question="James buys 3 apples...", target_tokens=3500

Step 1: 计算问题本身的 token 数
  test_prompt = "Question: James buys 3 apples...\nLet's think step by step\nAnswer:\n"
  test_token_ids = tokenizer.encode(test_prompt)        # 例如: 45 tokens
  remaining_tokens = 3500 - 45 = 3455                   # 还需填充 3455 tokens

Step 2: 随机打乱 few-shot 池索引
  shuffled_ids = shuffle([0, 1, 2, ..., N-1])

Step 3: 从 few-shot 池中逐个拼接示例，填满 remaining_tokens
  for idx in shuffled_ids:
      fs_ids = few_shot_pool_token_ids[idx]
      if len(prefix_ids) + len(fs_ids) <= remaining_tokens:
          prefix_ids.extend(fs_ids)       # 完整添加这个 few-shot 示例
      else:
          # 最后一个示例只能部分添加（截断）
          partial_gap = remaining_tokens - len(prefix_ids)
          prefix_ids.extend(fs_ids[:partial_gap])
          break

Step 4: 如果池中所有示例用完还不够 → 循环重复第一个示例填满
  if len(prefix_ids) < remaining_tokens:
      repeat_count = (remaining_tokens // len(padding_source_ids)) + 1
      prefix_ids.extend((padding_source_ids * repeat_count)[:gap])

Step 5: 拼接并返回
  full_ids = prefix_ids + test_token_ids
  return tokenizer.decode(full_ids[:target_tokens])
```

### 关键设计点

| 设计点 | 说明 |
|--------|------|
| 前缀填充 | 在问题文本**前面**填充 few-shot 示例，而不是后面，避免截断问题本身 |
| 随机顺序 | 每次生成的 few-shot 前缀顺序不同，增加数据多样性 |
| 截断容错 | 最后一个示例不够完整时截断部分 token，而不是丢弃 |
| 循环兜底 | 整个池用完还不够（target_tokens 远大于池大小），则重复第一个示例 |

### 图解

```
┌────────────────────────────────────────────────┬──────────────────┐
│         few-shot prefix (3455 tokens)          │   question       │
│  ┌──────┐ ┌──────┐ ┌──────┐      ┌────┐       │   (45 tokens)    │
│  │ fs_3 │ │ fs_7 │ │ fs_1 │ ...  │截断 │       │                  │
│  └──────┘ └──────┘ └──────┘      └────┘       │                  │
└────────────────────────────────────────────────┴──────────────────┘
                  ← 总计 3500 tokens →
```

---

# 二、核心数据集生成函数

## 1. generate_gsm8k_dataset（第 267-332 行）

```python
def generate_gsm8k_dataset(
    model_path, source_dataset_path, batch_size, input_len, output_file
):
```

**调用方**: `run_aisbench()` 中 `aisbench_dataset_type == "gsm8k"` 时（也是本测试用例使用的数据集类型）

### 算法流程

```
Step 1: 加载 tokenizer + 读取 GSM8K 源文件
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  dataset = [每条数据的 "question" 字段]

Step 2: 对每条 question 做 token 级填充/截断
  for sentence in dataset:
      words = tokenizer.tokenize(sentence)       # 分词 → list[str]
      len_num = len(words) // input_len           # 原句是目标的几倍？

      if len_num == 0:                            # 原句 < 目标长度
          multiplier = (input_len // len(words)) + 1
          repeated_len = words * multiplier       # 重复整句直到够长
          words = repeated_len[:input_len]        # 截断到精确长度

      decoded_text = tokenizer.convert_tokens_to_string(words)
      dataset_new.append(decoded_text)

Step 3: 调整样本数量到 batch_size
  if len(dataset_new) < batch_size:
      循环重复 dataset_new 直到 >= batch_size，截取前 batch_size 个
  else:
      截取前 batch_size 个

Step 4: 打乱 + 写入 JSONL
  random.shuffle(dataset_new)
  每行: {"question": "<text>", "answer": "none"}
```

### 与 pad_to_target_tokens 的区别

| 维度 | generate_gsm8k_dataset | pad_to_target_tokens |
|------|----------------------|---------------------|
| 填充方式 | **直接重复**问题自身的 token | 用 **few-shot 示例**填充前缀 |
| 数据来源 | 只需要问题文本 | 需要 few-shot 训练集 |
| 多样性 | 低（同一句话反复重复） | 高（不同 few-shot 组合） |
| 算法复杂度 | O(n) 简单 | O(n × m) 需要池搜索 |
| 使用场景 | `run_aisbench()` 的 gsm8k 路径 | `generate_custom_dataset()` |

### 示例

```
原句: "James buys 3 apples for $2 each." (10 tokens)
目标: 35 tokens
处理后: "James buys 3 apples for $2 each. James buys 3 apples for $2 each.
         James buys 3 apples for $2 each. James buys 3 apples for $2 each."
         (10 * 4 = 40 → 截断到 35 tokens)
```

---

## 2. generate_random_dataset（第 335-471 行）

```python
def generate_random_dataset(
    model_path, source_dataset_path, batch_size, input_len,
    output_file, output_len=1024, range_ratio=1,
):
```

**调用方**: `run_aisbench()` 中 `dataset_type == "sharegpt"` 时

### 算法流程

```
Step 1: 随机采样每个请求的 input/output 长度
  input_lens = np.random.randint(
      max(int(input_len * range_ratio), 1),    # 下界
      input_len + 1,                            # 上界
      size=batch_size
  )
  例如: input_len=3500, range_ratio=1 → 全是 3500（无波动）
        input_len=3500, range_ratio=0.8 → [2800, 3500] 之间的随机值

  然后减去 special tokens 数量，避免实际编码后超出目标

Step 2: 检查/下载 ShareGPT 数据集
  if not _is_file_valid_json(source_dataset_path):
      # 自动从 HuggingFace hub 下载
      # anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json

Step 3: 解析 ShareGPT，提取对话
  # 过滤：至少 2 轮对话
  dataset = [data for data in dataset if len(conversations) >= 2]

  # 提取：第一轮 human 提问作为 prompt，第二轮 gpt 回复作为参考
  dataset = [(conversations[0]["value"], conversations[1]["value"]) for ...]

Step 4: 遍历，对每个 prompt 做 token 级调整
  for data in dataset (直到收集够 batch_size 个):
      prompt_token_ids = tokenizer.encode(prompt)

      if len(prompt_token_ids) > input_lens[i]:
          input_ids = prompt_token_ids[:input_lens[i]]      # 截断
      else:
          ratio = (input_lens[i] + len - 1) // len
          input_ids = (prompt_token_ids * ratio)[:input_lens[i]]  # 重复到够长再截断

      input_requests.append({
          "id": str(i),
          "conversations": [
              {"from": "human", "value": tokenizer.decode(input_ids)},
              {"from": "gpt", "value": "none"},
          ]
      })

Step 5: 写入 JSON 数组文件（不是 JSONL！）
  json.dump(input_requests, f)
  # ais_bench 的 ShareGPTDataset 期望的是 JSON 数组，不是 JSONL
```

### 关键设计点

| 设计点 | 说明 |
|--------|------|
| 长度随机化 | `range_ratio < 1` 时每条请求长度不同，模拟真实场景的负载多样性 |
| special tokens 扣除 | 用 `tokenizer.num_special_tokens_to_add()` 预估，防止编码后超长 |
| 自动下载 | ShareGPT 数据集不存在时从 HuggingFace 自动拉取，无需人工准备 |
| 输出格式 | JSON 数组（不是 JSONL），因为 ais_bench 内部用 `json.load()` 读取 |
| prompt 来源 | 取 ShareGPT 对话中第一轮 human 发言，确保是自然语言而非代码或格式化文本 |

---

## 3. generate_custom_dataset（第 95-170 行）

```python
def generate_custom_dataset(
    train_path, test_path, tokenizer_path, target_tokens,
    num_prompts, trust_remote_code=False,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
):
```

**调用方**: `generate_mm_dataset()` 内部使用，也可通过 CLI 直接调用（第 474-521 行 `main()`）

### 算法流程

```
Step 1: 加载 tokenizer，从 train/test JSONL 文件读取数据

Step 2: 调整 test 数据集大小
  if num_prompts > len(test_data):
      # 循环重复直到够
      test_data = (test_data * multiplier)[:num_prompts]
  else:
      test_data = test_data[:num_prompts]

Step 3: 构建 few-shot 池
  few_shot_pool = [format_qa(item) for item in train_data]
  # 例如: ["Question: ...\n...\nAnswer:\n...\n\n", ...]
  few_shot_pool_token_ids = [tokenizer.encode(fs) for fs in few_shot_pool]

Step 4: 对每条 test 数据，调用 pad_to_target_tokens 填充
  for test_item in test_data:
      padded_question = pad_to_target_tokens(
          question=test_item["question"],
          few_shot_pool_token_ids=few_shot_pool_token_ids,
          tokenizer=tokenizer,
          target_tokens=target_tokens,
          test_template=test_template,
      )
      output_data.append({"question": padded_question, "answer": test_item["answer"]})

Step 5: 统计 token 分布（min/max/avg）并返回
```

### 与 generate_gsm8k_dataset 的对比

| 维度 | generate_custom_dataset | generate_gsm8k_dataset |
|------|------------------------|----------------------|
| 填充方式 | few-shot 示例前缀填充 | 自身 token 重复 |
| 数据源 | 需要 train + test 两个 JSONL | 只需要 test JSONL |
| 自然度 | 高（真实 GSM8K 示例作为上下文） | 低（同一句话反复重复） |
| 内存占用 | 高（需加载整个 train set 并 tokenize） | 低（逐条处理） |
| 使用场景 | 精度测试、官方 GSM8K 基准 | 快速性能测试 |

### 生成的 prompt 结构

```
┌──────────────────────────────────────────────────┬────────────────────┐
│                 few-shot prefix                   │     test question  │
│                                                  │                    │
│ Question: A train leaves at 3PM...               │ Question: James    │
│ Let's think step by step                         │ buys 3 apples...   │
│ Answer:                                         │ Let's think step   │
│ The train arrives at 7PM.                        │ by step           │
│                                                  │ Answer:            │
│ Question: If x + y = 10...                       │                    │
│ Let's think step by step                         │                    │
│ Answer:                                         │                    │
│ x = 4, y = 6.                                    │                    │
│ ...                                              │                    │
└──────────────────────────────────────────────────┴────────────────────┘
                       ← 总计 target_tokens tokens →
```

---

## 4. generate_mm_dataset（第 212-264 行）

```python
def generate_mm_dataset(
    train_path, test_path, tokenizer_path, target_tokens=3500,
    num_prompts=1024, trust_remote_code=False,
    test_template="Question: {question}\nLet's think step by step\nAnswer:\n",
    image_dir="/tmp/datasets/image", size=None,
):
```

**调用方**: `run_aisbench()` 中 `dataset_type == "mm-custom-gen"` 时

### 算法流程

```
Step 1: 先生成文本数据集
  text_data = generate_custom_dataset(...)
  # 与纯文本数据集完全相同的 few-shot padding 流程

Step 2: 为每条数据附加随机图片信息
  for item in text_data:
      random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
      item["type"] = "image"
      item["path"] = [f"{image_dir}/{random_string}.jpg"]
      output_data.append(item)

  结果示例:
  {
      "question": "... padded question ...",
      "answer": "He spent 6 dollars.",
      "type": "image",
      "path": ["/tmp/datasets/image/aB3xY7kLmN.jpg"]
  }

Step 3: 生成实际的随机图片文件
  size = tuple(map(int, "1080x1920".split("x")))   # (1080, 1920)
  for item in output_data:
      random_array = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
      img = Image.fromarray(random_array)
      img.save(image_path, quality=95)
```

### 关键设计点

| 设计点 | 说明 |
|--------|------|
| 文本复用 | 多模态数据集本质 = 文本数据集 + 随机图片，文本部分完全复用 `generate_custom_dataset` |
| 随机图片 | 用 `np.random.randint(0, 256, ...)` 生成随机像素，创建纯噪声图片，只测吞吐不测视觉质量 |
| 文件名随机 | 10 位字母数字随机名，避免碰撞 |
| type 标记 | `"type": "image"` 告诉 ais_bench 按多模态模式处理请求 |

---

# 三、数据集选择决策树

在 `run_aisbench()` 中的分支逻辑：

```
run_aisbench(dataset_type, ...):
│
├─ dataset_type == "sharegpt"
│   └─ generate_random_dataset()
│       从 ShareGPT 采真实对话 → token 截断/重复 → 输出 JSON 数组
│       特点: 自然语言 prompt, 支持 range_ratio 随机长度
│
├─ dataset_type == "gsm8k"
│   └─ generate_gsm8k_dataset()
│       从 GSM8K JSONL 取问题 → token 重复到目标长度 → 输出 JSONL
│       特点: 简单快速, 固定长度, answer 恒为 "none"
│
├─ dataset_type == "mm-custom-gen"
│   └─ generate_mm_dataset()
│       └─ generate_custom_dataset()  ← few-shot padding 生成文本
│       └─ generate_random_images()   ← 生成随机 JPEG 图片
│       特点: 需要 train+test 两个文件, 文本自然度高, 图片为噪声
│
└─ dataset_path 已存在 (任意类型)
    └─ 跳过生成，直接使用已有文件
```

---

# 四、测试用例中的实际调用

以 `test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py` 为例：

```python
# 类属性配置
aisbench_dataset_type = "gsm8k"     # 走 generate_gsm8k_dataset
aisbench_dataset_path = None        # 不指定已有路径，触发自动生成
num_prompts = 400
input_len = 3500
output_len = 1500
max_concurrency = 100
```

实际执行 `run_aisbench()` 中的代码路径：

```python
elif dataset_type == "gsm8k" and not dataset_path:
    dataset_file = f"/tmp/datasets/test.jsonl"
    if not os.path.exists(dataset_file):
        generate_gsm8k_dataset(
            model_path="/root/.../Qwen3.6-35B-A3B",
            source_dataset_path="/root/.cache/.../test.jsonl",  # GSM8K 测试集
            batch_size=400,     # num_prompts=400
            input_len=3500,      # 每条输入 3500 tokens
            output_file="/tmp/datasets/test.jsonl"
        )
    dataset_path = dataset_file
```

### 生成结果

```jsonl
{"question": "James buys 3 apples... [重复到3500 tokens]", "answer": "none"}
{"question": "A train leaves at... [重复到3500 tokens]", "answer": "none"}
... (共 400 行)
```

---

# 五、三种填充策略对比

| 策略 | 函数 | 优点 | 缺点 | 适用 |
|------|------|------|------|------|
| **自身重复** | `generate_gsm8k_dataset` | 简单、快速、无需训练集 | prompt 不自然（同一句话反复重复） | 快速性能测试 |
| **few-shot 前缀** | `generate_custom_dataset` | prompt 自然（真实 QA 示例作为上下文） | 需要训练集，内存占用高 | 精度测试 |
| **真实对话截断** | `generate_random_dataset` | prompt 最自然（ShareGPT 真实对话） | 依赖外部数据集下载，需过滤低质量对话 | 通用性能测试 |
