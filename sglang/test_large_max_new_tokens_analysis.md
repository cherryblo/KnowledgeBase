# TestLargeMaxNewTokens 测试脚本逐行解析

## 文件概述

- **测试目标**: 验证 NPU 上大 `max_new_tokens` 场景下的并发请求处理能力
- **测试模型**: Llama-3.2-1B-Instruct（轻量模型，快速加载）
- **核心验证点**: 4 个并发请求能否同时达到 running 状态（并发度 ≥ 4）
- **CI 注册**: `suite="full-2-npu-a3"`, `est_time=400s`, `nightly=True`

---

# 第一部分：导入（第 1-14 行）

```python
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
)
```

| 导入项 | 来源 | 含义 |
|--------|------|------|
| `ThreadPoolExecutor` | `concurrent.futures` | 线程池，用于并发发送多个请求 |
| `openai` | 第三方包 | OpenAI Python SDK，以 OpenAI 兼容 API 方式调 SGLang |
| `kill_process_tree` | `sglang.srt.utils` | 杀死进程及其所有子进程 |
| `get_tokenizer` | `sglang.srt.utils.hf_transformers_utils` | 加载 HuggingFace tokenizer |
| `LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH` | `test_ascend_utils.py:118` | 模型路径，值为 `/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct` |
| `register_npu_ci` | `ci_register.py` | CI 注册装饰器 |
| `DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH` | `test_utils.py` | 服务器启动超时，默认 3600 秒 |
| `DEFAULT_URL_FOR_TEST` | `test_utils.py` | 动态计算的测试 URL，如 `http://127.0.0.1:20066` |
| `STDERR_FILENAME` | `test_utils.py:1701` | 固定值 `"/tmp/stderr.txt"` |
| `STDOUT_FILENAME` | `test_utils.py:1702` | 固定值 `"/tmp/stdout.txt"` |
| `CustomTestCase` | `test_utils.py:2158` | 增强版 TestCase，包装 setUpClass 确保 tearDown 总是执行 |
| `popen_launch_server` | `test_utils.py:861` | 以 subprocess 启动 `sglang serve` 并等待就绪 |

---

# 第二部分：CI 注册（第 16 行）

```python
register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)
```

| 参数 | 值 | 含义 |
|------|-----|------|
| `est_time` | `400` | 预计运行时间 400 秒（约 7 分钟） |
| `suite` | `"full-2-npu-a3"` | 属于 full-2-npu-a3 套件，由 `full-test-npu.yml` 的 `full-2-npu-a3` job 通过 `run_suite.py` 批量执行 |
| `nightly` | `True` | nightly 测试，不在每次 PR 中运行 |

---

# 第三部分：测试类定义（第 19-22 行）

```python
class TestLargeMaxNewTokens(CustomTestCase):
    """Test large max_new_tokens handling on NPU.

    [Test Category] Interface
    [Test Target] large max_new_tokens, concurrent requests
    """
```

- 继承 `CustomTestCase`（非 `TestAscendPerformanceTestCaseBase`），说明这是一个**功能验证测试**而非性能测试
- `[Test Category] Interface` — 测试的是 API 接口层面的行为
- `[Test Target]` — 关注大 `max_new_tokens` + 并发场景

---

# 第四部分：setUpClass（第 24-51 行）

## 第 25-27 行：基础配置

```python
@classmethod
def setUpClass(cls):
    cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    cls.base_url = DEFAULT_URL_FOR_TEST
    cls.api_key = "sk-123456"
```

- `api_key` 设为 `"sk-123456"`，这是一个测试用的假 key，SGLang 在 `--api-key` 模式下要求请求头携带此 key

## 第 29-30 行：打开日志文件

```python
    cls.stdout = open(STDOUT_FILENAME, "w")   # /tmp/stdout.txt
    cls.stderr = open(STDERR_FILENAME, "w")   # /tmp/stderr.txt
```

将服务器进程的 stdout/stderr 重定向到 `/tmp/` 下的文件。**这是此测试的关键设计**——后续通过读取 `stderr.txt` 来监控服务器内部的并发状态（`#running-req` 日志）。

## 第 32-49 行：启动服务器

```python
    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        api_key=cls.api_key,
        other_args=[
            "--max-total-token", "1536",
            "--context-len", "8192",
            "--decode-log-interval", "2",
            "--attention-backend", "ascend",
            "--disable-cuda-graph",
        ],
        env={"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "256", **os.environ},
        return_stdout_stderr=(cls.stdout, cls.stderr),
    )
```

### 服务端参数详解

| 参数 | 值 | 作用 |
|------|-----|------|
| `--max-total-token` | `1536` | 单请求最大总 token 数（input + output）。设为 1536 意味着输入 100 tokens 时，最多输出 1436 tokens |
| `--context-len` | `8192` | 模型最大上下文窗口 |
| `--decode-log-interval` | `2` | **每 2 个 decode token 打印一次日志**。这是监控并发的关键：日志中会包含 `#running-req: N` 来显示当前正在运行的请求数 |
| `--attention-backend` | `ascend` | 使用 Ascend NPU 原生 attention 实现 |
| `--disable-cuda-graph` | (flag) | 禁用 CUDA Graph 编译。大 `max_new_tokens` 场景下，输出长度不确定，禁掉固定 batch size 的 graph 可避免不必要的编译开销 |

### 环境变量

```python
env={"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "256", **os.environ}
```

| 变量 | 值 | 作用 |
|------|-----|------|
| `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION` | `256` | **核心变量**。将 `max_new_tokens` 的估算值裁剪到 256。当用户请求的 `max_tokens` 很大时，调度器在内存规划和批次编排时不会按全量预留，而是按 256 来估算，避免因单个大请求独占资源而阻塞其他请求的调度。这是此测试验证的关键行为 |

### return_stdout_stderr

```python
return_stdout_stderr=(cls.stdout, cls.stderr)
```

将 `popen_launch_server` 内部创建的 subprocess 的 stdout/stderr 重定向到已打开的两个文件。`popen_launch_server` 内部（`test_utils.py` 中 `_launch_server_process`）会将这两个 file handle 传给 `subprocess.Popen` 的 `stdout` 和 `stderr` 参数。

## 第 50-51 行：URL 拼接 + 加载 tokenizer

```python
    cls.base_url += "/v1"   # http://127.0.0.1:20066/v1
    cls.tokenizer = get_tokenizer(LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH)
```

- URL 加上 `/v1` 前缀，与 OpenAI API 兼容路径对齐
- 加载 tokenizer（虽然后续代码中未直接使用，但可能是为扩展预留）

---

# 第五部分：tearDownClass（第 53-58 行）

```python
@classmethod
def tearDownClass(cls):
    kill_process_tree(cls.process.pid)
    cls.stdout.close()
    cls.stderr.close()
    os.remove(STDOUT_FILENAME)
    os.remove(STDERR_FILENAME)
```

- 杀死 `sglang serve` 进程树
- 关闭并删除 `/tmp/stdout.txt` 和 `/tmp/stderr.txt`，确保每次测试有干净的日志环境

---

# 第六部分：run_chat_completion（第 60-73 行）

```python
def run_chat_completion(self):
    client = openai.Client(api_key=self.api_key, base_url=self.base_url)
    response = client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant"},
            {
                "role": "user",
                "content": "Please repeat the word 'hello' for 100 times.",
            },
        ],
        temperature=0,
    )
    return response
```

### 逐行解析

1. **`openai.Client(api_key="sk-123456", base_url="http://127.0.0.1:20066/v1")`**：创建 OpenAI SDK 客户端，实际请求会打到本地的 SGLang 服务器

2. **`client.chat.completions.create()`**：调用 Chat Completions API（对应 `/v1/chat/completions` 端点）

3. **`temperature=0`**：贪婪解码，确保输出确定性

4. **关键点**：没有设置 `max_tokens` 参数。由于服务端配置了 `--max-total-token 1536`，而输入 prompt 很短（约 30 tokens），`max_tokens` 会被自动推断为 `1536 - 30 ≈ 1500`。

   但在 `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=256` 的作用下，调度器在估算资源占用时会将 `max_new_tokens` 裁剪为 **256**，这意味着调度器认为这些请求不会占用太多 decode 槽位，从而允许更多请求同时运行。

> 此方法在 `ThreadPoolExecutor` 的线程中被调用，`openai.Client` 的 HTTP 请求是**同步阻塞**的。线程池的意义在于让多个请求同时发出并各自等待流式响应。

---

# 第七部分：test_chat_completion（第 75-116 行）

## 第 76-82 行：初始化变量

```python
def test_chat_completion(self):
    num_requests = 4          # 发送 4 个并发请求
    min_concurrent = 4        # 期望至少 4 个请求同时 running

    futures = []
    max_running_reqs = 0
    all_requests_running = False
    start_time = time.time()
    max_wait_time = 300       # 最多等待 5 分钟
```

## 第 84 行：创建线程池并提交任务

```python
    with ThreadPoolExecutor(num_requests) as executor:
        for i in range(num_requests):
            futures.append(executor.submit(self.run_chat_completion))
```

`ThreadPoolExecutor(4)` 创建 4 个工作线程：

```
主线程                    线程1               线程2               线程3               线程4
  │                        │                  │                  │                  │
  ├─ submit(task1) ──────► ├─ run_chat()      │                  │                  │
  ├─ submit(task2) ──────► │                  ├─ run_chat()      │                  │
  ├─ submit(task3) ──────► │                  │                  ├─ run_chat()      │
  ├─ submit(task4) ──────► │                  │                  │                  ├─ run_chat()
  │                        │ POST /v1/chat..   │ POST /v1/chat..   │ POST /v1/chat..   │ POST /v1/chat..
  │                        │    (阻塞等待)      │    (阻塞等待)      │    (阻塞等待)      │    (阻塞等待)
```

4 个请求几乎同时到达服务器。由于 `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=256`，调度器认为它们只需要 256 个 decode 槽位，全部被接纳进 running 队列。

## 第 86-107 行：轮询 stderr 监控并发状态

```python
        pt = 0
        while pt >= 0:
            time.sleep(5)
            if time.time() - start_time > max_wait_time:
                print(f"Timeout after {max_wait_time} seconds")
                pt = -1
                break
            lines = open(STDERR_FILENAME).readlines()
            for line in lines[pt:]:
                print(line, end="", flush=True)
                if "#running-req:" in line:
                    import re
                    match = re.search(r"#running-req:\s*(\d+)", line)
                    if match:
                        current = int(match.group(1))
                        max_running_reqs = max(max_running_reqs, current)
                        if current >= min_concurrent:
                            all_requests_running = True
                            pt = -1
                            break
                pt += 1
```

### 轮询机制详解

```
stderr.txt 文件（服务器不断追加日志）:

... 之前的日志 ...
[2026-06-04 10:00:01] Decode batch. #running-req: 1, ...
[2026-06-04 10:00:02] Decode batch. #running-req: 2, ...    ← pt=最初指向这里
[2026-06-04 10:00:03] Decode batch. #running-req: 3, ...
[2026-06-04 10:00:04] Decode batch. #running-req: 4, ...    ← 匹配到 current=4 ≥ min_concurrent=4
```

1. **`pt` 指针**：记录已读取到的行号，每次只读新增的行（`lines[pt:]`），避免重复处理
2. **`time.sleep(5)`**：每 5 秒检查一次日志，平衡响应性和 CPU 开销
3. **正则 `#running-req:\s*(\d+)`**：从日志中提取当前 running 的请求数
4. **`max_running_reqs = max(...)`**：持续跟踪峰值并发
5. **`pt = -1` → 跳出 while 循环**：一旦 `current >= 4`，立即退出轮询

### 退出条件

| 条件 | 触发 | 结果 |
|------|------|------|
| `current >= min_concurrent(4)` | 4 个请求同时 running | `all_requests_running = True`，正常退出 |
| `time.time() - start_time > 300` | 超时 5 分钟 | `all_requests_running = False`，断言失败 |

## 第 109-112 行：断言

```python
    assert (
        all_requests_running
    ), f"At least {min_concurrent} requests should be running concurrently, but max was {max_running_reqs}"
```

如果 5 分钟内日志中从未出现 `#running-req: 4`（或更高），测试失败并报告实际达到的最大并发数。

---

# 完整运行时序

```
python test_large_max_new_tokens.py
│
├─ setUpClass()
│   ├─ 打开 /tmp/stdout.txt, /tmp/stderr.txt
│   ├─ popen_launch_server(
│   │       model=Llama-3.2-1B-Instruct,
│   │       other_args=[
│   │           --max-total-token 1536,
│   │           --context-len 8192,
│   │           --decode-log-interval 2,      ← 每 2 token 打印 #running-req
│   │           --attention-backend ascend,
│   │           --disable-cuda-graph,
│   │       ],
│   │       env={SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION: 256}  ← 核心！
│   │       return_stdout_stderr=(stdout, stderr)
│   │   )
│   │   └─ sglang serve --model-path <model> --max-total-token 1536 ...
│   │       等待 GET /health → 200
│   └─ base_url += "/v1"
│
├─ test_chat_completion()
│   ├─ ThreadPoolExecutor(4) 创建 4 个线程
│   ├─ executor.submit(run_chat_completion) × 4
│   │   └─ 每个线程: POST /v1/chat/completions
│   │       {
│   │         "messages": [..., "repeat 'hello' for 100 times"],
│   │         "temperature": 0
│   │         // 无 max_tokens → 自动推断 ≈1500 → 被 CLIP 裁为 256
│   │       }
│   │
│   ├─ while 轮询 /tmp/stderr.txt
│   │   ├─ 每 5 秒读一次新增行
│   │   ├─ 正则匹配 "#running-req: N"
│   │   └─ N >= 4 → 退出轮询, all_requests_running=True
│   │
│   └─ assert all_requests_running == True
│
└─ tearDownClass()
    ├─ kill_process_tree(pid)
    ├─ close stdout/stderr
    └─ remove /tmp/stdout.txt, /tmp/stderr.txt
```

---

# 设计要点分析

## 1. 为什么用 `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=256`？

这是此测试的核心逻辑。在 SGLang 调度器内部，当一个请求没有显式设置 `max_tokens` 时，会从 `max_total_tokens (1536) - input_len (≈30)` 推断出 `max_new_tokens ≈ 1500`。调度器在资源规划时，会按 1500 来预留 decode 槽位。4 个请求各需要 1500 个槽位，总共 6000 个——可能超出 NPU 的计算能力，调度器会拒绝部分请求进入 running 队列。

`SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=256` 将每个请求的估算值裁剪到 256，4 个请求总共只需 1024 个槽位，调度器将它们全部接纳。

## 2. 为什么用 `--decode-log-interval 2`？

默认的日志间隔可能太大（如每 100 个 token 才打印一次），在 300 秒超时内可能来不及观测到 `#running-req: 4` 的状态。设为 2 意味着每生成 2 个 token 就打印一次日志，几乎可以实时反映并发状态。

## 3. 为什么用 stderr 而非 stdout？

SGLang 的运行时日志（包括 `#running-req`）默认输出到 stderr。stdout 通常留给 API 访问日志。

## 4. 为什么用 file 轮询而非 signal/event？

- **简单可靠**：不需要修改 SGLang 服务器代码添加回调机制
- **CI 兼容**：CI 环境中多数 IPC 机制不可用或受限
- **可审计**：日志文件同时作为测试证据保存，便于事后分析

## 5. 与性能测试的区别

| 维度 | 此测试（功能） | 性能测试（如 Qwen3.6） |
|------|---------------|----------------------|
| 基类 | `CustomTestCase` | `TestAscendPerformanceTestCaseBase` |
| 启动方式 | `popen_launch_server` + 自定义 stderr 监控 | `popen_launch_server`（基类 setUpClass 中） |
| 请求方式 | `openai.Client` SDK（线程池同步调用） | aisbench / bench_serving（外部工具） |
| 验证方式 | 解析 server 内部日志确认并发 | 解析压测工具的指标值 |
| 关注点 | 并发是否正确达成 | 吞吐量/TPOT/TTFT 是否达标 |
