# run_bench_serving 方法详细解析

## 概述

- **文件位置**: `python/sglang/test/ascend/e2e/test_npu_performance_utils.py:319-420`
- **调用方**: `TestAscendPerformanceTestCaseBase.run_throughput()` 中当 `benchmark_tool != "aisbench"` 时
- **作用**: 以子进程方式调用 `python -m sglang.bench_serving`，实时解析 stdout 提取性能指标（Mean TTFT、Mean TPOT、Output token throughput、Mean E2E Latency）

---

# 第一部分：run_bench_serving 方法本体（第 319-420 行）

## 第 319-337 行：函数签名与参数

```python
def run_bench_serving(
    host,                    # 服务器 IP，如 "127.0.0.1"
    port,                    # 服务器端口，如 20066
    model_path=None,         # 模型路径，传给 bench_serving 的 --model 参数
    backend="sglang",        # 后端类型: sglang / sglang-oai / vllm / trt 等
    dataset_name=None,       # 数据集名: random / sharegpt / gsm8k 等
    dataset_path=None,       # 数据集文件路径
    request_rate=None,       # 请求速率 (req/s)，None = 最大压力
    max_concurrency=None,    # 最大并发数
    num_prompts=None,        # 总请求数
    input_len=None,          # 输入 token 长度
    output_len=None,         # 输出 token 长度
    random_range_ratio=1,    # 输入/输出长度随机波动比例
    image_resolution=None,   # 多模态图片分辨率
    image_count=None,        # 多模态图片数量
    warmup_requests=None,    # 预热请求数
    seed=None,               # 随机种子
    output_file=None,        # 结果输出文件
):
```

所有参数均为可选（都有默认值），实际由 `run_throughput()` 中的 `bench_params` 字典填充。

---

## 第 338-344 行：确定结果文件路径

```python
metrics_path = os.getenv("METRICS_DATA_FILE")
result_file = (
    "./bench_log.txt"
    if not metrics_path
    else f"{metrics_path}/bench_serving_metrics.txt"
)
logger.info(f"The metrics result file: {result_file}")
```

- `METRICS_DATA_FILE` 由 workflow `nightly-test-npu-e2e-single-node.yml` 设置，格式为 `/root/.cache/tests/output/perf/20260603/<test_case_name>`
- 如果未设置（本地手动运行），降级为当前目录的 `./bench_log.txt`

---

## 第 346 行：写入软件包版本信息

```python
write_pkg_info_to_file(result_file)
```

### write_pkg_info_to_file（第 281-316 行）做了什么：

1. 执行 `pip list` 获取所有已安装包
2. 按 `PACKAGE_FILTER_KEYWORDS = ["sglang", "sgl", "torch", "deep-ep", "memfabric_hybrid"]` 过滤
3. 写入结果文件，包括：
   - 过滤后的 pip 包列表
   - CANN 版本号（从 `/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/ascend_toolkit_install.info` 读取）
   - transformers 版本号

**目的**: 事后可追溯到压测时的确切软件环境，排查性能回归时能定位到具体包版本。

---

## 第 348-360 行：构建命令前缀（必选参数）

```python
cmd_args = [
    PYTHON_FOR_TEST_TOOL,         # "python3"
    "-m",
    "sglang.bench_serving",        # 调用 sglang 内置 benchmark 模块
    "--host", host,
    "--port", str(port),
    "--model", model_path,
    "--backend", backend,
]
```

`PYTHON_FOR_TEST_TOOL` 在文件顶部（第 54-58 行）定义：
```python
PYTHON_FOR_TEST_TOOL = "python_venv_for_test_tool/bin/python"
if not os.path.exists(PYTHON_FOR_TEST_TOOL):
    PYTHON_FOR_TEST_TOOL = "python3"
```

优先使用虚拟环境中的 Python（隔离依赖），不存在则用系统 python3。

---

## 第 362-387 行：拼接可选参数

```python
if dataset_name:     cmd_args.extend(["--dataset-name", str(dataset_name)])
if dataset_path:     cmd_args.extend(["--dataset-path", str(dataset_path)])
if request_rate:     cmd_args.extend(["--request-rate", str(request_rate)])
if max_concurrency:  cmd_args.extend(["--max-concurrency", str(max_concurrency)])
if num_prompts:      cmd_args.extend(["--num-prompts", str(num_prompts)])
if input_len:        cmd_args.extend(["--random-input-len", str(input_len)])
if output_len:       cmd_args.extend(["--random-output-len", str(output_len)])
if random_range_ratio: cmd_args.extend(["--random-range-ratio", str(random_range_ratio)])
if image_resolution: cmd_args.extend(["--image-resolution", str(image_resolution)])
if image_count:      cmd_args.extend(["--image-count", str(image_count)])
if warmup_requests:  cmd_args.extend(["--warmup-requests", str(warmup_requests)])
if seed:             cmd_args.extend(["--seed", str(seed)])
if output_file:      cmd_args.extend(["--output-file", str(output_file)])
```

**关键差异点**：`run_bench_serving` 的参数名与 `run_aisbench` 不同：

| 参数含义 | run_bench_serving 的 CLI 参数 | run_aisbench 的 CLI 参数 |
|----------|-------------------------------|--------------------------|
| 输入长度 | `--random-input-len` | `--input-len` |
| 输出长度 | `--random-output-len` | `--output-len` |
| 并发数 | `--max-concurrency` | `--batch-size` |

---

## 第 390-395 行：启动子进程

```python
metrics = {"mean_ttft": None, "mean_tpot": None, "total_tps": None}

process = subprocess.Popen(
    cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1
)
```

| 参数 | 含义 |
|------|------|
| `stdout=subprocess.PIPE` | 捕获标准输出 |
| `stderr=subprocess.STDOUT` | 标准错误合并到标准输出（统一解析） |
| `text=True` | 以文本模式读取（bytes → str） |
| `bufsize=1` | 行缓冲，每行立即可读，确保实时输出 |
| `metrics` 初始化 | 三个指标默认为 `None`，未解析到时保持 `None` |

---

## 第 396-419 行：逐行读取输出并提取指标

```python
with open(result_file, "a", encoding="utf-8") as f:
    for line in process.stdout:
        if line.strip():
            print(line, end="")          # 实时打印到 CI 日志
        f.write(line)                     # 同时写入结果文件
        stripped_line = line.strip()

        # 用字符串包含匹配提取指标
        if "Mean TTFT" in stripped_line:
            parts = stripped_line.split()
            if len(parts) >= 4:
                metrics["mean_ttft"] = parts[3]
        elif "Mean TPOT" in stripped_line:
            parts = stripped_line.split()
            if len(parts) >= 4:
                metrics["mean_tpot"] = parts[3]
        elif "Output token throughput" in stripped_line:
            parts = stripped_line.split()
            if len(parts) >= 5:
                metrics["total_tps"] = parts[4]
        elif "Mean E2E Latency" in stripped_line:
            parts = stripped_line.split()
            if len(parts) >= 5:
                metrics["mean_e2e_latency"] = parts[4]
```

### 指标提取解析

`bench_serving.py` 的 `benchmark()` 函数末尾（第 1472-1573 行）用固定格式打印结果：

```
========================= Serving Benchmark Result ========================
Backend:                                 sglang
...
-----------------------Time to First Token------------------------------
Mean TTFT (ms):                          45.23
Median TTFT (ms):                        44.10
P99 TTFT (ms):                           52.67
-------------------Time per Output Token (excl. 1st token)-----------------
Mean TPOT (ms):                          10.56
Median TPOT (ms):                        10.21
P99 TPOT (ms):                           19.87
...
Output token throughput (tok/s):         2031.71
...
----------------------End-to-End Latency-------------------------------
Mean E2E Latency (ms):                   16000.00
...
```

`run_bench_serving` 的指标提取逻辑就是匹配这些固定格式的输出行：

| 指标 | 匹配行 | 取值位置 | 示例值 |
|------|--------|----------|--------|
| `mean_ttft` | `Mean TTFT (ms): 45.23` | `split()[3]` | `"45.23"` |
| `mean_tpot` | `Mean TPOT (ms): 10.56` | `split()[3]` | `"10.56"` |
| `total_tps` | `Output token throughput (tok/s): 2031.71` | `split()[4]` | `"2031.71"` |
| `mean_e2e_latency` | `Mean E2E Latency (ms): 16000.00` | `split()[4]` | `"16000.00"` |

### 与 run_aisbench 指标提取的对比

| 维度 | run_bench_serving | run_aisbench |
|------|-------------------|--------------|
| 提取方式 | `split()` 按空格切分取固定位置 | 正则表达式 `re.search()` |
| 容错性 | 低（格式变了就失败） | 高（正则匹配更鲁棒） |
| 输出实时性 | 边运行边解析，实时打印 | 运行完后从完整输出解析 |
| 结果归档 | 所有 stdout 写入 `result_file` | aisbench 自己写结果，`run_aisbench` 只解析 |

---

## 第 410-420 行：等待进程结束并返回

```python
process.wait()
if process.returncode != 0:
    logger.error(
        f"Benchmark command failed with return code: {process.returncode}"
    )
except Exception as e:
    logger.error(f"Error running benchmark: {e}")
finally:
    if process.stdout is not None and not process.stdout.closed:
        process.stdout.close()

return metrics
```

- 阻塞等待子进程结束
- 非零退出码只记录错误，**不抛异常**（允许 `assert_metrics` 中因缺失指标而失败）
- `finally` 确保 stdout 管道关闭

---

# 第二部分：sglang.bench_serving 内部流程

`python -m sglang.bench_serving` 的核心执行链路：

```
run_benchmark(args)                                    ← 入口（第 1689 行）
  │
  ├─ Step 1: 设置随机种子、构建 extra_request_body
  ├─ Step 2: 根据 backend 类型确定 API URL
  │     "sglang"      → http://{host}:{port}/generate
  │     "sglang-oai"  → http://{host}:{port}/v1/completions
  │     "sglang-oai-chat" → http://{host}:{port}/v1/chat/completions
  │
  ├─ Step 3: 等待服务器就绪
  │     GET http://{host}:{port}/v1/models   → 200
  │
  ├─ Step 4: 获取模型列表，自动推断 model name
  │
  ├─ Step 5: 加载数据集
  │     get_dataset(args, tokenizer, model_id)
  │     ├── dataset_name="random"  → 生成随机 token 数据
  │     ├── dataset_name="sharegpt" → 读取 ShareGPT JSON
  │     └── dataset_name="custom"  → 从 dataset_path 加载
  │
  ├─ Step 6: asyncio.run(benchmark(...))              ← 核心异步函数
  │     │
  │     ├── 6.1 根据 backend 选择 request_func
  │     │     ASYNC_REQUEST_FUNCS = {
  │     │       "sglang": async_request_sglang_generate,
  │     │       "sglang-oai": async_request_openai_completions,
  │     │       "sglang-oai-chat": async_request_openai_chat_completions,
  │     │       "vllm": async_request_openai_completions,
  │     │       "trt": async_request_trt_llm,
  │     │       ...
  │     │     }
  │     │
  │     ├── 6.2 创建 asyncio.Semaphore(max_concurrency) 控制并发
  │     │
  │     ├── 6.3 Warmup: 发送 warmup_requests 个简单请求
  │     │     确保服务器完全就绪、编译缓存预热
  │     │
  │     ├── 6.4 Flush cache (CI 模式下)
  │     │     POST /flush_cache → 清空 Radix Cache
  │     │
  │     ├── 6.5 (可选) 启动 profiler
  │     │     POST /start_profile
  │     │
  │     ├── 6.6 按 request_rate 生成请求流（异步生成器）
  │     │     get_request(input_requests, request_rate):
  │     │       for request in input_requests:
  │     │           yield request
  │     │           interval = np.random.exponential(1.0 / request_rate)
  │     │           await asyncio.sleep(interval)
  │     │     # request_rate=inf 时不 sleep，发送最大压力
  │     │
  │     ├── 6.7 并发执行所有请求
  │     │     tasks = [asyncio.create_task(limited_request_func(...)) for ...]
  │     │     outputs = await asyncio.gather(*tasks)
  │     │
  │     ├── 6.8 (可选) 停止 profiler
  │     │     POST /stop_profile
  │     │
  │     ├── 6.9 获取 speculative decoding accept_length (如果启用了推测解码)
  │     │     GET /server_info → avg_spec_accept_length
  │     │
  │     ├── 6.10 calculate_metrics(input_requests, outputs, dur_s, tokenizer)
  │     │     ├── 统计 TTFT / TPOT / ITL / E2E Latency 的 mean/median/std/p99
  │     │     ├── 计算 request_throughput / input_throughput / output_throughput
  │     │     ├── 逐秒统计 tokens_per_second 和 concurrent_requests
  │     │     └── 返回 BenchmarkMetrics dataclass
  │     │
  │     └── 6.11 打印格式化结果到 stdout
  │
  └─ 返回 metrics
```

---

# 第三部分：async_request_openai_completions 请求函数详解（第 232-343 行）

当 `backend="sglang-oai"` 时使用此函数。以 OpenAI Completions API 格式发送流式请求。

### Step 1: 构造 payload

```python
payload = {
    "model": request_func_input.model,
    "prompt": prompt,
    "best_of": 1,
    "max_tokens": request_func_input.output_len,
    "stream": not args.disable_stream,       # 默认开启流式
    "temperature": 0.0,                       # 贪婪解码（确定性输出）
    "ignore_eos": not args.disable_ignore_eos,
}
payload.update(request_func_input.extra_request_body)  # 合并额外参数
```

### Step 2: 发送 POST 请求并逐 chunk 解析

```python
async with session.post(url=api_url, json=payload, headers=headers) as response:
    async for chunk_bytes in response.content:
        chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
        # SSE 格式: "data: {"choices":[...]}\n\n"

        if chunk == "[DONE]":
            pass   # 流结束信号
        else:
            data = json.loads(chunk)
            timestamp = time.perf_counter()

            if ttft == 0.0:
                # 第一个 token → 记录 TTFT
                ttft = timestamp - st
                output.ttft = ttft
            else:
                # 后续 token → 记录 ITL
                output.text_chunks.append(data["choices"][0]["text"])
                output.itl.append(timestamp - most_recent_timestamp)

            most_recent_timestamp = timestamp
```

### 时序图解

```
时间轴:
st ────────────────────────────────────────────────→

st        t1        t2        t3        t4
│         │         │         │         │
├─────────┼─────────┼─────────┼─────────┤
│  TTFT   │  ITL1   │  ITL2   │  ITL3   │
│ (50ms)  │ (10ms)  │ (11ms)  │ (9ms)   │
│         │         │         │         │
└── chunk1(Hello)    chunk2(world) chunk3(!) ───→ [DONE]

latency = t_last - st = 180ms
output_len = 4 (4 个 token)
TTFT = t1 - st = 50ms
TPOT = (latency - TTFT) / (output_len - 1) = (180-50)/3 = 43.3ms
ITL = [10, 11, 9] ms
```

---

# 第四部分：calculate_metrics 指标计算（第 964-1133 行）

## TTFT（Time To First Token）

```python
ttfts.append(outputs[i].ttft)  # 直接来自请求函数记录的 ttft

# 统计输出:
mean_ttft_ms   = np.mean(ttfts) * 1000
median_ttft_ms = np.median(ttfts) * 1000
p99_ttft_ms    = np.percentile(ttfts, 99) * 1000
```

## TPOT（Time Per Output Token，排除第一个 token）

```python
if output_len > 1:
    tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
# TPOT = (总延迟 - 首token延迟) / (总token数 - 1)
```

## ITL（Inter-Token Latency，token 间延迟）

```python
itls += outputs[i].itl  # 每个 output 的 itl 列表是 token 间的时间间隔

mean_itl_ms   = np.mean(itls) * 1000
p99_itl_ms    = np.percentile(itls, 99) * 1000
max_itl_ms    = np.max(itls) * 1000
```

## 吞吐量

```python
request_throughput  = completed / dur_s                    # 请求吞吐 (req/s)
input_throughput    = total_input / dur_s                   # 输入吞吐 (tok/s)
output_throughput   = sum(output_lens) / dur_s              # 输出吞吐 (tok/s)
total_throughput    = (total_input + sum(output_lens)) / dur_s  # 总吞吐 (tok/s)
```

## 峰值并发与峰值吞吐

```python
# 逐秒统计
for output in successful_outputs:
    for token_time in token_times:
        second_bucket = int(token_time - min_start_time)
        tokens_per_second[second_bucket] += 1

    for second in range(request_start_second, request_end_second + 1):
        concurrent_requests_per_second[second] += 1

max_output_tokens_per_s = float(np.max(tokens_per_second))
max_concurrent_requests = int(np.max(concurrent_requests_per_second))
```

---

# 第五部分：run_bench_serving vs run_aisbench 完整对比

| 维度 | run_bench_serving | run_aisbench |
|------|-------------------|--------------|
| **底层工具** | `python -m sglang.bench_serving` (Python 异步) | `bash run_aisbench.sh` → aisbench CLI (外部独立工具) |
| **数据处理** | `get_dataset()` 内置多种数据集加载器 | 先生成数据集 JSONL 文件，再传给 aisbench |
| **请求发送** | `asyncio` + `aiohttp`，支持 Semaphore 精确定义并发 | aisbench 自行管理线程/进程池发送 |
| **后端支持** | sglang-oai, sglang, vllm, trt, lmdeploy 等 6+ 种 | sglang（固定） |
| **指标提取** | `split()` 取固定位置 | 正则 `re.search()` 匹配 |
| **输出格式** | 控制台格式化表格 | 自定义格式 + JSON |
| **优点** | 不需要额外工具依赖，与 sglang 开发同步 | 输出更丰富（Concurrency、Request Throughput 等），正则匹配更鲁棒 |
| **测试中使用** | 较少（大多数测试用 aisbench） | **默认**（`BENCHMARK_TOOL_DEFAULT = "aisbench"`） |
