# test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py 逐行分析

## 文件概述

- **路径**: `test/registered/ascend/performance/test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py`
- **测试模型**: Qwen3.6-35B-A3B（MoE 架构，35B 参数，3B 激活）
- **测试场景**: 单机 TP=2，输入 3500 tokens / 输出 1500 tokens，TPOT ≤ 50ms
- **继承链**: `TestNPUQwen3_6_35BA3B_1P_In3k5_Out1k5_50ms` → `TestAscendPerformanceTestCaseBase` → `CustomTestCase` → `unittest.TestCase`

---

# 第一部分：测试用例文件本身（共 101 行）

## 第 1-9 行：导入模块

```python
import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_6_35B_A3B_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci
```

| 导入项 | 来源 | 含义 |
|--------|------|------|
| `unittest` | Python 标准库 | 测试框架 |
| `AISBENCHMARK_DATASET_DEFAULT` | `test_npu_performance_utils.py:44` | 默认数据集类型，值为 `"gsm8k"` |
| `BENCHMARK_TOOL_DEFAULT` | `test_npu_performance_utils.py:40` | 默认压测工具，值为 `"aisbench"` |
| `QWEN3_6_35B_A3B_MODEL_PATH` | `test_npu_performance_utils.py:89` | 模型路径: `/root/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B` |
| `TestAscendPerformanceTestCaseBase` | `test_npu_performance_utils.py:741` | 单机性能测试基类 |
| `register_npu_ci` | `ci_register.py` | CI 注册装饰器，标记此测试的元数据 |

---

## 第 11-16 行：CI 注册

```python
register_npu_ci(
    est_time=3600,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="performance testcase",
)
```

一个装饰器式调用，将测试元数据注册到 CI 系统：

| 参数 | 值 | 含义 |
|------|-----|------|
| `est_time` | `3600` | 预计运行时间 3600 秒（1 小时），用于 CI 调度估算 |
| `suite` | `"nightly-2-npu-a3"` | 所属测试套件名称 |
| `nightly` | `True` | 标记为 nightly 测试（而非 PR 触发） |
| `disabled` | `"performance testcase"` | 禁用原因备注 |

> `disabled` 非空意味着在常规 CI 运行中被跳过，只在显式调用（如 full-test-npu.yml 中的 POC 测试）时运行。这是因为完整性能测试耗时长，不适合每次提交都跑。

---

## 第 18-29 行：环境变量配置

```python
QWEN3_6_35B_A3B_3K5_1K5_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "50",
}
```

| 变量 | 值 | 作用 |
|------|-----|------|
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` | PyTorch NPU 内存分配器配置，启用可扩展段以减少显存碎片 |
| `STREAMS_PER_DEVICE` | `32` | 每个 NPU 设备的 CUDA-like stream 数量 |
| `HCCL_SOCKET_IFNAME` | `lo` | 集合通信库 HCCL 使用的网卡接口，`lo` 表示单机（走 loopback，不经过物理网卡） |
| `GLOO_SOCKET_IFNAME` | `lo` | Gloo 分布式通信后端使用的网卡接口，同单机走 loopback |
| `HCCL_OP_EXPANSION_MODE` | `AIV` | HCCL 算子扩展模式，`AIV` 表示使用 Ascend Intelligent Vision 加速 |
| `SGLANG_SET_CPU_AFFINITY` | `1` | 启用 CPU 亲和性绑定 |
| `SGLANG_ENABLE_SPEC_V2` | `1` | 启用 speculative decoding V2 版本（NEXTN 推测解码） |
| `SGLANG_ENABLE_OVERLAP_PLAN_STREAM` | `1` | 启用 plan stream 与计算 stream 的重叠执行 |
| `ASCEND_USE_FIA` | `1` | 启用 Ascend 的 FIA（Fused Infer Attention）加速 |
| `SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES` | `50` | prefill delayer 最大延迟轮次，控制 prefill 批次合并策略 |

---

## 第 31-75 行：服务启动参数配置

```python
QWEN3_6_35B_A3B_3K5_1K5_OTHER_ARGS = [
    "--tp-size", 2,
    "--nnodes", 1,
    "--attention-backend", "ascend",
    "--device", "npu",
    "--chunked-prefill-size", -1,
    "--max-prefill-tokens", 35000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--enable-prefill-delayer",
    "--max-running-requests", 100,
    "--max-mamba-cache-size", 105,
    "--mem-fraction-static", 0.78,
    "--cuda-graph-bs", 4, 16, 32, 64, 84, 100,
    "--enable-multimodal",
    "--mm-attention-backend", "ascend_attn",
    "--dtype", "bfloat16",
    "--mamba-ssm-dtype", "bfloat16",
    "--speculative-algorithm", "NEXTN",
    "--speculative-num-steps", 3,
    "--speculative-eagle-topk", 1,
    "--speculative-num-draft-tokens", 4,
]
```

会被传给 `sglang serve` 命令。逐项解析：

| 参数 | 值 | 作用 |
|------|-----|------|
| `--tp-size` | `2` | Tensor Parallel 度为 2（在 2 张 NPU 卡上切分模型权重） |
| `--nnodes` | `1` | 单节点（不跨机器） |
| `--attention-backend` | `ascend` | 使用 Ascend NPU 原生 attention 实现 |
| `--device` | `npu` | 设备类型为 NPU |
| `--chunked-prefill-size` | `-1` | `-1` 表示不限制 chunked prefill 的 chunk 大小 |
| `--max-prefill-tokens` | `35000` | 单次 prefill 最大 token 数上限 |
| `--disable-radix-cache` | (flag) | 禁用 RadixAttention 前缀缓存。性能测试中为获得稳定可复现结果，通常关闭缓存 |
| `--trust-remote-code` | (flag) | 信任 HuggingFace 模型仓库中的远程代码执行 |
| `--enable-prefill-delayer` | (flag) | 启用 prefill 延迟合并，将多个 prefill 请求攒批后一起执行，提升吞吐 |
| `--max-running-requests` | `100` | 最大并发请求数 |
| `--max-mamba-cache-size` | `105` | Mamba SSM cache 最大 size |
| `--mem-fraction-static` | `0.78` | 静态内存分配比例为 78%（NPU 显存的 78% 预留给 KV Cache 和权重） |
| `--cuda-graph-bs` | `4 16 32 64 84 100` | CUDA Graph 捕获的 batch size 列表。NPU 上等价于编译优化，预生成这些 bs 下的执行图 |
| `--enable-multimodal` | (flag) | 启用多模态支持（Qwen3.6 支持图文输入） |
| `--mm-attention-backend` | `ascend_attn` | 多模态 attention 后端使用 Ascend 实现 |
| `--dtype` | `bfloat16` | 模型推理精度为 BF16 |
| `--mamba-ssm-dtype` | `bfloat16` | Mamba SSM 部分精度为 BF16 |
| `--speculative-algorithm` | `NEXTN` | 推测解码算法：EAGLE 风格的 NEXTN 预测下一个 token |
| `--speculative-num-steps` | `3` | 每次推测最多 3 步（每步预测 draft token） |
| `--speculative-eagle-topk` | `1` | EAGLE 推测解码的 top-k 采样数 |
| `--speculative-num-draft-tokens` | `4` | 每次推测生成 4 个 draft token（最终通过 target model 验证） |

---

## 第 78-93 行：测试类定义

```python
class TestNPUQwen3_6_35BA3B_1P_In3k5_Out1k5_50ms(TestAscendPerformanceTestCaseBase):
    """Test NPU performance for Qwen3.6-35B-A3B 1p in3k5 out1k5 50ms"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT      # "aisbench"
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT  # "gsm8k"
    model = QWEN3_6_35B_A3B_MODEL_PATH           # "/root/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B"
    other_args = QWEN3_6_35B_A3B_3K5_1K5_OTHER_ARGS
    envs = QWEN3_6_35B_A3B_3K5_1K5_ENVS
    dataset_name = "random"
    max_concurrency = 100
    num_prompts = 400
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50                                    # 期望 TPOT ≤ 50ms
    output_token_throughput = 2031.71            # 期望吞吐 ≥ 2031.71 tokens/s

    def test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms(self):
        """Run NPU performance test for Qwen3.6-35B-A3B in3k5 out1k5 50ms"""
        self.run_throughput()
```

### 类属性解析

这些类属性**覆盖**了基类 `TestAscendPerformanceTestCaseBase` 中的默认值（默认值见下文第二部分）：

| 属性 | 设置值 | 来源/基类默认值 | 说明 |
|------|--------|-----------------|------|
| `benchmark_tool` | `"aisbench"` | 基类: `BENCHMARK_TOOL_DEFAULT` = `"aisbench"` | 压测工具选择 |
| `aisbench_dataset_type` | `"gsm8k"` | 基类: `"gsm8k"` | aisbench 数据集类型 |
| `model` | Qwen3.6-35B-A3B 路径 | 基类: `None` | 模型在节点上的绝对路径 |
| `other_args` | 22 个参数的列表 | 基类: `None` | `sglang serve` 额外参数 |
| `envs` | 10 个环境变量 | 基类: `None` | 测试专用环境变量 |
| `dataset_name` | `"random"` | 基类: `"random"` | bench-serving 数据集名 |
| `max_concurrency` | `100` | 基类: `None` | 最大并发请求数 |
| `num_prompts` | `400` | 基类: `None` | 总请求数 |
| `input_len` | `3500` | 基类: `None` | 每个请求输入 token 数 |
| `output_len` | `1500` | 基类: `None` | 每个请求输出 token 数 |
| `random_range_ratio` | `1` | 基类: `1` | 输入/输出长度随机波动比例 |
| `tpot` | `50` | 基类: `None` | TPOT 阈值（毫秒），用于断言 |
| `output_token_throughput` | `2031.71` | 基类: `None` | 输出吞吐阈值（tokens/s），用于断言 |

### 测试方法

```python
def test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms(self):
    self.run_throughput()
```

`unittest` 发现以 `test_` 开头的方法后自动执行。`self.run_throughput()` 是从基类 `TestAscendPerformanceTestCaseBase` 继承的方法（见下文第 803 行）。

---

## 第 100-101 行：程序入口

```python
if __name__ == "__main__":
    unittest.main()
```

当直接 `python` 执行此文件时，`unittest.main()` 扫描当前模块中所有 `TestCase` 子类，按顺序执行：
1. `setUpClass` → 启动服务器
2. `test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms` → `run_throughput()`
3. `tearDownClass` → 清理进程

---

# 第二部分：父类 TestAscendPerformanceTestCaseBase

**来源**: `python/sglang/test/ascend/e2e/test_npu_performance_utils.py:741`

## 第 741-772 行：类定义与属性默认值

```python
class TestAscendPerformanceTestCaseBase(CustomTestCase):
    model = None
    benchmark_tool = BENCHMARK_TOOL_DEFAULT        # "aisbench"
    backend = "sglang"
    dataset_name = "random"
    dataset_path = SHAREGPT_DATASET_TEST_FILE      # "/tmp/ShareGPT_V3_unfiltered_cleaned_split.json"
    aisbench_dataset_type = "gsm8k"
    aisbench_dataset_path = None
    other_args = None
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH    # 3600 秒
    envs = None
    max_attempts = 2                               # 失败重试 2 次
    request_rate = None
    max_concurrency = None
    num_prompts = None
    input_len = None
    output_len = None
    random_range_ratio = 1
    image_resolution = None
    image_count = None
    warmup_requests = None
    seed = None
    ttft = None                                    # 不检查 TTFT
    tpot = None                                    # 不检查 TPOT（子类覆盖）
    mean_e2e_latency = None                        # 不检查 E2E 延迟
    output_token_throughput = None                 # 不检查吞吐（子类覆盖）

    prefix_hit_rate = None
    aisbench_request_rate = None
    aisbench_repeat_rate = None
    dp = None
    generation_kwargs = None
```

所有属性在子类中可通过类级别变量覆盖。

---

## 第 774-793 行：setUpClass — 启动服务器

```python
@classmethod
def setUpClass(cls):
    cls.base_url = DEFAULT_URL_FOR_TEST
    env = os.environ.copy()
    for key, value in env.items():
        logger.info(f"ENV_VAR_SYS {key}:{value}")
    if cls.envs:
        for key, value in cls.envs.items():
            logger.info(f"ENV_VAR_CASE {key}:{value}")
            env[key] = value

    other_args = list(cls.other_args)

    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=cls.timeout,
        other_args=other_args,
        env=env,
    )
```

逐步拆解：

1. **`cls.base_url = DEFAULT_URL_FOR_TEST`**
   `DEFAULT_URL_FOR_TEST`（第 199 行）根据 `ASCEND_RT_VISIBLE_DEVICES` 或 `ASCEND_VISIBLE_DEVICES` 环境变量动态计算端口号：
   ```python
   # 例如 ASCEND_VISIBLE_DEVICES=0 → port=20066
   DEFAULT_SERVER_PORT_FOR_TEST = 20000 + int("0"[0]) * 100 = 20000
   DEFAULT_URL_FOR_TEST = f"http://127.0.0.1:{20000 + 66}"  # http://127.0.0.1:20066
   ```

2. **复制系统环境变量并打印**：逐项打印所有系统环境变量到日志，方便事后追溯。

3. **合并测试专用 envs**：将子类定义的 `cls.envs`（10 个 Ascend 专用变量）合并进环境变量。

4. **`popen_launch_server()`**（详见第三部分）：
   - 输入: `model`=模型路径, `base_url`=URL, `timeout`=3600s, `other_args`=22个参数, `env`=合并后的环境变量
   - 底层执行: `sglang serve --model-path <model> --tp-size 2 --nnodes 1 ...`
   - 返回: 一个 `subprocess.Popen` 对象
   - 阻塞等待直到 `/health` 端点返回 200

---

## 第 795-801 行：tearDownClass — 清理服务器进程

```python
@classmethod
def tearDownClass(cls):
    if hasattr(cls, "process") and cls.process:
        try:
            kill_process_tree(cls.process.pid)
        except Exception as e:
            logger.error(f"Error during tearDown: {e}")
```

- 在 `CustomTestCase` 的包装下，即使 `setUpClass` 失败也会执行此方法
- `kill_process_tree()` 杀死 `sglang serve` 进程及其所有子进程（model runner、tokenizer manager 等）

---

## 第 201-232 行：retry 装饰器

```python
def retry(max_attempts: int = None):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            last_exception = None
            attempts = max_attempts or getattr(self, "max_attempts", 2)
            for attempt in range(1, attempts + 1):
                try:
                    logger.info(f"Executing test attempt {attempt}/{attempts}")
                    return func(self, *args, **kwargs)
                except (AssertionError, Exception) as e:
                    last_exception = e
                    logger.info(f"Test failed on attempt {attempt}")
            raise last_exception
        return wrapper
    return decorator
```

- `run_throughput` 方法上标注了 `@retry()`（第 803 行）
- 默认重试 `max_attempts` 次（`TestAscendPerformanceTestCaseBase.max_attempts=2`）
- 任何 `AssertionError`（指标不达标）或 `Exception`（压测脚本崩溃）都触发重试
- 所有尝试都失败后才把最后的异常抛出

---

## 第 803-850 行：run_throughput — 核心压测逻辑

```python
@retry()
def run_throughput(self):
    parsed_url = urlparse(self.base_url)
    host = parsed_url.hostname
    port = parsed_url.port
    if self.benchmark_tool == AISBENCHMARK:
        metrics = run_aisbench(
            host=host, port=port, model_path=self.model,
            dataset_type=self.aisbench_dataset_type,
            dataset_path=self.aisbench_dataset_path,
            input_len=self.input_len, output_len=self.output_len,
            max_concurrency=self.max_concurrency, num_prompts=self.num_prompts,
            image_resolution=self.image_resolution,
            random_range_ratio=self.random_range_ratio,
            prefix_hit_rate=self.prefix_hit_rate,
            aisbench_request_rate=self.aisbench_request_rate,
            aisbench_repeat_rate=self.aisbench_repeat_rate,
            dp=self.dp, generation_kwargs=self.generation_kwargs,
        )
        assert_metrics(self, metrics)
    else:
        # bench-serving 路径 ...
```

**逻辑分支**：

```
benchmark_tool == "aisbench" ？
  ├─ YES → run_aisbench() → bash run_aisbench.sh → aisbench CLI → 解析指标
  └─ NO  → run_bench_serving() → python -m sglang.bench_serving → 解析指标

assert_metrics(self, metrics) ← 对指标做断言
```

对于此测试用例，`benchmark_tool = "aisbench"`，所以走 `run_aisbench()` 路径。

---

## 第 423-702 行：run_aisbench — 执行压测并提取指标

关键流程：

### 1. 数据集准备（第 442-497 行）

```python
if dataset_type == "sharegpt":
    # 从 ShareGPT 源文件生成指定长度的随机数据集
    generate_random_dataset(...)
elif dataset_type == "gsm8k" and not dataset_path:
    # 从 GSM8K 源文件生成指定长度的数据集
    generate_gsm8k_dataset(...)
elif dataset_type == "mm-custom-gen" and not dataset_path:
    # 生成多模态数据集（图片+文本）
    generate_mm_dataset(...)
```

本测试 `aisbench_dataset_type="gsm8k"`，无 `aisbench_dataset_path`，所以调用 `generate_gsm8k_dataset()`，从 GSM8K 测试集生成 400 条（`num_prompts`）输入约 3500 tokens（`input_len`）的数据。

### 2. 执行 aisbench（第 503-537 行）

```bash
/bin/bash /root/sglang/python/sglang/test/ascend/e2e/run_aisbench.sh \
    --mode perf \
    --ip 127.0.0.1 \
    --port 20066 \
    --model Qwen3.6-35B-A3B \
    --model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B \
    --dataset-type gsm8k \
    --dataset-path /tmp/datasets/test.jsonl \
    --input-len 3500 \
    --output-len 1500 \
    --batch-size 100 \
    --num-prompts 400 \
    --output-path /root/.cache/tests/output/perf/20260602/test_npu_...
```

`run_aisbench.sh` 内部会启动 aisbench 工具，以 `max_concurrency=100` 的并发向 `http://127.0.0.1:20066` 发送 400 个请求。

### 3. 指标提取（第 554-683 行）

从 aisbench stdout 中用**正则**提取 7 个指标：

| 指标 | 正则 | 变量名 |
|------|------|--------|
| TPOT | `TPOT\s+total\s+([\d.]+)\s+ms` | `mean_tpot` |
| Output Token Throughput | `Output\s+Token\s+Throughput\s+total\s+([\d.]+)\s+token\s*/?\s*s` | `total_tps` |
| TTFT | `TTFT\s+total\s+([\d.]+)\s+ms` | `mean_ttft` |
| E2E Latency | `E2EL\s+total\s+([\d.]+)\s+ms` | `mean_e2e_latency` |
| Concurrency | `Concurrency\s+total\s+([\d.]+)` | `concurrency` |
| Request Throughput | `Request\s+Throughput\s+total\s+([\d.]+)\s+req\s*/?\s*s` | `request_throughput` |
| Failed Requests | `Failed\s+Requests\s+total\s+(\d+)` | `failed_requests` |

---

## 第 704-738 行：assert_metrics — 指标断言校验

```python
def assert_metrics(self, metrics):
    if not metrics:
        raise Exception("No metrics obtained from benchmark")

    if self.tpot:
        if self.tpot < TPOT_THRESHOLD:           # 阈值 < 50ms
            self.assertLessEqual(float(metrics["mean_tpot"]), self.tpot + TPOT_TOLERANCE_LOW)
        else:                                     # 阈值 >= 50ms
            self.assertLessEqual(float(metrics["mean_tpot"]), self.tpot * TPOT_TOLERANCE_HIGH)

    if self.output_token_throughput:
        self.assertGreaterEqual(float(metrics["total_tps"]),
            self.output_token_throughput * OUTPUT_TOKEN_THROUGHPUT_TOLERANCE)

    if self.ttft:
        self.assertLessEqual(float(metrics["mean_ttft"]), self.ttft * TTFT_TOLERANCE)

    if self.mean_e2e_latency:
        self.assertLessEqual(float(metrics["mean_e2e_latency"]),
            self.mean_e2e_latency * E2E_TOLERANCE)
```

对于本测试，`tpot=50`，`output_token_throughput=2031.71`，`ttft=None`：

| 检查项 | 实际计算 | 含义 |
|--------|----------|------|
| TPOT | `actual_tpot ≤ 50 * 1.02 = 51ms` | TPOT 不超过 51ms |
| 吞吐 | `actual_tps ≥ 2031.71 * 0.98 = 1991.08` | 吞吐不低于 1991 tokens/s |
| TTFT | 跳过 | `ttft=None` 不检查 |
| E2E | 跳过 | `mean_e2e_latency=None` 不检查 |

---

# 第三部分：CustomTestCase — setUpClass 安全包装

**来源**: `python/sglang/test/test_utils.py:2158`

```python
class CustomTestCase(unittest.TestCase):

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        setup = cls.setUpClass
        if getattr(setup, "_safe_setup_wrapped", False):
            return

        orig_func = setup.__func__

        def safe_setUpClass(klass):
            try:
                orig_func(klass)
            except Exception:
                try:
                    klass.tearDownClass()
                except Exception:
                    pass
                raise

        safe_setUpClass._safe_setup_wrapped = True
        cls.setUpClass = classmethod(safe_setUpClass)
```

### 解决的问题

Python 原生 `unittest` 的行为是：**如果 `setUpClass` 抛出异常，`tearDownClass` 不会被调用**。这意味着如果服务器启动到一半失败（端口已被占用、模型加载到一半 OOM），`sglang serve` 子进程不会被清理，造成资源泄漏。

### 包装逻辑

`__init_subclass__` 在子类定义时自动触发（Python 3.6+ 特性），将原始的 `setUpClass` 替换为 `safe_setUpClass`：

```
safe_setUpClass(klass):
  try:
    orig_func(klass)          # 原始 setUpClass
  except Exception:
    try:
      klass.tearDownClass()   # ← 即使 setUpClass 失败，也调用 tearDownClass
    except Exception:
      pass                    # teardown 的异常不覆盖 setUp 的异常
    raise                     # 重新抛出 setUpClass 的原始异常
```

---

# 第四部分：popen_launch_server — 服务器进程启动

**来源**: `python/sglang/test/test_utils.py:861`

## 第 892-896 行：设备自动检测

```python
if device == "auto":
    device = auto_config_device()
    other_args += ["--device", str(device)]
```

不过在测试用例中已显式传入 `--device npu`（通过 `other_args`），`auto_config_device()` 的逻辑不会被触发（`popen_launch_server` 的 `device` 参数默认值是 `"auto"`，但调用方没有传 `device`）。

## 第 918-944 行：命令构建

```python
_, host, port = base_url.split(":")     # "http://127.0.0.1:20066" → host="127.0.0.1", port="20066"
host = host[2:]

command = [
    "sglang", "serve",
    "--model-path", model,
    *[str(x) for x in other_args],
    "--host", host,
    "--port", port,
]
```

实际拼出的命令：

```bash
sglang serve \
    --model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3.6-35B-A3B \
    --tp-size 2 --nnodes 1 --attention-backend ascend --device npu \
    --chunked-prefill-size -1 --max-prefill-tokens 35000 --disable-radix-cache \
    --trust-remote-code --enable-prefill-delayer --max-running-requests 100 \
    --max-mamba-cache-size 105 --mem-fraction-static 0.78 \
    --cuda-graph-bs 4 16 32 64 84 100 \
    --enable-multimodal --mm-attention-backend ascend_attn \
    --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --host 127.0.0.1 --port 20066
```

## 第 958-959 行：启动并等待

```python
process = _launch_server_process(command, env, return_stdout_stderr, model)
success, error_msg = _wait_for_server_health(process, base_url, api_key, timeout)
```

- `_launch_server_process`：`subprocess.Popen(command, env=env, ...)`
- `_wait_for_server_health`：轮询 `GET http://127.0.0.1:20066/health`，直到返回 200 或超时（`timeout=3600` 秒）

---

# 第五部分：完整运行时序

```
python -u test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py
│
├─ unittest.main() 扫描测试类
├─ __init_subclass__() 包装 setUpClass（安全 teardown）
│
├─ setUpClass()                                        ← TestAscendPerformanceTestCaseBase
│   ├─ 合并 env: 系统环境变量 + QWEN3_6_35B_A3B_3K5_1K5_ENVS
│   └─ popen_launch_server(model, base_url, timeout, other_args, env)
│       ├─ 拼装命令行: sglang serve --model-path <model> --tp-size 2 --device npu ...
│       ├─ subprocess.Popen(command, env=env)           ← _launch_server_process
│       ├─ 轮询 GET http://127.0.0.1:20066/health       ← _wait_for_server_health
│       └─ 返回 Popen 对象，存入 cls.process
│
├─ test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms()    ← 用户定义
│   └─ self.run_throughput()                            ← TestAscendPerformanceTestCaseBase
│       ├─ urlparse(self.base_url) → host, port
│       ├─ run_aisbench(host, port, model_path, ...)    ← benchmark_tool="aisbench"
│       │   ├─ generate_gsm8k_dataset() 生成 400 条 3500-token 输入
│       │   ├─ bash run_aisbench.sh ... --batch-size 100 --num-prompts 400
│       │   │     └─ aisbench CLI → POST /v1/completions × 400 (concurrency=100)
│       │   ├─ 正则提取: TPOT, TPS, TTFT, E2EL, Concurrency, Request Throughput
│       │   └─ 返回 metrics dict
│       └─ assert_metrics(self, metrics)
│           ├─ actual_tpot ≤ 50 * 1.02 = 51ms ？
│           └─ actual_tps ≥ 2031.71 * 0.98 = 1991.08 ？
│             任一失败 → @retry(max_attempts=2) 重试
│
└─ tearDownClass()                                      ← TestAscendPerformanceTestCaseBase
    └─ kill_process_tree(cls.process.pid)
```

---

# 关键数据流总结

```
类属性 (test case file)
  ├─ model, envs, other_args                 → popen_launch_server() → sglang serve 启动
  ├─ input_len, output_len, num_prompts,
  │   max_concurrency, random_range_ratio   → run_aisbench()         → aisbench 压测
  └─ tpot, output_token_throughput          → assert_metrics()       → 断言校验
```
