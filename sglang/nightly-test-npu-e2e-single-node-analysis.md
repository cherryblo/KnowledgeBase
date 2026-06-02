# nightly-test-npu-e2e-single-node.yml 逐行解析

## 文件概述

- **路径**: `.github/workflows/nightly-test-npu-e2e-single-node.yml`
- **类型**: 可复用 workflow（`workflow_call`），被 `full-test-npu.yml` 的 `nightly-poc-single-node-tests` 等 job 调用
- **作用**: 在 Ascend NPU 单节点容器内执行一个 e2e 测试用例（性能或精度），包括环境准备、服务启动、压测执行、指标收集、日志归档

---

## 第 1 行：workflow 名称

```yaml
name: 'e2e nightly test single-node (Ascend NPU)'
```

GitHub Actions UI 中的显示名称。

---

## 第 3-37 行：触发方式与输入参数

```yaml
on:
  workflow_call:
    inputs:
```

不是由 push/PR 事件触发，而是 `workflow_call`——仅作为可复用 workflow 被其他 workflow 调用。

### 输入参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `runner` | string | `linux-aarch64-a3-16` | 运行节点标签，对应不同 NPU 卡数的机器（a3-2/a3-4/a3-8/a3-16） |
| `test_type` | string | `perf` | `perf`（性能测试）或 `accuracy`（精度测试） |
| `test_config_name` | string | **必填** | 测试配置名称，如 `test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms` |
| `test_case` | string | **必填** | 测试用例 `.py` 文件路径，如 `test/registered/ascend/performance/xxx.py` |
| `image` | string | `swr.cn-southwest-2.myhuaweicloud.com/...` | 华为云 SWR 上的 CANN Docker 镜像 |
| `install_sglang_from_source` | boolean | `false` | `true` 时从源码安装 sglang，否则使用镜像内置版本 |
| `transformers_version` | string | `""` | 指定 transformers 版本号，为空则使用镜像内置版本 |

---

## 第 39-41 行：并发控制

```yaml
concurrency:
  group: ascend-nightly-e2e-singlenode-${{ github.workflow_ref }}-${{ github.ref }}-${{ inputs.test_config_name }}
  cancel-in-progress: true
```

- 相同 `group` 的运行互斥，防止同一测试用例被重复触发
- `cancel-in-progress: true`：新触发自动取消正在运行的旧任务（避免资源争抢）

---

## 第 43-48 行：Job 定义 / Runner / 容器

```yaml
jobs:
  e2e:
    name: ${{ inputs.test_config_name }}
    runs-on: ${{ inputs.runner }}
    container:
      image: ${{ inputs.image }}
```

- 只有一个 job：`e2e`
- job 显示名称用 `test_config_name`，在 Actions UI 中直接显示具体用例名
- `runs-on` 由调用方传入，决定使用几卡节点
- `container:` 以容器模式运行，镜像为调用方传入的 CANN 镜像

---

## 第 49-51 行：Step 1 — Checkout 代码

```yaml
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
```

用 `actions/checkout@v4` 拉取代码到容器内 `$GITHUB_WORKSPACE`。

---

## 第 53-55 行：Step 2 — 检查 NPU 信息

```yaml
      - name: Check npu info
        run: |
          npu-smi info
```

`npu-smi info` 等同于 `nvidia-smi`，打印 NPU 设备信息（卡数、显存、驱动、CANN 版本），用于日志中确认硬件状态正常。

---

## 第 57-66 行：Step 3 — 环境变量声明

```yaml
      - name: Run test
        timeout-minutes: 120
        env:
          SGLANG_USE_MODELSCOPE: true
          HF_ENDPOINT: https://hf-mirror.com
          SGLANG_IS_IN_CI: true
          TRANSFORMERS_VERBOSITY: "error"
          GDN_ATTN_BACKEND_TRITON: 1
        shell: bash
```

| 变量 | 值 | 作用 |
|------|-----|------|
| `SGLANG_USE_MODELSCOPE` | `true` | 优先从 ModelScope（国内源）下载模型，避免 HF 网络不通 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像站 |
| `SGLANG_IS_IN_CI` | `true` | 标记 CI 环境，影响离线缓存验证等逻辑 |
| `TRANSFORMERS_VERBOSITY` | `error` | 减少日志噪音，仅输出错误级别日志 |
| `GDN_ATTN_BACKEND_TRITON` | `1` | 启用 Triton 后端的 GDN Attention（Ascend NPU 专用） |

`timeout-minutes: 120` 限制该步最长运行 2 小时，防止测试卡死无限等待。

---

## 第 67-69 行：记录源码路径并建软链接

```bash
sglang_source_path=$(pwd)
echo "Source code path: ${sglang_source_path}"
ln -sf ${sglang_source_path} /root/sglang
```

后续脚本（如 `run_aisbench.sh`）硬编码了 `/root/sglang` 路径，软链接使其指向实际 checkout 位置。

---

## 第 71-75 行：校验测试文件存在

```bash
test_case=${{ inputs.test_case }}
if [ ! -f "${sglang_source_path}/${test_case}" ]; then
  echo "The testcase does not exit: ${sglang_source_path}/${test_case}"
  exit 1
fi
```

防御性检查：如果传入的 `test_case` 路径不存在则立即失败退出，避免到 `python -u` 时才报错。

---

## 第 77-85 行：创建输出目录和指标文件路径

```bash
current_date=$(date +%Y%m%d)
test_data_output_path=/root/.cache/tests/output/${{ inputs.test_type }}/${current_date}
mkdir -p ${test_data_output_path}
tc_name=${test_case##*/}
tc_name=${tc_name%.*}
export METRICS_DATA_FILE=${test_data_output_path}/${tc_name}
mkdir -p ${METRICS_DATA_FILE}
```

- 按日期 + 测试类型创建目录：`/root/.cache/tests/output/perf/20260602/`
- `${test_case##*/}` 去掉路径前缀，`${tc_name%.*}` 去掉 `.py` 后缀
- `METRICS_DATA_FILE` 环境变量被测试脚本内部读取，用于写入指标文件

示例：`test/registered/ascend/performance/test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py` → `tc_name=test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms`

---

## 第 87-90 行：复制测试数据集

```bash
cp ~/.cache/modelscope/hub/datasets/otavia/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
curl -o /tmp/test.jsonl -L https://gh-proxy.test.osinfra.cn/https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
```

- 从宿主机预缓存的 ModelScope 目录复制 ShareGPT 数据集
- 通过内部代理下载 GSM8K 测试集
- 两个文件后续被 `run_aisbench` 或 `run_bench_serving` 用于构造压测请求

---

## 第 92-111 行：Transformers 版本管理

```bash
export TRANSFORMERS_VERSION_FOR_SGLANG="${{ inputs.transformers_version }}"
PYTHON_FOR_SGLANG="python"
PIP_FOR_SGLANG="pip"
if [ -n "${TRANSFORMERS_VERSION_FOR_SGLANG}" ];then
  TRANSFORMERS_PKG_PATH_SOURCE=/root/.cache/.cache/transformers/${TRANSFORMERS_VERSION_FOR_SGLANG}
  if [ ! -d "${TRANSFORMERS_PKG_PATH_SOURCE}" ]; then
    pip install transformers=="${TRANSFORMERS_VERSION_FOR_SGLANG}" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  else
    TRANSFORMERS_PKG_PATH_TARGET=/tmp/transformers/${TRANSFORMERS_VERSION_FOR_SGLANG}
    mkdir -p "${TRANSFORMERS_PKG_PATH_TARGET}"
    cp "${TRANSFORMERS_PKG_PATH_SOURCE}/*" "${TRANSFORMERS_PKG_PATH_TARGET}/"
    pip install --no-index --find-links="${TRANSFORMERS_PKG_PATH_TARGET}" transformers=="${TRANSFORMERS_VERSION_FOR_SGLANG}"
  fi
fi
```

- 如果调用方指定了 `transformers_version`：
  - 先查本地预缓存路径 `/root/.cache/.cache/transformers/<version>`
  - 缓存命中 → 离线安装（用 `--no-index --find-links`），快速且不依赖网络
  - 缓存未命中 → 从清华 PyPI 镜像在线安装
- 未指定则跳过，使用镜像自带的 transformers

---

## 第 113-117 行：打印系统调优参数

```bash
echo "scaling_governor performance num: $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | grep performance | wc -l)"
echo "swappiness: $(cat /proc/sys/vm/swappiness)"
echo "numa_balancing: $(cat /proc/sys/kernel/numa_balancing)"
echo "sched_migration_cost_ns: $(cat /proc/sys/kernel/sched_migration_cost_ns)"
```

| 参数 | 含义 | 推理场景推荐 |
|------|------|-------------|
| `scaling_governor` | CPU 频率调节策略 | `performance`，锁定最高频率 |
| `swappiness` | 内存交换倾向（0-100） | 低值，避免 swap 干扰延迟 |
| `numa_balancing` | NUMA 自动页面迁移 | `0`（关闭），避免内存迁移导致抖动 |
| `sched_migration_cost_ns` | 调度器迁移成本阈值 | 打印记录，用于排查调度延迟 |

---

## 第 119-121 行：重试与 CPU 亲和性

```bash
export SGLANG_TEST_MAX_RETRY=0
export SGLANG_SET_CPU_AFFINITY=1
```

- `SGLANG_TEST_MAX_RETRY=0`：禁用框架层自动重试，失败即失败（在 workflow 层面通过 `max_attempts` 控制重试）
- `SGLANG_SET_CPU_AFFINITY=1`：启用 CPU 亲和性绑定，将进程绑定到特定 CPU 核，减少跨 NUMA 访问延迟

---

## 第 123-136 行：镜像模式 vs 源码模式

```bash
install_sglang_from_source=${{ inputs.install_sglang_from_source }}
if [ "$install_sglang_from_source" = "true" ] || [ "$install_sglang_from_source" = "True" ];then
  echo "Install sglang from source"
  commit_id=${{ github.sha }}
  echo "commit id: ${commit_id}" > ${test_data_output_path}/commit_id
  export PYTHONPATH=${sglang_source_path}/python:$PYTHONPATH
else
  echo "Use sglang from image: ${{ inputs.image }}"
  sglang_pkg_path=/sgl-workspace/sglang/python
  ascend_test_util_path=${sglang_pkg_path}/sglang/test/ascend
  mkdir -p ${ascend_test_util_path}
  mv ${ascend_test_util_path} ${ascend_test_util_path}_bak
  cp -r ${sglang_source_path}/python/sglang/test/ascend ${ascend_test_util_path}
fi
```

两种运行模式：

**源码模式**（`install_sglang_from_source: true`）：
- 将 checkout 代码的 `python/` 加入 `PYTHONPATH`
- 记录 commit id 到输出目录，方便追溯

**镜像模式**（`false`，默认）：
- 使用镜像内预装的 sglang 运行
- 但**测试工具代码总是从源码取最新版本**：
  1. 备份镜像内原有 `sglang/test/ascend` 目录
  2. 复制源码中的 `python/sglang/test/ascend` 覆盖进去
- 这样既复用镜像的稳定 sglang 版本，又确保测试框架是最新的

---

## 第 138-139 行：加载 CANN 环境

```bash
source /usr/local/Ascend/cann/set_env.sh || true
source /usr/local/Ascend/nnal/atb/set_env.sh || true
```

- `set_env.sh`：设置 CANN 框架所需的环境变量（`ASCEND_HOME`、`LD_LIBRARY_PATH`、`PATH` 等）
- `atb/set_env.sh`：加载 Ascend Transformer Boost 加速库
- `|| true` 确保即使 source 失败（如文件不存在），该步不会导致整个 step 报错退出

---

## 第 141-145 行：创建运行时日志目录

```bash
log_path="/root/.cache/tests/logs/log/${current_date}/${tc_name}/${HOSTNAME}"
rm -rf ${log_path}
mkdir -p ${log_path}
```

按 `日期/测试用例名/主机名` 三级目录组织日志，`rm -rf` 确保每次运行是干净的日志环境。

---

## 第 147-149 行：核心 — 执行测试用例

```bash
echo "Running test case ${test_case}"
${PYTHON_FOR_SGLANG} -u ${test_case}
echo "Finished test case ${test_case}"
```

**整条链路的最核心行**。实际执行的命令等价于：

```bash
python -u test/registered/ascend/performance/test_npu_qwen3_6_35b_a3b_1p_in3k5_out1k5_50ms.py
```

- `-u` 关闭 Python stdout/stderr 缓冲，日志实时输出而不被截断
- 该 Python 文件是一个 `unittest` 用例：
  1. `setUpClass` → `popen_launch_server()` → `sglang serve ...` 启动服务
  2. `test_xxx()` → `self.run_throughput()` → `run_aisbench()` / `run_bench_serving()` 压测
  3. `assert_metrics()` 校验 TPOT / TTFT / 吞吐量 是否达标
  4. `tearDownClass` → `kill_process_tree()` 清理服务进程

> 详细调用链路见 [full-test-npu-call-chain-analysis.md](./full-test-npu-call-chain-analysis.md)

---

## 第 151-159 行：备份 plog 调试日志

```bash
plog_path="/root/ascend/log/debug/plog"
if [ -d "$plog_path" ];then
  echo "Plog files found. Begin to backup them."
  target_plog_path="/root/.cache/tests/logs/plog/${tc_name}/${HOSTNAME}"
  rm -rf ${target_plog_path}
  mkdir -p ${target_plog_path}
  cp ${plog_path}/* ${target_plog_path}
fi
```

- `plog`（Precision Log）是 CANN 框架的算子精度和执行日志，记录每个 NPU 算子的数值精度信息
- 如果该目录存在（CANN 可能在特定条件才生成），将其复制到归档目录
- 用于事后分析算子精度问题（如 FP16 溢出、量化误差等）

---

## 完整执行时序

```
Step 1: Checkout code
  → git clone 到容器内 $GITHUB_WORKSPACE

Step 2: Check npu info
  → npu-smi info 确认硬件可用

Step 3: Run test（单步顺序执行）
  ├─ 获取源码路径，建立 /root/sglang 软链接
  ├─ 校验 test_case 文件是否存在（不存在 → exit 1）
  ├─ 创建输出目录 /root/.cache/tests/output/{perf|accuracy}/{日期}/
  ├─ 设置 METRICS_DATA_FILE 环境变量
  ├─ 复制 ShareGPT + 下载 GSM8K 数据集到 /tmp
  ├─ （可选）安装指定版本的 transformers
  ├─ 打印系统内核调优参数（CPU governor、swappiness、NUMA balancing）
  ├─ 设置 SGLANG_TEST_MAX_RETRY=0、SGLANG_SET_CPU_AFFINITY=1
  ├─ 镜像模式：复制最新 test/ascend 代码到镜像 sglang 包中
  ├─ source /usr/local/Ascend/cann/set_env.sh          ← 加载 CANN 环境
  ├─ source /usr/local/Ascend/nnal/atb/set_env.sh      ← 加载 ATB 加速库
  ├─ 创建日志目录 /root/.cache/tests/logs/log/{日期}/{用例名}/{主机名}/
  ├─ ★ python -u <test_case>.py                          ← 执行测试
  └─ 备份 plog 到 /root/.cache/tests/logs/plog/
```

---

## 被哪条 workflow 调用

`nightly-test-npu-e2e-single-node.yml` 被 `full-test-npu.yml` 中的以下 job 调用：

| 调用方 Job | 用途 |
|-----------|------|
| `nightly-poc-single-node-tests` | 单节点单机 POC 性能/精度测试（~50 个用例，max-parallel=3） |

调用方式：

```yaml
uses: ./.github/workflows/nightly-test-npu-e2e-single-node.yml
with:
  runner: ${{ matrix.test_config.runner }}
  test_type: ${{ matrix.test_config.test_type }}
  test_config_name: ${{ matrix.test_config.name }}
  test_case: ${{ matrix.test_config.test_case }}
  image: ${{ needs.set-image-config.outputs.image_a3 }}
  install_sglang_from_source: false
  transformers_version: ''
```
