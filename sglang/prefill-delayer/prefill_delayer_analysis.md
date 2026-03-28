# Prefill Delayer 参数分析与测试用例

## 一、使用场景分析

### 1.1 使用场景
Prefill Delayer（预填充延迟器）是一组用于优化数据并行（DP）场景下预填充操作的参数。主要使用场景包括：

- **大模型数据并行推理**：当使用多卡数据并行推理大模型时
- **高吞吐量服务**：在需要高吞吐量的在线推理服务中
- **资源受限环境**：在 NPU 显存或带宽受限的环境中
- **混合负载场景**：当预填充和解码操作混合进行时

### 1.2 作用说明
该组参数的作用包括：

1. **延迟预填充**：通过延迟预填充操作，避免与解码操作竞争资源
2. **资源均衡**：在预填充和解码之间平衡资源使用
3. **性能优化**：减少空闲时间，提高整体吞吐量
4. **监控支持**：提供详细的性能指标和直方图数据

### 1.3 使用约束
- 必须启用数据并行（`--enable-dp-attention`）
- 必须禁用聚合模式（`--disaggregation-mode null`）
- 必须禁用重叠调度（`--disable-overlap-schedule`）
- 主要用于多卡数据并行场景

### 1.4 带来的增益
- **提升吞吐量**：通过减少资源竞争提高整体吞吐量
- **降低延迟**：优化预填充时机，减少请求延迟
- **更好的资源利用**：平衡预填充和解码的资源使用
- **可观测性**：提供详细的性能监控指标

## 二、实现原理分析

### 2.1 数据流

```
服务启动
    ↓
解析 Prefill Delayer 参数
    ↓
初始化 PrefillDelayer
    ↓
调度器接收请求
    ↓
negotiate_should_allow_prefill()
    ↓
判断是否允许预填充
    ↓
允许预填充：执行预填充
不允许预填充：延迟或等待
    ↓
记录指标到 MetricsCollector
```

### 2.2 内部处理逻辑

#### 2.2.1 初始化阶段
1. **参数解析**：从命令行读取所有 Prefill Delayer 相关参数
2. **延迟器创建**：创建 `PrefillDelayer` 实例
3. **直方图初始化**：根据参数配置指标直方图
4. **验证约束**：检查 DP 相关配置是否正确

#### 2.2.2 协商逻辑（Negotiation）
核心协商逻辑在 `_negotiate_should_allow_prefill()` 中实现：

**状态判断**：
- `all_prefillable`: 所有 DP rank 都可以预填充
- `none_prefillable`: 没有任何 rank 可以预填充
- `mixed_prefillable`: 部分 rank 可以预填充

**决策输出**：
- `output_allow`: 是否允许预填充（True/False）
- `output_reason`: 决策原因
  - `wait_success`: 成功等待后允许
  - `no_wait`: 无需等待直接允许
  - `delay`: 延迟预填充
  - `wait_timeout`: 等待超时
  - `token_watermark`: 由于低水位标记强制允许

#### 2.2.3 延迟控制
- **max_delay_passes**: 最大延迟轮数，超过此值将超时
- **delayed_count**: 当前已延迟的轮数
- **skip_first_delayer**: 跳过第一次延迟（用于避免初始抖动）

#### 2.2.4 水位标记机制
- **token_usage_low_watermark**: 当 token 使用率低于此值时强制允许预填充
- 避免在低负载时过度延迟

### 2.3 配置参数

#### 2.3.1 --enable-prefill-delayer
- **参数名**：`--enable-prefill-delayer`
- **参数类型**：布尔值
- **有效值**：`True`（启用），`False`（禁用）
- **默认值**：`False`
- **作用**：启用预填充延迟器

#### 2.3.2 --prefill-delayer-max-delay-passes
- **参数名**：`--prefill-delayer-max-delay-passes`
- **参数类型**：整数
- **有效值**：正整数
- **默认值**：`30`
- **作用**：最大允许的延迟轮数

#### 2.3.3 --prefill-delayer-token-usage-low-watermark
- **参数名**：`--prefill-delayer-token-usage-low-watermark`
- **参数类型**：浮点数
- **有效值**：0.0 到 1.0 之间的浮点数
- **默认值**：`None`
- **作用**：token 使用率低水位标记，低于此值时强制允许预填充

#### 2.3.4 --prefill-delayer-forward-passes-buckets
- **参数名**：`--prefill-delayer-forward-passes-buckets`
- **参数类型**：浮点数列表
- **有效值**：浮点数列表（如 `[5, 20, 50, 100, 200]`）
- **默认值**：`None`（自动添加 0 和 max_delay_passes-1）
- **作用**：自定义前向传递的直方图桶边界

#### 2.3.5 --prefill-delayer-wait-seconds-buckets
- **参数名**：`--prefill-delayer-wait-seconds-buckets`
- **参数类型**：浮点数列表
- **有效值**：浮点数列表（如 `[1, 2, 5, 10, 20, 50, 100, 200, 500]`）
- **默认值**：`None`（自动添加 0）
- **作用**：自定义等待秒数的直方图桶边界

### 2.4 参数生效点和影响

#### 2.4.1 生效点
1. **服务启动时**：`server_args.py:3974-4002` 添加命令行参数
2. **调度器初始化时**：`scheduler.py:878-891` 创建 `PrefillDelayer` 实例
3. **请求调度时**：在 `negotiate_should_allow_prefill()` 中应用延迟逻辑
4. **指标收集时**：`metrics_collector.py:724-847` 记录性能指标

#### 2.4.2 影响
- **预填充决策**：直接影响是否允许预填充操作
- **延迟轮数**：控制预填充的延迟次数
- **性能指标**：影响收集的直方图数据
- **资源利用**：影响 NPU 资源的分配和利用

## 三、测试用例设计

### 3.1 测试用例 1：启用 Prefill Delayer 基础测试

**测试目的**：验证启用 Prefill Delayer 后，服务正常启动且预填充延迟逻辑生效

**测试步骤**：
1. 启动 SGLang 服务，启用数据并行和 Prefill Delayer
2. 发送多个并发请求
3. 观察服务是否正常响应
4. 验证性能指标中包含 Prefill Delayer 相关指标

**测试环境**：多张 NPU 卡（至少 2 张）

**预期结果**：
- 服务成功启动
- 请求正常响应
- 指标中包含 `sglang:prefill_delayer_outcomes_total`
- 指标中包含 `sglang:prefill_delayer_wait_forward_passes`

### 3.2 测试用例 2：自定义 max_delay_passes 测试

**测试目的**：验证自定义最大延迟轮数参数生效

**测试步骤**：
1. 启动服务，设置 `--prefill-delayer-max-delay-passes 50`
2. 发送请求并观察延迟行为
3. 验证延迟逻辑符合配置

**测试环境**：多张 NPU 卡（至少 2 张）

**预期结果**：
- 服务成功启动
- 延迟逻辑使用自定义的最大值
- 性能指标正常收集

### 3.3 测试用例 3：token_usage_low_watermark 测试

**测试目的**：验证低水位标记机制正常工作

**测试步骤**：
1. 启动服务，设置 `--prefill-delayer-token-usage-low-watermark 0.7`
2. 发送请求
3. 验证在低 token 使用率时预填充决策正确

**测试环境**：多张 NPU 卡（至少 2 张）

**预期结果**：
- 服务成功启动
- 低水位标记生效
- 在低负载时预填充决策正确

### 3.4 测试用例 4：自定义 forward_passes_buckets 测试

**测试目的**：验证自定义前向传递直方图桶生效

**测试步骤**：
1. 启动服务，设置自定义 `--prefill-delayer-forward-passes-buckets`
2. 发送请求
3. 验证指标中使用的桶边界与配置一致

**测试环境**：多张 NPU 卡（至少 2 张）

**预期结果**：
- 服务成功启动
- 直方图使用自定义桶边界
- 指标数据正确

### 3.5 测试用例 5：自定义 wait_seconds_buckets 测试

**测试目的**：验证自定义等待秒数直方图桶生效

**测试步骤**：
1. 启动服务，设置自定义 `--prefill-delayer-wait-seconds-buckets`
2. 发送请求
3. 验证指标中使用的桶边界与配置一致

**测试环境**：多张 NPU 卡（至少 2 张）

**预期结果**：
- 服务成功启动
- 直方图使用自定义桶边界
- 指标数据正确

### 3.6 测试用例 6：禁用 Prefill Delayer 对比测试

**测试目的**：验证禁用 Prefill Delayer 后，延迟逻辑不生效

**测试步骤**：
1. 启动服务，不启用 Prefill Delayer
2. 发送请求
3. 观察响应时间和行为

**测试环境**：多张 NPU 卡（至少 2 张）

**预期结果**：
- 服务成功启动
- 请求正常响应
- 没有 Prefill Delayer 延迟行为

### 3.7 测试用例 7：多 DP rank 协商测试

**测试目的**：验证多 DP rank 之间的预填充协商正确

**测试步骤**：
1. 启动 2 卡数据并行服务
2. 发送多个并发请求
3. 验证不同 rank 的预填充决策协调一致

**测试环境**：2 张 NPU 卡

**预期结果**：
- 服务成功启动
- 多 rank 协商正常
- 预填充决策一致

### 3.8 测试用例 8：参数组合测试

**测试目的**：验证多个 Prefill Delayer 参数组合正常工作

**测试步骤**：
1. 启动服务，设置多个 Prefill Delayer 参数
2. 发送请求
3. 验证所有参数都生效

**测试环境**：多张 NPU 卡（至少 2 张）

**预期结果**：
- 服务成功启动
- 所有配置参数生效
- 性能指标正常收集

## 四、测试说明

### 4.1 测试覆盖范围
- ✅ 启用 Prefill Delayer 基础功能
- ✅ 自定义 max_delay_passes
- ✅ token_usage_low_watermark 机制
- ✅ 自定义 forward_passes_buckets
- ✅ 自定义 wait_seconds_buckets
- ✅ 禁用 Prefill Delayer 对比
- ✅ 多 DP rank 协商
- ✅ 参数组合测试

### 4.2 测试注意事项
1. 测试使用多卡数据并行配置
2. 所有测试在 NPU 环境下执行
3. 测试使用 ascend attention backend
4. 需要至少 2 张 NPU 卡才能测试数据并行功能
5. 测试端口使用默认的 30000

### 4.3 测试依赖
- SGLang 框架
- 多张 NPU 卡
- 支持数据并行的模型
- OpenAI Python SDK（用于发送请求）

### 4.4 测试模型
推荐使用以下模型进行测试：
- Llama-3-8B-Instruct
- Qwen2.5-7B-Instruct
- 其他支持数据并行的模型
