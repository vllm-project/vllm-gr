# L20 在线性能对比结果

## 1. 结论

在同一台 L20 上，vLLM 0.22.1rc1 + vllm-gr 相对 vLLM 0.14.1
及其配套 vllm-gr 的精度结果一致，但当前在线 Beam Search 性能仍有回退：

- TTFT P99 从 427.75 ms 增加到 462.65 ms，回退 8.16%；
- Beam Search 总耗时从 313.35 ms 增加到 332.68 ms，回退 6.17%；
- 平均 Prefill 耗时回退 4.49%，平均 Decode 耗时回退 6.84%；
- 之前 10 请求测试中出现的 P99 766.95 ms 未在 100 请求测试中复现，属于小样本下不稳定的尾延迟结果；
- 当前结果不满足“性能不允许回退”的验收要求，状态为**待定位，不通过性能门禁**。

L20 结果只用于判断软件升级是否引入相对回退。目标 CI 使用 A100，最终结论仍需在
A100 上用旧、新两套软件栈做同卡 A/B 测试。

## 2. 测试环境与约束

| 项目 | 配置 |
| --- | --- |
| 硬件 | 同一台 NVIDIA L20 |
| 旧版本服务 | vLLM 0.14.1 + 对应的 vllm-gr，ModelRunner V1，端口 7001 |
| 新版本服务 | vLLM 0.22.1rc1 + 升级后的 vllm-gr，ModelRunner V1，端口 7002 |
| 客户端 | 两组测试均从 0.22.1rc1 客户端容器及代码仓执行 |
| 模型 | `/home/z00469465/OneRec/OneRec-1.7B` |
| 数据集 | `/home/z00469465/OneRec/OpenOneRec-RecIF/benchmark_data` |
| Catalog | `/home/z00469465/OneRec/generated_data/video_catalog.json` |
| Task | `video` |
| Beam width | 128 |
| 请求数 | 100 个正式请求，另有 10 个 benchmark 内部 warmup 请求 |
| 流量 | 0.5 RPS，Poisson，未限制最大并发 |
| 实际峰值并发 | 两组均为 4 |
| 输入 | `--disable-shuffle`，两组均为 238,753 input tokens |

两组服务不能同时占用同一张 GPU。每次只启动一个服务，通过不同端口避免访问残留进程，
并使用仓库根目录的 [do_test.sh](../do_test.sh) 从同一个 0.22.1rc1 客户端环境发压。
两套服务的完整启动、版本自检和停止命令见
[升级验证指南的 Online 基准章节](upgrade_validation_guide.md#63-online-基准)。

示例：

```bash
# 测试旧服务
PORT=7001 ./do_test.sh 0.14.1

# 停止旧服务并确认端口释放后，再测试新服务
PORT=7002 ./do_test.sh 0.22.1rc1
```

## 3. 在线性能结果

正数 delta 表示 0.22.1rc1 的数值高于 0.14.1。对于时延指标，正数表示性能回退。

| 指标 | 0.14.1 | 0.22.1rc1 | 变化量 | 相对变化 |
| --- | ---: | ---: | ---: | ---: |
| Successful requests | 100 | 100 | 0 | 0.00% |
| Failed requests | 0 | 0 | 0 | - |
| Benchmark duration | 200.47 s | 200.46 s | -0.01 s | 约 0.00% |
| Request throughput | 0.50 req/s | 0.50 req/s | 0 | 0.00% |
| Total input tokens | 238,753 | 238,753 | 0 | 0.00% |
| Total generated tokens | 62,761 | 64,000 | +1,239 | +1.97% |
| Output token throughput | 313.06 tok/s | 319.26 tok/s | +6.20 tok/s | +1.98% |
| Total token throughput | 1504.00 tok/s | 1510.25 tok/s | +6.25 tok/s | +0.42% |
| Mean TTFT | 291.24 ms | 297.94 ms | +6.70 ms | **+2.30%** |
| Median/P50 TTFT | 289.84 ms | 293.24 ms | +3.40 ms | **+1.17%** |
| P90 TTFT | 341.42 ms | 356.44 ms | +15.02 ms | **+4.40%** |
| P95 TTFT | 390.96 ms | 403.93 ms | +12.97 ms | **+3.32%** |
| P99 TTFT | 427.75 ms | 462.65 ms | +34.90 ms | **+8.16%** |
| Mean TPOT | 0.01 ms | 0.01 ms | 约 0 | 约 0.00% |
| P99 TPOT | 0.01 ms | 0.01 ms | 约 0 | 约 0.00% |
| Mean ITL | 0.02 ms | 0.02 ms | 约 0 | 约 0.00% |
| P99 ITL | 0.04 ms | 0.04 ms | 约 0 | 约 0.00% |
| Avg Prefill Time | 88.77 ms | 92.76 ms | +3.99 ms | **+4.49%** |
| Avg Decode Time | 224.54 ms | 239.89 ms | +15.35 ms | **+6.84%** |
| Avg Sort Time | 0.038 ms | 0.038 ms | 约 0 | 约 0.00% |
| Total Beam Time | 313.35 ms | 332.68 ms | +19.33 ms | **+6.17%** |

`Beam Search Requests` 为 110，因为服务端自定义 Beam 指标同时统计了 10 个内部
warmup 请求和 100 个正式请求；客户端输出的 TTFT 分位数只统计正式请求。

## 4. 离线结果旁证

此前同一台 L20 的 500 请求离线 Beam Search 测试也表现出小幅回退：

| 指标 | 0.14.1 | 0.22.1rc1 | 相对变化 |
| --- | ---: | ---: | ---: |
| Elapsed time | 229.74 s | 234.57 s | +2.10% |
| Requests/s | 2.18 | 2.13 | -2.29% |
| Tokens/s | 1392.91 | 1364.17 | -2.06% |

离线和在线结果都指向新版本存在回退，但幅度不同。在线低负载结果显示 Beam
单请求路径约回退 6%，离线满负载吞吐约回退 2%，需要进一步拆分 CPU、调度和 GPU
执行时间后才能归因。

## 5. 正确解读当前指标

### 5.1 0.50 req/s 不是最大吞吐

测试的到达速率固定为 0.50 req/s，两组服务都能跟上流量，因此两者的 request
throughput 都是 0.50 req/s。这个值只能说明测试过程中没有积压，不能证明两组的容量相同。
最大吞吐应以离线测试或逐级提高在线 request rate 的饱和测试为准。

### 5.2 Output token throughput 暂不能作为新版本更快的证据

新版本报告了 64,000 generated tokens，旧版本为 62,761，相差 1.97%。在测试持续时间
由固定请求到达速率决定时，output token throughput 的增加主要由 token 计数增加造成，
不能直接解释为引擎计算更快。需确认 EOS 行为、响应 usage 统计以及每个 choice 的实际长度。

### 5.3 TPOT 和 ITL 的区分度有限

当前请求使用 `n=128` 的自定义 Beam Search 流式响应，服务端可能按聚合 chunk 返回候选。
因此 0.01 ms 量级的 TPOT 和 ITL 主要反映接口返回粒度，不宜用于判断底层逐 token
解码 kernel 是否发生回退。本轮应优先关注服务端 Prefill、Decode 和 Beam 总耗时。

### 5.4 10 请求 P99 不作为门禁依据

最初 10 请求结果中，新版本 P99 TTFT 为 766.95 ms，旧版本为 327.31 ms。P99 在只有
10 个样本时实际上接近单个最慢请求，容易受到 JIT、首次 shape、GPU 抖动等因素影响。
100 请求且完成 warmup 后，新版本 P99 为 462.65 ms。后续正式比较至少使用 100 个请求，
每个配置独立重复 3 次，并报告中位数和波动区间。

## 6. 精度状态

关闭 ModelRunner V2、强制使用 ModelRunner V1 后，0.14.1 与 0.22.1rc1 的精度指标一致，
当前没有发现升级导致的精度回退。ModelRunner V2 曾产生全零精度结果，因此不纳入本轮
性能对比和 CI 放行范围；服务启动日志必须能证明实际使用的是 V1。

精度一致只说明输出质量满足当前样本检查，不能抵消性能门禁失败。每个性能候选修复在最终
合入前仍需重新跑完整精度测试。

## 7. 原始结果位置

```text
/home/z00469465/OneRec/results/online_compare/
├── 0.14.1/
│   ├── performance.log
│   ├── performance-detail.json
│   └── accuracy/
├── 0.14.1_server.log
├── 0.22.1rc1/
│   ├── performance.log
│   ├── performance-detail.json
│   └── accuracy/
└── 0.22.1rc1_server.log
```

后续定位步骤、A/B 变量和判定方法见
[性能回退定位与待测试计划](performance_regression_investigation.md)。
