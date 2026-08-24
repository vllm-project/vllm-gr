# EngineCore Beam Batch 统一入口设计

> 关联 issue: [#37](https://github.com/zhanghanleo10/vllm-gr/issues/37)
> 父 RFC: [#35](https://github.com/zhanghanleo10/vllm-gr/issues/35)
> 契约依赖: PR #292（已合入，`vllm_gr/v1/beam/batch_contracts.py`）
> 下游: #39 Scheduler、#40 Lifecycle、#41 Output、#42 E2E

## 1. 任务是什么

#37 是 RFC #35 的 2/10 子任务，职责一句话：

> 实现 Online/Offline 共用的 EngineCore Beam Batch 入口：原子构造 B 个
> native child `Request` 和一个 `BeamBatch`，只注册一个 FCFS 队列项，并把
> 一个最终结果路由回调用方。

### 1.1 必须交付的能力

| 能力 | 说明 |
| --- | --- |
| submit | `submit_single(request)` / `submit_batch(BeamBatchRequest)`，验证不可变 `BeamBatchRequest` |
| identity | 为 Batch 与 B 个 child 建立稳定 identity；每个 child 保留独立 prompt history |
| 原子构造 | 先完整构造所有 Host 对象，再原子登记 Batch identity，并向 Scheduler 添加一个 `BeamBatch` |
| cancel | waiting / active cancel 转发 |
| final-only | 调用方只拿到最终结果，最多一个 terminal handle |
| registry | 最小 registry 与 terminal 去重；按 `batch_id` 路由结果 |
| fail-fast | Beam runtime 启用时拒绝普通 Request 与 BeamBatch 混合 |

### 1.2 注册线性化点（RFC #35 定义）

```text
BeamBatch 已完成验证
+ B 个 child 已完整构造
+ Batch identity 已登记
+ 一个 BeamBatch 已进入 native waiting
```

四个条件全部成功才允许返回 handle；不得出现「child-0 已入队、child-1
构造失败」的中间状态。

### 1.3 边界（#37 不做什么）

- 不分配 PromptKV / BeamKV / RunSlot；
- 不实现 Scheduler 原子准入（#39）、Worker Decode loop（#43）；
- 不实现 `BatchBeamCoordinator`、Prefill ready mask、Barrier、step/epoch、
  continuation（RFC #35 已废除旧 Session 模型）；
- 不读 Trie、不算 Beam score、不维护 sequence/parent；
- 不建立 EngineCore Constraint Resource registry（#38）。

## 2. 架构决策：这些类是「被调用」还是「被 patch」？

这是本设计的核心问题。直接回答：

**`beam_batch.py` 里的类是「被调用的组件」；「patch」的是把它们接进
vLLM 的接线层。两者是不同层的东西，缺一不可。**

### 2.1 仓库性质决定了答案

vllm-gr 不 fork、不修改上游 vLLM 源码。它通过 vLLM 的
`vllm.general_plugins` 入口在每个进程（main / EngineCore / worker）加载，
在运行时安装幂等 monkey-patch。这是仓库既有、且被
`docs/runtime_patch_inventory.md` / `docs/patch_inventory.yaml` 固化的架构
（GR-PATCH-000 ~ GR-PATCH-015）。

在这个架构里存在两类代码：

| 类型 | 例子 | 特点 |
| --- | --- | --- |
| **库组件（被调用）** | `BeamSessionRegistry`、`batch_contracts` 的 validate 函数、`EngineCoreBeamBatchIngress` | 不 import vLLM 的 patch 点，纯逻辑，可独立单测；由 patch 出来的 handler 调用 |
| **接线 patch（安装器）** | `engine_core_patch.py`、`core_client_patch.py`、`wire.py` | 幂等安装函数 + validator，登记进 `patch.py` 的进程 patch plan 与 patch inventory |

`EngineCoreBeamBatchIngress` 属于前者：它没有资格也没有必要自己 patch
EngineCore。它只负责「登记 + 路由」这块业务逻辑；把 `BEAM_BATCH_*` wire
消息解码、在 `_handle_client_request` 里分发、给 client 加方法，是接线层
（#37 自己的 patch 部分）的工作。

### 2.2 为什么不能反过来（类直接 patch EngineCore）

1. **可测性**：ingress 目前 6 个单测不启动 EngineCore、不依赖 ZMQ，靠注入
   `enqueue` / `cancel` 回调即可覆盖。如果它在模块里直接
   `from vllm.v1.engine.core import EngineCore` 并改写类方法，单测必须起
   完整 EngineCore，且无法覆盖「enqueue 失败 rollback」这类纯逻辑路径。
2. **幂等与进程差异**：patch 安装有 per-process plan（frontend 一套、
   EngineCore 一套），需要 `verify()` 判断是否已装、失败 fail-fast。把业务
   类和安装逻辑混在一起，validator 无从写起。
3. **避免循环依赖**：ingress 要引用 `batch_contracts`；接线层要引用 vLLM
   EngineCore；让 ingress 同时依赖两者会把它和上游私有接口耦合。

### 2.3 结论：调用方是谁

```text
Online/Offline Adapter
    │  submit_beam_batch(BeamBatchRequest)      ← client patch 提供的方法
    ▼
EngineCore（vLLM 类）
    │  _handle_client_request(BEAM_BATCH_SUBMIT, ...)   ← GR patch 的分支
    ▼
EngineCoreBeamBatchIngress            ← 本模块：被调用的组件
    │  validate → 构造 Host 对象 → registry 登记 → enqueue 回调
    ▼
Scheduler.waiting（native FCFS）       ← enqueue 回调由 #39 提供

Worker 返回 execution result
    │  EngineCore 收到 final result
    ▼
EngineCoreBeamBatchIngress.route_terminal_result(result)   ← 被调用
    │  batch_id → record → future.set_result
    ▼
调用方 handle.result() 得到唯一 final result
```

一句话：**ingress 是被 EngineCore 内部调用的单例组件；让它能被调到的
handler 和 wire 协议，由 vllm-gr 的 patch 安装。** 当前代码只写了组件，
没写接线，所以看起来像「几个类放在这里没人管」。

## 3. 类职责与调用关系

### 3.1 类/函数职责表

| 符号 | 角色 | 谁创建 | 谁调用 | 上游可见性 |
| --- | --- | --- | --- | --- |
| `GRLLM.beam_batch` / `AsyncLLM.beam_batch` | 用户侧统一入口（离线/在线 facade） | 前端 patch 添加方法 | 用户 / API Server | 用户可见 |
| `submit_beam_batch` / `abort_beam_batch`（client） | 前端发消息方法 | 前端 patch 添加 | `beam_batch` facade 内部调用 | 进程内可见 |
| `BeamBatchState` | 内部生命周期枚举 | — | ingress 内部 | 不可见 |
| `BeamBatchIngressError` | 基异常 | ingress 抛出 | patched handler / 调用方捕获 | 不可见 |
| `BeamBatchDuplicateBatchError` | 重复 batch_id / terminal 异常 | ingress 抛出 | 同上 | 不可见 |
| `BeamBatchMixedModeError` | 混合模式异常 | ingress 抛出 | 同上 | 不可见 |
| `BeamBatchRecord` | registry 内部记录（含 future） | ingress | ingress 内部 | 不可见 |
| `BeamBatchHandle` | 返回给调用方的 final-only 句柄 | ingress | Online/Offline 调用方 | 调用方可见 |
| `build_beam_batch_request` | 工具：从 native request 组 `BeamBatchRequest` | — | `submit_single` 与外部 adapter | 调用方可见 |
| `EngineCoreBeamBatchIngress` | 门面：submit / cancel / route / fail | EngineCore（patch 装配） | patched handler | EngineCore 进程可见 |
| `EnqueueCallback` / `CancelCallback` | 注入点类型 | 调用方（EngineCore） | ingress | 内部 seam |

### 3.2 用户侧统一入口（关键决策）

**用户侧始终只有一个函数；内部实现可以拆两个。**

```text
用户: grllm.beam_batch(prompts, beam_params, output_options)
        │  prompts 传 1 个 = B=1，传 B 个 = B>1，同一函数
        ▼
      build_beam_batch_request()   ← 统一构造 BeamBatchRequest
        ▼
      submit_beam_batch()          ← 统一发 BEAM_BATCH_SUBMIT
        ▼
      ingress.submit_batch()       ← 内部唯一实现（submit_single 只是薄包装）
```

用户不需要知道 `submit_single` / `submit_batch` 的区别，内部保留两个是
实现细节，避免 `isinstance` 分叉的 API 设计。

阻塞/非阻塞也用**同一个函数的一个参数**区分，不新增第二个入口：

```python
def beam_batch(self, prompts, beam_params, output_options, *, blocking=True):
    handle = self._submit(...)          # 内部永远先拿 handle
    return handle.result() if blocking else handle
```

- 默认 `blocking=True`：调用完直接拿 final result（与旧 `beam_search` 用法一致）；
- `blocking=False`：返回 handle，调用方自行 `handle.result()`。

### 3.3 依赖方向

```text
beam_batch.py
    ├── imports → vllm_gr/v1/beam/batch_contracts.py（#292 契约，只读）
    └── 被调用 ← vllm_gr/v1/engine/engine_core_patch.py（接线，patch 层）
                      └── 安装进 vllm.v1.engine.core.EngineCore / EngineCoreProc
```

`batch_contracts` 是唯一的下游依赖，且方向是「ingress 消费契约」，不是
「ingress 被契约消费」。接线层（engine_core_patch / wire / core_client_patch）
反过来消费 ingress。

### 3.4 对象生命周期（EngineCore 进程内）

```text
EngineCore.__init__（patched）
    └─ 创建 ingress 实例，挂到 self（如 self._beam_batch_ingress）
       └─ enqueue 回调 → Scheduler.waiting 添加（#39 注入）
       └─ cancel 回调   → Scheduler waiting 移除 / abort 转发（#39/#40 注入）

每来一个 BEAM_BATCH_SUBMIT 消息：
    handler 调 ingress.submit_batch(request)
    └─ 成功：返回 handle，消息侧只保留 batch_id
    └─ 失败：handler 把异常转成失败结果，不回灌 registry

Worker 终态返回后：
    EngineCore 调 ingress.route_terminal_result(result)
    └─ future.set_result → 调用方 handle.result() 解除阻塞
```

## 4. 接线方案（patch 部分，GR-PATCH-016 草案）

为了让 ingress 真正被调用，#37 还需要安装以下 patch。它们全部遵循仓库
既有惯例（参照 GR-PATCH-007/008/009）。

### 4.1 wire 枚举扩展

现有占用：vLLM `0x00`~`0x05`，vllm-gr `ADD_BATCH=0x06`、
`BEAM_REQUEST_STEP_UPDATE=0x07`。

新增（`vllm_gr/v1/engine/wire.py`）：

```text
BEAM_BATCH_SUBMIT = b"\x08"
BEAM_BATCH_ABORT  = b"\x09"
```

复用 `register_engine_core_request_types()` 的 `_add_enum_member` 机制。

### 4.2 EngineCore 侧（engine_core_patch.py）

- `EngineCore.__init__` / `EngineCoreProc.__init__`（已有 wrapper）追加：
  创建 `EngineCoreBeamBatchIngress`，注入 enqueue / cancel 回调；
- `_handle_client_request` 增加分支：`BEAM_BATCH_SUBMIT` → `submit_beam_batch`，
  `BEAM_BATCH_ABORT` → `abort_beam_batch`；
- `process_input_sockets` 的 wire 解码自然覆盖新字节（msgpack 帧已按枚举解码）；
- final result 路由：EngineCore 收到 Worker execution result 后调用
  `ingress.route_terminal_result(...)`（结果 schema 与 #41 对齐）。

### 4.3 前端调用接口（新）

#### 离线（GRLLM）

```python
grllm = GRLLM(model=model, attention_config={"backend": "CUSTOM"})

# 一个函数，1 个或 B 个 prompt 都走这里；默认阻塞拿结果
result = grllm.beam_batch(
    prompts=[p0, p1, ..., pB],
    beam_params=BeamBatchParams(beam_width=4, max_tokens=32, ...),
    output_options=BeamOutputOptions(num_return_sequences=4, ...),
)
# result: BeamBatchResult（final-only）

# 或者 blocking=False 拿 handle，不阻塞
handle = grllm.beam_batch(prompts, beam_params, output_options, blocking=False)
result = handle.result()
```

#### 在线（OpenAI 风格）

```python
client = OpenAI(base_url="http://localhost:8000/v1")
    resp = client.beam_search.create(
    model="qwen3",
    prompts=[p0, p1, ..., pB],   # 一个 HTTP 请求 = 一个 BeamBatch
    beam_width=4,
    max_tokens=32,
    num_return_sequences=4,
)   # 返回一个 final 结果
```

在线**不需要 `blocking` 参数**：HTTP 天然一个请求一个响应，服务端内部等
handle 完成再回包，调用方无感。

#### 内部流程

```text
beam_batch(prompts, ...)
  → prepare_request × B           # 插件方法：收集 B 个 EngineCoreRequest（不发）
  → build_beam_batch_request()    # 打包成 BeamBatchRequest
  → submit_beam_batch()           # client 方法 → BEAM_BATCH_SUBMIT
  → EngineCore ingress → Scheduler（一个 FCFS 队列项）
  → Worker 跑完 → route_terminal_result → handle.result()
```

### 4.4 client 侧（core_client_patch.py）

参照 `add_requests_async` / `beam_request_step_update_async`：

- `AsyncMPClient` / `SyncMPClient` / `InprocClient` 增加
  `submit_beam_batch` / `abort_beam_batch`；
- `AsyncLLM` 增加 `beam_batch` 高层入口（在线 facade）；
- `GRLLM` 增加 `beam_batch`（离线 facade，复用 `prepare_request`）。

### 4.5 patch plan 与 inventory 登记

- `patch.py` 的 `_engine_core_beam_patches()` 增加一项
  `("beam_batch_ingress", "vllm_gr.v1.engine.beam_batch_ingress_patch", ...)`
  + 对应 validator；
- `docs/runtime_patch_inventory.md` 与 `docs/patch_inventory.yaml` 增加
  GR-PATCH-016，字段：owner / installer / activation / validator_or_test /
  process_roles / target / risk / removal_condition。

### 4.6 为什么是一个 patch 条目，而不是每个类一个

patch 的粒度是「能力」（EngineCore 能处理 Beam Batch 消息），不是「类」。
`BeamBatchState`、`BeamBatchRecord`、`BeamBatchHandle` 是能力内部的零件，
只有 `EngineCoreBeamBatchIngress` + 接线函数构成一个可验证的能力单元。

## 5. 文件地图与端到端调用顺序

这一节回答「改了哪些文件、每个文件干什么、整个调用顺序是什么」。

### 5.1 文件地图

| 文件 | 角色 | 新增的关键内容 |
| --- | --- | --- |
| `vllm_gr/v1/engine/beam_batch.py` | **核心逻辑（大脑）**：被调用的组件，不自己 patch | `BeamBatch` 宿主对象、`default_child_builder`、`build_beam_batch_request`、`EngineCoreBeamBatchIngress`（submit / cancel / route / fail / registry / rollback / handle） |
| `vllm_gr/v1/engine/wire.py` | **协议编号**：注册 EngineCoreRequestType 新枚举 | `BEAM_BATCH_SUBMIT=0x08`、`BEAM_BATCH_ABORT=0x09`（沿用既有 `_add_enum_member` 机制） |
| `vllm_gr/v1/engine/codec.py` | **消息翻译官**：按 request_type 选 decoder 解包 | `EngineCoreDecoders.beam_batch_submit`、`decode_engine_core_payload` 增加 BEAM_BATCH_SUBMIT 分支 |
| `vllm_gr/v1/engine/core.py` | **EngineCore 侧管家**：装配 + 分发 | `_ensure_beam_batch_ingress`（创建 ingress 实例）、`_handle_beam_batch_submit` / `_handle_beam_batch_abort`、`process_input_sockets` 新分支、`_enqueue_beam_batch` / `_abort_beam_batch`（#39/#40 seam） |
| `vllm_gr/v1/engine/engine_core_patch.py` | **安装工**：把管家装进 vLLM | patched `__init__` 里调 `_ensure_beam_batch_ingress`；`apply_engine_core_request_patches` 绑定两个 handler |
| `vllm_gr/v1/engine/core_client.py` | **前端消息发送器**：client 方法本体 | `submit_beam_batch_async` / `abort_beam_batch_async`（AsyncMP）、`submit_beam_batch` / `abort_beam_batch`（SyncMP）、`inproc_submit_beam_batch` / `inproc_abort_beam_batch`（Inproc） |
| `vllm_gr/v1/engine/core_client_patch.py` | **前端接线工**：把方法赋给 client 类 | `apply_engine_client_patches`（三个 client）、`apply_async_llm_patches`（`AsyncLLM.beam_batch = beam_batch_fn`） |
| `vllm_gr/v1/engine/async_llm.py` | **在线用户入口**：`AsyncLLM.beam_batch` 本体 | `beam_batch_fn`：tokenize B 个 prompt → `build_beam_batch_request` → `submit_beam_batch_async` → 返回 batch_id |
| `vllm_gr/entrypoints/gr.py` | **离线用户入口**：`GRLLM.beam_batch` 本体 | `beam_batch`：tokenize B 个 prompt → `build_beam_batch_request` → `engine_core.submit_beam_batch` → `handle.result()` |
| `tests/test_beam_batch_ingress.py` | 单测 | 9 个用例：submit / cancel / route / rollback / 重复 ID / 重复 terminal / mixed / child 物化 / child 失败 / wire 值 |
| `docs/patch_inventory.yaml` + `docs/runtime_patch_inventory.md` | patch 清单 | GR-PATCH-016 登记（owner / installer / validator / 进程角色） |

### 5.2 端到端调用顺序

#### 场景 A：离线默认（InprocClient，同进程）

```text
GRLLM.beam_batch(prompts, beam_params, output_options)      gr.py
  → input_processor.process_inputs() × B                    创建 B 个 EngineCoreRequest
  → build_beam_batch_request()                              打包成 BeamBatchRequest
  → engine_core.submit_beam_batch(request)                  InprocClient（core_client.py）
  → ingress.submit_batch(request)                           同进程直调（beam_batch.py）
      → validate → child_builder（native Request × B）→ registry 登记 → enqueue 回调
  → handle.result()                                         阻塞等 final 结果
```

#### 场景 B：在线 / 离线多进程（AsyncMPClient / SyncMPClient，跨进程）

```text
AsyncLLM.beam_batch(...)                                     async_llm.py beam_batch_fn
  → input_processor.process_inputs() × B
  → build_beam_batch_request()
  → engine_core.submit_beam_batch_async(request)             AsyncMPClient（core_client.py）
      → self._send_input(BEAM_BATCH_SUBMIT, request)         ZMQ 发送序列化消息
                                                             ↓ 跨进程
EngineCoreProc.process_input_sockets()                       core.py
  → decode_engine_core_payload()                             codec.py：按 request_type 解码
  → 分支 BEAM_BATCH_SUBMIT
  → self._handle_beam_batch_submit(batch_request)            core.py
      → ingress.submit_batch(request)                        核心逻辑（beam_batch.py）
```

#### 场景 C：取消

```text
client.abort_beam_batch(batch_id)
  ├─ InprocClient：直调 ingress.cancel(batch_id)
  └─ Async/SyncMPClient：_send_input(BEAM_BATCH_ABORT, batch_id)
       → process_input_sockets 分支 → _handle_beam_batch_abort → ingress.cancel(...)
```

### 5.3 patch 安装顺序（谁先被装）

```text
1. vLLM 加载 general plugin → vllm_gr.plugin.register → re_apply_patches
2. 前端进程：patch_batch_and_fork()
     → apply_engine_client_patches()   绑定三个 client 的 submit/abort
     → apply_async_llm_patches()       绑定 AsyncLLM.beam_batch
3. EngineCore 进程：run_engine_core() wrapper
     → apply_engine_core_request_patches()
         → register_engine_core_request_types()   注册 0x08/0x09
         → 包装 EngineCore.__init__ / EngineCoreProc.__init__
         → 绑定 _handle_beam_batch_submit / _handle_beam_batch_abort
         → 替换 EngineCoreProc.process_input_sockets
4. EngineCore 构造（patched __init__）
     → _ensure_beam_batch_ingress(self)   创建 ingress 实例，挂到 self._beam_batch_ingress
```

### 5.4 一句话记忆

- `beam_batch.py` = 大脑（核心逻辑）
- `wire.py` = 协议编号
- `codec.py` = 消息翻译官
- `core.py` = EngineCore 侧管家（装配 + 分发）
- `engine_core_patch.py` = 安装工（装进 vLLM）
- `core_client.py` = 前端消息发送器
- `core_client_patch.py` = 前端接线工
- `async_llm.py` / `gr.py` = 用户入口（在线 / 离线 facade）

## 6. 与相邻子任务的边界

| 子任务 | 与 #37 的接口 | #37 负责 | 对方负责 |
| --- | --- | --- | --- |
| #39 Scheduler | enqueue / cancel 回调 | 定义回调签名、保证「只 enqueue 一次」 | 把 `BeamBatch` 作为一个 FCFS 队列项做 all-or-none 准入 |
| #40 Lifecycle | cancel 转发与 terminal 去重 | 提供 cancel 转发、拒绝重复 terminal | 整组 cancel/failure/poison 与 exactly-once 清理 |
| #41 Output | `BeamBatchResult` 路由 | 复用 #292 schema 并路由 | final-only 结果 schema、指标、detokenize |
| #42 E2E | 同一入口契约 | 保证 Online/Offline 走同一 ingress | 端到端验收 |

## 7. 当前实现与交付物差距

本地 `vllm_gr/v1/engine/beam_batch.py` + `tests/test_beam_batch_ingress.py`
已实现（9 个单测通过）：

- [x] `submit_single` / `submit_batch` / `cancel` / `route_terminal_result`；
- [x] native child builder + `BeamBatch` 宿主对象；
- [x] 原子构造 + enqueue 失败 rollback；
- [x] 最小 registry（batch_id 单一索引）；
- [x] 重复 batch_id、重复 terminal fail-fast；
- [x] mixed-mode fail-fast；
- [x] final-only handle。

对照 issue 交付物，仍有差距：

| 差距 | 现状 | 归属 |
| --- | --- | --- |
| **enqueue 真实 Scheduler 注入** | EngineCore 已装配 ingress，`enqueue` 回调目前是 #39 占位（不把 B 个 child 推入公共 waiting） | #39 |
| **Worker final result 通道** | `route_terminal_result` 已就绪，但真实 Worker 终态还未调用它 | #40/#41/#43 |
| **在线跨进程 final 结果回传** | `AsyncLLM.beam_batch` 已能提交并返回 batch_id；跨进程 completion handle 尚待输出 adapter | #41/#42 |
| **离线 multiprocess final 结果回传** | `GRLLM.beam_batch` 默认 in-process 可直接 `handle.result()`；mp 后端暂明确报错 | #41/#42 |

## 8. 验收标准映射

| 验收标准（issue #37） | 由谁保证 | 测试/验证 |
| --- | --- | --- |
| 不会出现部分 child 已 enqueue、其余 child 构造失败 | child 构造 + 原子登记（待实现）+ rollback | 新增部分构造失败用例 |
| 一个业务 Batch 只向 native waiting 添加一个 `BeamBatch` | enqueue 回调只调用一次 | `test_submit_batch_enqueues_once_and_routes_terminal` |
| B 个 child 不作为公共 waiting 的独立节点 | #39 队列项实现 | #39 用例 |
| EngineCore 不接收或产生中间 Prefill/Decode 结果 | final-only 路由 | handle 只暴露 result/done/cancelled |
| 每个 Batch 最多完成一个 terminal handle | registry terminal 去重 | `test_duplicate_terminal_result_rejected` |
| Online 与 Offline 使用同一入口契约 | 同一 ingress 实例被两端调用 | E2E（#42） |

## 9. 非目标

- 不分配 PromptKV / BeamKV / RunSlot；
- 入口层不实现 Scheduler 原子准入；
- 不实现 Worker Decode loop；
- 不读取 Trie，不计算 Beam score；
- 不维护 sequence / parent 状态；
- 不实现组 batch 策略（组 batch 属于业务侧/Scheduler）。

## 10. 设计决策记录

1. **ingress 是被调用组件，接线是 patch**：遵守仓库既有 patch 架构，避免
   业务类与上游私有接口耦合。
2. **enqueue / cancel 用构造器注入，不硬编码 Scheduler**：让 ingress 可独立
   单测，也让 #39 的实现可以后到。
3. **child 构造必须在 ingress 侧完成**：issue 明确要求「先完整构造所有
   Host 对象，再原子登记」，这是 #37 的交付物，不能推给 #39。
4. **结果按 `batch_id` 路由**：batch_id 由前端/ingress 内部生成，全局唯一，
   不接收外部 ID。
5. **用户侧一个函数，内部两个实现**：`beam_batch` 统一处理 B=1 与 B>1；
   `submit_single` / `submit_batch` 是内部实现细节，不做 `isinstance` 分叉。

## 11. 待办（按依赖顺序）

- [x] native child builder 与 `BeamBatch` 宿主对象，补原子构造 + rollback；
- [x] 前端 facade：`GRLLM.beam_batch` / `AsyncLLM.beam_batch` 统一用户入口；
- [x] 接线层（wire 0x08/0x09、EngineCore handler、client 方法）；
- [x] GR-PATCH-016（patch plan + inventory）；
- [x] 测试：部分 child 构造失败、重复 ID、重复 terminal、mixed fail-fast、wire 值；
- [ ] 与 #39 对齐 enqueue 回调；与 #41 对齐 final result schema；
- [ ] 在线/离线 multiprocess 的跨进程 completion handle（#41/#42）。
