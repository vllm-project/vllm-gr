# tools/ 与 tests/development/ 迁移方案

> 日期: 2026-05-29
> 状态: 待审批
> 基于: vllm-gr 仓库 `merge` 分支 (GitHub: parallel-star8/vllm-gr)

## 一、背景

vllm-gr 仓库原有完善的开发工具（GPU 资源管理、pre-commit 钩子、统一测试入口等），这些工具在合并 vllm-GR 功能时未被带入 `merge` 分支。当前 `merge` 分支中：

- 没有 `tools/` 目录
- 没有 `tests/development/` 目录
- 没有 `.pre-commit-config.yaml`
- 运行测试需要手动指定 GPU（如 `CUDA_VISIBLE_DEVICES=1 pytest ...`），没有自动资源分配

**仓库信息**：

```
当前分支: merge
origin:  https://github.com/parallel-star8/vllm-gr.git
upstream: https://github.com/JiusiServe/vllm-gr.git
```

由于仓库在 GitHub 上，所有 GitHub 相关工具（act、actionlint、export_changes）均可正常使用。

## 二、迁移范围

### 2.1 `tools/` 目录（6 个文件）

| 文件 | 功能 | 依赖 |
|------|------|------|
| `resource.sh` | GPU 资源管理函数库（get_gpu, lock_gpu, unlock_gpu 等） | nvidia-smi, lock_gpu.py |
| `lock_gpu.py` | 通过 EGL/GL 分配显存锁定 GPU（Python ctypes） | libEGL, libGL |
| `export_changes.sh` | 基于 upstream/main 导出 diff 到新分支 | git, upstream remote |
| `run_test.sh` | 找空闲 GPU → 锁定 → 运行 pre-commit | resource.sh, pre-commit |
| `run_act.sh` | 用 act 本地运行 GitHub Actions | act CLI, Docker |
| `pre_commit/` (8 个文件) | 见下方详表 | — |

### 2.2 `tools/pre_commit/` 目录（8 个文件）

| 文件 | 功能 |
|------|------|
| `check_pickle_imports.py` | 禁止 pickle/cloudpickle 导入（安全检查） |
| `check_spdx_header.py` | 检查/补全 SPDX 许可证头 |
| `enforce_regex_import.py` | 强制使用 regex 替代 re |
| `mypy.py` | mypy 类型检查（按目录分组，路径已针对 `vllm_gr/`） |
| `png-lint.sh` | 检查 excalidraw PNG 元数据 |
| `run_tests.sh` | 运行 pytest 测试套件（`python3 -m pytest --run-slow --durations=10 -s tests`） |
| `shellcheck.sh` | shell 脚本 lint |
| `validate_config.py` | 校验 config dataclass 有默认值和 docstring |

### 2.3 `tests/development/` 目录（1 个文件）

| 文件 | 功能 |
|------|------|
| `test_resource.sh` | resource.sh 集成测试（验证 GPU 查找/锁定/解锁） |

### 2.4 `.pre-commit-config.yaml`（1 个文件）

完整的 pre-commit 配置，包含 20+ 个钩子：

| 类别 | 钩子 |
|------|------|
| 通用 | check-yaml, debug-statements, end-of-file-fixer, mixed-line-ending, trailing-whitespace |
| Python | isort, ruff-check, ruff-format |
| 拼写 | typos |
| C/C++ | clang-format |
| Markdown | markdownlint |
| GitHub Actions | actionlint |
| 自定义 | mypy, shellcheck, png-lint, check-spdx-header, check-filenames, enforce-import-regex, validate-config, run-tests, check-pickle-imports, signoff-commit |
| 提示 | suggestion |

## 三、迁移方式

**全部直接复制，无需适配。**

由于仓库在 GitHub 上，之前的顾虑全部消除：

| 之前的顾虑 | 当前情况 |
|------------|----------|
| `run_act.sh` 依赖 GitHub Actions | 仓库在 GitHub 上，直接可用 |
| `actionlint` 钩子检查 `.github/workflows/` | 仓库有 GitHub Actions，直接可用 |
| `.pre-commit-config.yaml` 需要适配 | 直接可用，无需修改 |
| `export_changes.sh` 依赖 upstream remote | 已配置 upstream remote，直接可用 |
| `validate_config.py` 路径正则 | 不影响功能，先复制后续按需调整 |
| `run_tests.sh` 跑全部测试太慢 | pre-commit 中已配置 `stages: [manual]`，不自动触发 |

## 四、文件迁移清单

| # | 源文件（merge 分支历史） | 目标路径 | 方式 |
|---|---|---|---|
| 1 | `tools/resource.sh` | `tools/resource.sh` | 直接复制 |
| 2 | `tools/lock_gpu.py` | `tools/lock_gpu.py` | 直接复制 |
| 3 | `tools/export_changes.sh` | `tools/export_changes.sh` | 直接复制 |
| 4 | `tools/run_test.sh` | `tools/run_test.sh` | 直接复制 |
| 5 | `tools/run_act.sh` | `tools/run_act.sh` | 直接复制 |
| 6 | `tools/pre_commit/check_pickle_imports.py` | `tools/pre_commit/check_pickle_imports.py` | 直接复制 |
| 7 | `tools/pre_commit/check_spdx_header.py` | `tools/pre_commit/check_spdx_header.py` | 直接复制 |
| 8 | `tools/pre_commit/enforce_regex_import.py` | `tools/pre_commit/enforce_regex_import.py` | 直接复制 |
| 9 | `tools/pre_commit/mypy.py` | `tools/pre_commit/mypy.py` | 直接复制 |
| 10 | `tools/pre_commit/png-lint.sh` | `tools/pre_commit/png-lint.sh` | 直接复制 |
| 11 | `tools/pre_commit/run_tests.sh` | `tools/pre_commit/run_tests.sh` | 直接复制 |
| 12 | `tools/pre_commit/shellcheck.sh` | `tools/pre_commit/shellcheck.sh` | 直接复制 |
| 13 | `tools/pre_commit/validate_config.py` | `tools/pre_commit/validate_config.py` | 直接复制 |
| 14 | `tests/development/test_resource.sh` | `tests/development/test_resource.sh` | 直接复制 |
| 15 | `.pre-commit-config.yaml` | `.pre-commit-config.yaml` | 直接复制 |

**共 15 个文件，全部直接复制，0 个需要适配。**

## 五、迁移后目录结构

```
vllm-gr/ (merge 分支)
├── .pre-commit-config.yaml          # [新增] pre-commit 钩子配置
├── tools/                           # [新增目录]
│   ├── resource.sh                  # [新增] GPU 资源管理
│   ├── lock_gpu.py                  # [新增] GPU 锁定
│   ├── export_changes.sh            # [新增] 变更导出
│   ├── run_test.sh                  # [新增] 开发测试入口
│   ├── run_act.sh                   # [新增] 本地 GitHub Actions 运行
│   └── pre_commit/                  # [新增目录]
│       ├── check_pickle_imports.py  # [新增]
│       ├── check_spdx_header.py     # [新增]
│       ├── enforce_regex_import.py  # [新增]
│       ├── mypy.py                  # [新增]
│       ├── png-lint.sh              # [新增]
│       ├── run_tests.sh             # [新增]
│       ├── shellcheck.sh            # [新增]
│       └── validate_config.py       # [新增]
├── tests/
│   └── development/                 # [新增目录]
│       └── test_resource.sh         # [新增] resource.sh 集成测试
├── docs/                            # 已有
├── vllm_gr/                         # 已有（含合并后的完整代码）
├── benchmarks/                      # 已有
├── examples/                        # 已有
└── ...                              # 其余已有文件不变
```

## 六、迁移前后对比

| 场景 | 迁移前 | 迁移后 |
|------|--------|--------|
| 选择 GPU | 手动 `nvidia-smi` 查看，手动指定 `CUDA_VISIBLE_DEVICES` | `get_gpu` 自动查找空闲 GPU |
| 防止 GPU 冲突 | 无机制，多进程可能抢同一张卡 | `lock_gpu` / `unlock_gpu` 自动锁定释放 |
| 跑全部测试 | 手动 `pytest tests/` + 手动指定 GPU | `tools/run_test.sh` 或 `tools/pre_commit/run_tests.sh` 一条命令 |
| 代码质量检查 | 无自动化 | `git commit` 时自动运行 pre-commit 钩子（lint、格式、类型等） |
| 本地验证 CI | 需推送到远程触发 | `tools/run_act.sh` 本地运行 GitHub Actions |
| 导出变更 | 手动操作 | `tools/export_changes.sh` 基于上游分支自动导出 |
| 资源管理验证 | 无 | `tests/development/test_resource.sh` 集成测试 |

## 七、风险与注意事项

| 风险 | 等级 | 说明 |
|------|------|------|
| `lock_gpu.py` 依赖 libEGL/libGL | 低 | GPU 服务器通常已安装；缺少时 lock_gpu 会 graceful fail |
| `validate_config.py` 路径正则 `(vllm\|tests)` 不匹配 `vllm_gr` | 低 | 该正则继承自上游 vLLM，实际不影响功能，后续按需调整 |
| pre-commit 钩子过重影响提交速度 | 低 | `run-tests` 配置为 `stages: [manual]`，不会在每次提交时触发；其余钩子均为秒级 |
| `mypy.py` 文件列表需与 vllm-gr 目录对应 | 低 | 已配置针对 `vllm_gr/` 路径，后续可根据合并后目录结构微调 |
| `shellcheck.sh` 自动下载 shellcheck 二进制 | 低 | 仅在缺失时下载，有 sha256 校验 |

## 八、迁移后验证方案

迁移完成后按以下顺序逐步验证：

### 8.1 pre-commit 框架验证

安装并运行所有钩子，确认 `.pre-commit-config.yaml` 中引用的所有 `tools/pre_commit/*` 脚本正常工作：

```bash
# 安装 pre-commit（如未安装）
pip install pre-commit

# 安装钩子到本地仓库
pre-commit install

# 跑一遍所有钩子
pre-commit run --all-files
```

**预期**：ruff、isort、mypy、shellcheck、SPDX、pickle 检查等钩子依次执行，输出各钩子的通过/失败状态。

### 8.2 GPU 资源管理验证

运行 resource.sh 集成测试，验证 `get_gpu`、`lock_gpu`、`unlock_gpu`、`get_gpus`、`lock_gpus` 等函数：

```bash
bash tests/development/test_resource.sh
```

**预期输出**：
```
Testing get_gpu... PASSED (Got GPU ID: X)
Testing lock_gpu on GPU X... PASSED (Locked with PID: XXXX)
Testing unlock_gpu... PASSED (GPU unlocked)
Testing get_gpus (requesting 2 GPUs)... PASSED (Got GPU IDs: X,Y)
Testing lock_gpus on GPUs 'X,Y'... PASSED (Locked with PIDs: XXXX,YYYY)
Testing unlock_gpus... PASSED (GPUs unlocked)
All integration tests passed successfully!
```

如无空闲 GPU 或 GPU 数量不足，相应测试会输出 `SKIP` 而非失败。

### 8.3 统一测试入口验证

验证 `pytest --run-slow` 能正常跑通 `tests/` 下所有测试：

```bash
bash tools/pre_commit/run_tests.sh
```

**预期**：`python3 -m pytest --run-slow --durations=10 -s tests` 正常执行，输出测试结果汇总。

### 8.4 开发测试入口验证

验证完整的"自动找 GPU → 锁定 → 跑 pre-commit"流程：

```bash
bash tools/run_test.sh
```

**预期**：
1. 先运行 `tests/development/test_resource.sh` 验证资源管理
2. 自动查找空闲 GPU 并锁定
3. 在该 GPU 上运行 `pre-commit run --all-files`

### 8.5 提交触发验证

做一个小改动，提交时观察 pre-commit 钩子是否自动运行：

```bash
# 做一个无害的小改动
echo "# test" >> vllm_gr/__init__.py
git add vllm_gr/__init__.py
git commit -m "test: verify pre-commit hooks"
# 观察输出是否显示各钩子执行结果
# 验证完成后回退
git reset HEAD~1
git checkout vllm_gr/__init__.py
```

**预期**：提交时自动触发 pre-commit 钩子（lint、格式化、SPDX 检查等），输出各钩子的执行状态。

### 8.6 验证清单

| 步骤 | 命令 | 验证内容 | 优先级 |
|------|------|----------|--------|
| 1 | `pre-commit run --all-files` | 所有钩子脚本是否可用 | 必须 |
| 2 | `bash tests/development/test_resource.sh` | GPU 资源管理是否正常 | 必须 |
| 3 | `bash tools/pre_commit/run_tests.sh` | pytest 统一入口是否正常 | 必须 |
| 4 | `bash tools/run_test.sh` | 完整开发测试流程 | 可选 |
| 5 | `git commit` 小改动 | 提交时钩子是否自动触发 | 可选 |

## 九、统计

| 分类 | 数量 |
|------|------|
| 直接复制 | 15 |
| 适配迁移 | 0 |
| 暂不迁移 | 0 |
| **总计** | **15** |
