# Benchmark

这个目录提供 `vllm-GR` 的 accuracy benchmark，覆盖 inference-only 的精度评测。

当前 runtime benchmark 结构已经尽量对齐 `vllm` 原生 benchmark 的数据集组织方式：

- `common/` 放模型无关的 benchmark 公共能力
- `models/<model_family>/` 放模型族专属的 dataset adapter 与 runner
- 当前 HSTU benchmark 入口位于 `vllm_gr/benchmarks/models/hstu/`
- `common/gr_accuracy_utils.py` 负责公共后处理，而不是把后处理逻辑散落在模型目录中

同时保留一处 GR 特例：

- `scripts/benchmark_data/*` 继续作为独立的数据准备脚本存在，不并入 runtime benchmark 抽象层

## 范围

- 支持数据集：
    - `movielens-1m`
    - `movielens-20m`
    - `kuairand-1k`
- 支持指标：
    - `NE`
    - `Accuracy`
    - `GAUC`

benchmark 直接复用 `GRLLM` 和 `GRPrompt`。

当前目录建议如下：

```text
vllm_gr/benchmarks/
  common/
    base.py
    gr_accuracy_utils.py
  models/
    hstu/
      datasets.py
      gr_accuracy.py
```

## 指标说明

当前 benchmark 的指标口径对齐 upstream `generative_recommenders/dlrm_v3/inference/accuracy.py` 与 `MetricsLogger`。`models/hstu/gr_accuracy.py` 的汇总 JSON 使用小写字段名 `accuracy`、`ne`、`gauc`，其中：

- `accuracy`
    - 二分类准确率。
    - 先把预测分数按 `0.5` 阈值转成 `0/1`，再与标签比较。
    - 当前实现按 `weights_by_task` 做加权平均，越大越好。
- `ne`
    - Normalized Entropy，归一化熵。
    - 本质是加权二元交叉熵，再除以标签分布自身的熵。
    - 越小越好。`1.0` 接近“只学到总体正样本率”的基线，`< 1.0` 通常表示优于该基线。
- `gauc`
    - 这里直接沿用 upstream 的 `GAUC` 语义，对应 group AUC，而不是简单的全局 ROC-AUC。
    - `accuracy` 关心“按 `0.5` 阈值判对没有”，`gauc` 关心“正样本整体有没有排在负样本前面”，因此它是排序型指标，不依赖 `0.5` 阈值做最终分类。
    - 计算口径是先在单条请求的 candidate 集合内计算组内 AUC，再在请求级做加权聚合。
    - 指标计算阶段按标签值划分正负样本：`label >= 0.5` 视为正样本，`label < 0.5` 视为负样本。由于当前 benchmark 标签实际是 `0.0` / `1.0`，可直接理解为 `1.0` 为正、`0.0` 为负。
    - 如果一条请求里有 `3` 个正样本、`2` 个负样本，则 `GAUC` 会形成 `3 x 2 = 6` 对正负样本比较；若存在权重，每一对的贡献按 `pos_weight * neg_weight` 计入。
    - 只有同时包含正负样本的请求才会对该指标有贡献；全正或全负请求会被跳过。越大越好。

## 数据准备

**若已有相应 csv 文件，可跳过这一步。**

直接执行：

```bash
python -m scripts.benchmark_data.movielens --dataset movielens-1m
```

即可完成 `movielens-1m` 的下载、解压和预处理。

若遇到 SSL 证书问题，执行：

```bash
python -m scripts.benchmark_data.movielens --dataset movielens-1m --insecure
```

但是由于内置下载通常较慢，推荐手动下载归档文件，并放到 `tmp/` 下对应文件名的位置；只要文件已经存在，脚本会跳过下载，直接解压和预处理。

官方下载链接如下：

- MovieLens 1M: [http://files.grouplens.org/datasets/movielens/ml-1m.zip](http://files.grouplens.org/datasets/movielens/ml-1m.zip)
- MovieLens 20M: [http://files.grouplens.org/datasets/movielens/ml-20m.zip](http://files.grouplens.org/datasets/movielens/ml-20m.zip)
- KuaiRand 1K: [https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz](https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz)

手动下载时，文件名需要对应到下面这些路径：

```text
tmp/movielens1m.zip
tmp/movielens20m.zip
tmp/KuaiRand-1K.tar.gz
```

放好归档后，再运行数据准备脚本：

准备 MovieLens：

```bash
python -m scripts.benchmark_data.movielens --dataset movielens-1m
python -m scripts.benchmark_data.movielens --dataset movielens-20m
```

准备 KuaiRand：

```bash
python -m scripts.benchmark_data.kuairand --dataset kuairand-1k
```

脚本会生成：

```text
data/ml-1m/sasrec_format.csv
data/ml-20m/sasrec_format.csv
data/KuaiRand-1K/processed_seqs.csv
```

MovieLens 的公开数据预处理与 upstream `generative-recommenders` 的 MovieLens 路径一致：

- `ml-1m` 读取 `users.dat` 与 `ratings.dat`
- `ml-20m` 读取 `ratings.csv`
- 先按 `unix_timestamp` 升序排序，再按 `user_id` 分组
- 分组后输出：
    - `sequence_item_ids`
    - `sequence_ratings`
    - `sequence_timestamps`
- `ml-1m` 额外包含：
    - `sex`
    - `age_group`
    - `occupation`
    - `zip_code`

KuaiRand 的公开数据预处理与 upstream `generative-recommenders` 的 KuaiRand 路径一致：

- `kuairand-1k` 读取两段公开 log CSV
- 按 `user_id` 聚合：
    - `video_id`
    - `time_ms`
    - `action_weights`
    - `play_time_ms`
    - `duration_ms`
- 合并用户侧特征：
    - `user_active_degree`
    - `follow_user_num_range`
    - `fans_user_num_range`
    - `friend_user_num_range`
    - `register_days_range`

## 请求构造

MovieLens 的请求构造：

- 每一行 CSV 对应一条用户序列
- 默认最后 `5` 个 item 作为 candidates，可通过 `--dataset-candidate-count` 调整
- 前面的 item 作为 `user_history`
- 序列长度 `<= candidate_count` 的样本会被跳过
- `item_query_time` 取 history 时间戳最大值
- `rating` 标签按原始 rating 阈值生成：
    - `raw_rating >= 3.5 => 1.0`
    - `raw_rating < 3.5 => 0.0`

KuaiRand 的请求构造：

- 每一行 `processed_seqs.csv` 对应一条用户序列
- 默认最后 `5` 个 item 作为 candidates，可通过 `--dataset-candidate-count` 调整，前面的作为 `user_history`
- 序列长度 `<= candidate_count` 的样本会被跳过
- `item_query_time` 取 history 时间戳最大值
- `labels_by_task` 从 candidate `action_weights` bitset 恢复 8 个任务：
    - `is_click`
    - `is_like`
    - `is_follow`
    - `is_comment`
    - `is_forward`
    - `is_hate`
    - `long_view`
    - `is_profile_enter`

## 运行 Benchmark

如果只是做链路联调或 smoke，可以把 `--model` 直接指向 `examples/configs` 里的配置目录，并配合 `--allow-missing-weights` 做 dummy run。benchmark 数据集优先使用 `*-benchmark` config；没有单独 benchmark 目录的数据集则继续使用对应普通 config。例如：

- `movielens-1m` -> `examples/configs/movielens-1m`
- `movielens-20m` -> `examples/configs/movielens-20m-benchmark`
- `kuairand-1k` -> `examples/configs/kuairand-benchmark`

```bash
python -m vllm_gr.benchmarks.models.hstu.gr_accuracy \
  --model examples/configs/movielens-1m \
  --dataset-name movielens-1m \
  --input-csv data/ml-1m/sasrec_format.csv \
  --allow-missing-weights
```

如果要有效评测模型精度，请把 `--model` 指向包含权重文件的目录：

运行 `movielens-1m`：

```bash
python -m vllm_gr.benchmarks.models.hstu.gr_accuracy \
  --model /path/to/model \
  --dataset-name movielens-1m \
  --input-csv data/ml-1m/sasrec_format.csv
```

运行 `movielens-20m`：

```bash
python -m vllm_gr.benchmarks.models.hstu.gr_accuracy \
  --model /path/to/model \
  --dataset-name movielens-20m \
  --input-csv data/ml-20m/sasrec_format.csv \
  --num-prompts 1024
```

运行 `kuairand-1k`：

```bash
python -m vllm_gr.benchmarks.models.hstu.gr_accuracy \
  --model /path/to/model \
  --dataset-name kuairand-1k \
  --input-csv data/KuaiRand-1K/processed_seqs.csv
```

当前 CLI 参数包括：

- `--model`
- `--dataset-name`
- `--input-csv`
- `--num-prompts`
- `--dataset-candidate-count`
- `--seed`
- `--max-seq-len`
- `--max-candidates`
- `--dtype`
- `--allow-missing-weights`
- `--disable-shuffle`
- `--disable-tqdm`
- `--output-records-jsonl`

如果设置了 `--allow-missing-weights`，benchmark 会在缺少权重文件时使用 dummy load。这个模式适合链路联调或 smoke，不适合将输出指标作为有效模型精度参考。

## 输出格式

命令会直接打印 JSON 汇总结果，例如:

```json
{
  "dataset_name": "movielens-1m",
  "metrics_by_task": {
    "rating": {
      "accuracy": 0.8,
      "gauc": 0.75,
      "ne": 0.9
    }
  },
  "num_candidates": 5120,
  "num_requests": 1024
}
```

如果设置了 `--output-records-jsonl`，还会额外输出逐请求 JSONL，每条记录包含：

- `request_id`
- `dataset_name`
- `task_names`
- `num_candidates`
- `predictions_by_task`
- `labels_by_task`
- `weights_by_task`

## 说明

- 当前 benchmark 覆盖 `movielens-1m`、`movielens-20m` 和 `kuairand-1k`
- MovieLens 输出单任务 `rating`
- KuaiRand 输出 8 个 task 的指标
- 指标计算依赖 `GRRequestOutput.all_scores`
- 未来新增模型 benchmark 时，在 `models/<model_family>/` 下扩展，并优先复用 `common/` 中的公共能力
