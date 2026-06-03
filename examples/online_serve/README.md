# vllm-GR Online Serve 示例

如果你是升级代码后的现有环境，请先按根 [README.md](../../README.md) 的安装方式重新安装环境，以带上新增依赖。

## 1. 前提

- 已安装 `vllm-gr` 及其运行依赖
- 可以访问本仓库目录
- `demo` 分支使用 `examples/configs/movielens-1m`，该目录当前只包含 `config.json`，不包含真实权重
- `load_weight` 分支使用你本地的已转换模型目录，该目录需要包含：
    - `config.json`
    - `model.safetensors` 或等价权重文件

## 2. Demo 分支

`demo` 分支使用仓库内的样例配置目录启动，不加载真实权重。

启动命令：

```bash
vllm-gr serve examples/configs/movielens-1m \
  --served-model-name movielens-1m \
  --model-family hstu \
  --allow-missing-weights \
  --host 127.0.0.1 \
  --port 8000
```

说明：

- 使用的是 `vllm-gr serve`，不是 `vllm serve`
- 模型目录使用仓库根目录相对路径
- `--allow-missing-weights` 只用于这条 demo 分支

## 3. Load Weight 分支

`load_weight` 分支使用已转换好的真实权重模型目录，场景与 [load_converted_model.py](../load_converted_model.py) 一致。

启动命令：

```bash
vllm-gr serve <converted_model_dir> \
  --served-model-name movielens-1m \
  --model-family hstu \
  --host 127.0.0.1 \
  --port 8000
```

说明：

- 这条分支不带 `--allow-missing-weights`
- 共享请求样例仍然使用 `model: movielens-1m`

## 5. 调用方式

无论你启动的是 `demo` 还是 `load_weight`，调用方式都相同。

先做健康检查：

```bash
curl -v --max-time 5 "http://127.0.0.1:8000/health"
```

再发送推荐请求：

```bash
curl -v --max-time 30 -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  --data-binary "@examples/online_serve/recommend_request.json"
```

返回 `200` 时，响应中应包含：

- `recommendations`
- 每个 `recommendation` 默认会包含 `index`、`item_id`、`score`
- 如果模型输出了多任务分数，还会包含 `task_scores`
- 响应中的 `model` 应为 `movielens-1m`

## 6. 使用 Python 客户端调用

脚本文件为 [recommend_client.py](recommend_client.py)。

```bash
python examples/online_serve/recommend_client.py
```
