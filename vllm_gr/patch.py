try:
    from vllm_gr.benchmarks.open_one_rec import onerec_patch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    onerec_patch = None  # type: ignore
