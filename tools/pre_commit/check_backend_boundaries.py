# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Guard the shared runtime side of the RFC #706 change-boundary table."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHARED = (
    "_env_check.py",
    "engine/arg_utils_gr.py",
    "entrypoints/beam_search_v1/input_processor.py",
    "config/models.py",
    "constraints/loader.py",
    "constraints/startup.py",
    "constraints/registry.py",
    "constraints/execution.py",
    "v1/engine/beam_search_v1.py",
    "v1/engine/core.py",
    "v1/engine/gr_async_scheduler.py",
    "v1/worker/gr_execution.py",
    "v1/worker/gr_startup.py",
    "v1/worker/gr_context.py",
    "v1/worker/graph_capture.py",
    "v1/worker/prefill_graph_common.py",
    "v1/worker/prefill_graph_runner.py",
    "v1/worker/model_runner_patch.py",
    "v1/worker/runner_lifecycle.py",
    "v1/attention/backends/beam_attn.py",
    "v1/attention/backends/beam_attn_metadata.py",
)
# Native kernel packages and the remaining flat worker implementations own their
# hardware dependencies. A new shared module is scanned regardless of its name.
_KERNEL_PACKAGES = ("ops/cuda/", "ops/npu/")
_HARDWARE_MODULES = {
    "ops/npu_stage_logits.py",
    "v1/beam/gpu_beam_postprocess_workspace.py",
    "v1/beam/npu_dynamic_beam.py",
    "v1/beam/npu_execution_batch.py",
}


def shared_paths(package: Path) -> list[str]:
    paths = set(SHARED)
    # The direct children of platforms are the shared API and discovery layer.
    # Built-in provider implementations live under their own subpackages.
    paths.update(p.relative_to(package).as_posix() for p in (package / "platforms").glob("*.py"))
    for tree in ("ops", "v1/beam"):
        for source in (package / tree).rglob("*.py"):
            path = source.relative_to(package).as_posix()
            if path not in _HARDWARE_MODULES and not path.startswith(_KERNEL_PACKAGES):
                paths.add(path)
    return sorted(paths)


_VENDOR = {"cuda", "npu", "ascend", "gpu", "rocm", "xpu", "hpu", "tpu", "mps"}
_IDENTITY = {
    "device",
    "device_type",
    "device_name",
    "backend_id",
    "platform_enum",
    "current_platform",
}


def _vendor_module(module: str) -> bool:
    parts = module.lower().split(".")
    if parts[0] in {"torch_npu", "vllm_ascend", "triton", "flash_attn", "habana_frameworks"}:
        return True
    if parts[:2] in (["torch", "ops"], ["torch", "cuda"], ["torch", "npu"]):
        return True
    return any(
        word in _VENDOR or word.startswith("cudagraph")
        for part in parts
        for word in part.split("_")
    )


class Visitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scopes: list[str] = []
        self.imports: dict[str, str] = {}
        self.identities: set[str] = set()
        self.dispatch_tables: set[str] = set()
        self.checked_predicates: set[ast.AST] = set()
        self.findings: list[tuple[str, str]] = []

    def record(self, value: str) -> None:
        self.findings.append((".".join(self.scopes) or "<module>", value))

    def scope(self, node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scopes.append(node.name)
        outer = self.identities
        tables = self.dispatch_tables
        self.identities = set(outer)
        self.dispatch_tables = set(tables)
        self.generic_visit(node)
        self.identities = outer
        self.dispatch_tables = tables
        self.scopes.pop()

    visit_FunctionDef = scope
    visit_AsyncFunctionDef = scope
    visit_ClassDef = scope

    def visit_Import(self, node: ast.Import) -> None:
        for name in node.names:
            self.imports[name.asname or name.name] = name.name
            if _vendor_module(name.name):
                self.record("import " + name.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for name in node.names:
            module = f"{node.module or ''}.{name.name}"
            self.imports[name.asname or name.name] = module
            if _vendor_module(node.module or ""):
                self.record("from " + (node.module or "") + " import " + name.name)
            elif _vendor_module(module) and name.name.islower():
                self.record("from " + (node.module or "") + " import " + name.name)

    def visit_Call(self, node: ast.Call) -> None:
        target = ast.unparse(node.func)
        if self.imports.get(target, target).split(".")[-1] in {"import_module", "__import__"}:
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                if _vendor_module(node.args[0].value):
                    self.record("import " + node.args[0].value)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get" and node.args:
            if self.literal_table(node.func.value) and self.identity(node.args[0]):
                self.record("dispatch " + ast.unparse(node))
        self.generic_visit(node)

    def identity(self, expr: ast.AST) -> bool:
        for part in ast.walk(expr):
            if isinstance(part, ast.Name) and (
                part.id in self.identities
                or self.imports.get(part.id, part.id).split(".")[-1] in _IDENTITY
            ):
                return True
            if isinstance(part, ast.Attribute) and part.attr in _IDENTITY:
                return True
            if (
                isinstance(part, ast.Attribute)
                and part.attr == "type"
                and isinstance(part.value, ast.Attribute)
                and part.value.attr == "device"
            ):
                return True
            if (
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Attribute)
                and part.func.attr in {"get", "getenv"}
            ):
                if (
                    part.args
                    and isinstance(part.args[0], ast.Constant)
                    and part.args[0].value in {"VLLM_GR_BACKEND", "VLLM_TARGET_DEVICE"}
                ):
                    return True
            if (
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Name)
                and part.func.id == "getattr"
            ):
                if (
                    len(part.args) >= 2
                    and isinstance(part.args[1], ast.Constant)
                    and part.args[1].value in _IDENTITY
                ):
                    return True
        return False

    def visit_Assign(self, node: ast.Assign) -> None:
        if self.identity(node.value):
            self.identities.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        if self.scopes == ["ConstraintTableBackendEnum"]:
            self.record(ast.unparse(node))
        if self.literal_table(node.value):
            self.dispatch_tables.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if self.scopes == ["ConstraintTableBackendEnum"]:
            self.record(ast.unparse(node))
        if node.value is not None and isinstance(node.target, ast.Name):
            if self.identity(node.value):
                self.identities.add(node.target.id)
            if self.literal_table(node.value):
                self.dispatch_tables.add(node.target.id)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # Only compare literal candidates on one side against a platform on the
        # other. Comparing plan.backend_id with a wire field is an identity check.
        for left, right in zip([node.left, *node.comparators], node.comparators):
            for literal, subject in ((left, right), (right, left)):
                if isinstance(literal, (ast.List, ast.Tuple, ast.Set)):
                    candidates = literal.elts
                else:
                    candidates = [literal]
                if any(
                    isinstance(item, ast.Constant) and isinstance(item.value, str) and item.value
                    for item in candidates
                ) and self.identity(subject):
                    self.record(ast.unparse(node))
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        if self.identity(node.subject):
            for case in node.cases:
                if isinstance(case.pattern, (ast.MatchValue, ast.MatchOr)):
                    self.record(
                        "match " + ast.unparse(node.subject) + " case " + ast.unparse(case.pattern)
                    )
        self.generic_visit(node)

    def check_predicate(self, expr: ast.AST) -> None:
        for part in ast.walk(expr):
            # A conditional expression inside a return has two visitors. Count
            # the predicate once, while still counting repeated source sites.
            if part in self.checked_predicates:
                continue
            self.checked_predicates.add(part)
            name = (
                part.attr
                if isinstance(part, ast.Attribute)
                else part.id
                if isinstance(part, ast.Name)
                else ""
            )
            name = self.imports.get(name, name).split(".")[-1]
            if name.startswith("is_") and set(name.split("_")) & _VENDOR:
                self.record("predicate " + name)
            elif isinstance(part, ast.Call) and isinstance(part.func, ast.Attribute):
                owner = ast.unparse(part.func.value).split(".")[-1]
                if owner in _VENDOR and part.func.attr == "is_available":
                    self.record("predicate " + ast.unparse(part))

    def visit_If(self, node: ast.If | ast.IfExp | ast.While) -> None:
        if isinstance(node, ast.If) and ast.unparse(node.test) in {
            "TYPE_CHECKING",
            "typing.TYPE_CHECKING",
        }:
            for child in node.orelse:
                self.visit(child)
            return
        self.check_predicate(node.test)
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            self.check_predicate(node.value)
        self.generic_visit(node)

    visit_IfExp = visit_If
    visit_While = visit_If

    def literal_table(self, node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id in self.dispatch_tables) or (
            isinstance(node, ast.Dict)
            and any(
                isinstance(key, ast.Constant) and isinstance(key.value, str) for key in node.keys
            )
        )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if self.literal_table(node.value) and self.identity(node.slice):
            self.record("dispatch " + ast.unparse(node))
        self.generic_visit(node)


def findings(source: str) -> list[tuple[str, str]]:
    visitor = Visitor()
    visitor.visit(ast.parse(source))
    return visitor.findings


def main(
    package: Path = ROOT / "vllm_gr",
    allowlist: Path = Path(__file__).with_name("backend_boundary_allowlist.json"),
) -> int:
    reviewed = json.loads(allowlist.read_text(encoding="utf-8"))
    allowed: Counter[tuple[str, str, str]] = Counter()
    for item in reviewed:
        allowed[item["file"], item["scope"], item["expression"]] += item.get("count", 1)
    actual = Counter(
        (path, scope, expression)
        for path in shared_paths(package)
        for scope, expression in findings((package / path).read_text(encoding="utf-8"))
    )
    # Backend-named modules belong alongside their provider even when the
    # module has no vendor imports or dispatch expressions of its own.
    platforms = package / "platforms"
    for owner in platforms.glob("*/provider.py"):
        for source in platforms.glob(f"{owner.parent.name}_*.py"):
            actual[source.relative_to(package).as_posix(), "<module>", "flat backend module"] += 1
    for description, difference in (
        ("unreviewed", actual - allowed),
        ("stale exemption", allowed - actual),
    ):
        for (path, scope, expression), count in sorted(difference.items()):
            print(f"{description}: {path}:{scope}: {expression} ({count})")
    return int(actual != allowed)


if __name__ == "__main__":
    raise SystemExit(main())
