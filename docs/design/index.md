# Design and architecture

Read the architecture overview to understand where a request runs and which
component owns its state. Use feature contracts when changing a specific
execution boundary, and the patch inventory when changing upstream integration.

| Document | Questions it answers |
| --- | --- |
| [Architecture overview](main-arch.md) | How do frontends, EngineCore and Workers cooperate? What comes from native vLLM? |
| [Beam execution contract](../features/beam-search.md) | Which batch identities, receipts, failure states and retirement rules must hold? |
| [Graph and storage contracts](../features/cuda-graphs.md) | What is captured, who owns fixed storage and when may it be released? |
| [Constraint tables](../features/constraint-tables.md) | How are artifacts validated and catalog roots selected? |
| [Resource updates](../features/hot-update.md) | How do checkpoint and Trie updates publish, reconcile and recover? |
| [Runtime patch inventory](runtime_patch_inventory.md) | Which native boundaries are extended, and how are installations validated? |
| [Machine-readable inventory](patch_inventory.yaml) | Which patch IDs, owners, scopes and contract tests are authoritative? |

## Read a design with its implementation

Design documents describe ownership and contracts. Use
[development navigation](../contributing/development.md) to find actual code,
callers and tests before changing behavior. A shared interface or a CPU fixture
does not establish support for every device, runner or deployment topology.
