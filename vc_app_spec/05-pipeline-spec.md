# 05 — Pipeline Specification

## Pipeline Engine

- **Class**: `Pipeline(name: str)`
- **Storage**: Nodes stored in a dict by `node.name`; execution order cached and invalidated when nodes are added/removed.
- **Execution order**: Topological sort (Kahn’s algorithm) from each node’s `dependencies` list. Circular dependencies raise `PipelineError`.

## Node Contract

Every node:

1. Has a **unique name** (e.g. `load_videos`, `render_subtitles`). Used as key in pipeline and in dependency lists.
2. Declares **dependencies**: list of node names that must run (and succeed) before this node.
3. Implements **process(context: PipelineContext) -> NodeResult**. Must not call other nodes directly; only read/write context.
4. Optionally implements **validate(context) -> bool**. If false, node is failed without running process.
5. **execute(context)** (base class): sets status RUNNING, runs validate, then process; on exception returns failure result. Do not override execute unless necessary.

## Execution Semantics

1. Resolve execution order once per run (topological order).
2. For each node in order:
   - If any dependency has a non-success result → mark node SKIPPED, continue.
   - Call `before_node` hook if registered.
   - Run `node.execute(context)`.
   - If result has `output` is not None, call `context.store_node_output(node_name, result.output)`.
   - Call `after_node` hook.
   - If result failed and `stop_on_failure` → trigger `on_error` hook and break.
3. After loop: trigger `after_run` hook; return dict of node_name → NodeResult.

## Hooks

Pipeline supports hooks (event name → list of callbacks):

- `before_run(context)`
- `after_run(results, context)`
- `before_node(node_name, context)`
- `after_node(node_name, result, context)`
- `on_error(node_name, error, context)`

## Dry Run

`pipeline.dry_run(context)` returns execution order and prints a numbered list of nodes with their dependencies; does not run any node.

## Building from Config

`Pipeline.from_config(config: PipelineConfig, node_registry?: dict[str, type[Node]])`:

- Uses `config.pipeline_stages` (list of stage names).
- For each name, looks up node class in registry (default: plugin registry’s nodes).
- Instantiates node with `config=config.to_dict()` and adds to pipeline.
- Order in list does not override dependency graph; execution order is still topological.

## Errors

- **PipelineError**: e.g. unknown dependency, circular dependency.
- Node failure: `NodeResult.failure_result(exception)`; pipeline may stop or skip dependents depending on `stop_on_failure`.

## Main Clip Convention

- After load (single file), main clip is set: `context.set_main_clip(clips[0])`.
- Merge/transition nodes set the merged result as main.
- Crop, preview, subtitle render, overlay, speed, export all operate on `context.get_main_clip()` and typically call `context.set_main_clip(result_clip)` with the new clip.
