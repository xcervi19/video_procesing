"""
Pipeline Engine - The main orchestrator for video processing.

Implements a DAG-based execution model where nodes can declare dependencies
and are executed in the correct order automatically.
"""

from __future__ import annotations

import logging
import json
import time
from collections import defaultdict
from typing import Any, Callable, Optional

from videopipe.core.node import Node, NodeResult, NodeStatus
from videopipe.core.context import PipelineContext
from videopipe.core.config import PipelineConfig

logger = logging.getLogger(__name__)

# #region agent log
LOG_PATH = "/Users/karel.cervicek/Documents/projects/video_procesing/.cursor/debug.log"
def _debug_log(hyp, loc, msg, data):
    with open(LOG_PATH, "a") as f: f.write(json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": time.time()}) + "\n")
# #endregion


class PipelineError(Exception):
    """Exception raised when pipeline execution fails."""
    
    def __init__(self, message: str, node_name: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.node_name = node_name
        self.cause = cause


class Pipeline:
    """
    Main pipeline class that orchestrates node execution.
    
    The pipeline:
    - Manages a collection of nodes
    - Resolves dependencies between nodes
    - Executes nodes in topological order
    - Handles errors and provides hooks for monitoring
    
    Example:
        pipeline = Pipeline()
        pipeline.add_node(LoadVideosNode())
        pipeline.add_node(SubtitleNode(dependencies=["LoadVideosNode"]))
        pipeline.add_node(ExportNode(dependencies=["SubtitleNode"]))
        
        context = PipelineContext(input_files=[Path("video.mp4")])
        result = pipeline.run(context)
    """
    
    def __init__(self, name: str = "VideoPipeline"):
        self.name = name
        self._nodes: dict[str, Node] = {}
        self._execution_order: list[str] = []
        self._hooks: dict[str, list[Callable]] = defaultdict(list)
        
    def add_node(self, node: Node) -> Pipeline:
        """
        Add a node to the pipeline.
        
        Args:
            node: The node to add
            
        Returns:
            Self for method chaining
        """
        if node.name in self._nodes:
            raise ValueError(f"Node with name '{node.name}' already exists")
        
        self._nodes[node.name] = node
        self._execution_order = []  # Invalidate cached order
        logger.debug(f"Added node: {node.name}")
        
        return self
    
    def add_nodes(self, nodes: list[Node]) -> Pipeline:
        """Add multiple nodes to the pipeline."""
        for node in nodes:
            self.add_node(node)
        return self
    
    def remove_node(self, name: str) -> Pipeline:
        """Remove a node from the pipeline by name."""
        if name in self._nodes:
            del self._nodes[name]
            self._execution_order = []  # Invalidate cached order
        return self
    
    def get_node(self, name: str) -> Optional[Node]:
        """Get a node by name."""
        return self._nodes.get(name)
    
    @property
    def nodes(self) -> list[Node]:
        """Get all nodes in the pipeline."""
        return list(self._nodes.values())
    
    def _resolve_execution_order(self) -> list[str]:
        """
        Resolve the execution order using topological sort.
        
        Returns:
            List of node names in execution order
            
        Raises:
            PipelineError: If there's a circular dependency
        """
        if self._execution_order:
            return self._execution_order
        
        # Kahn's algorithm for topological sort
        in_degree: dict[str, int] = {name: 0 for name in self._nodes}
        graph: dict[str, list[str]] = defaultdict(list)
        
        # Build dependency graph
        for name, node in self._nodes.items():
            for dep in node.dependencies:
                if dep not in self._nodes:
                    raise PipelineError(
                        f"Node '{name}' depends on unknown node '{dep}'"
                    )
                graph[dep].append(name)
                in_degree[name] += 1
        
        # Find nodes with no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            # Sort for deterministic order
            queue.sort()
            node_name = queue.pop(0)
            result.append(node_name)
            
            for dependent in graph[node_name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        if len(result) != len(self._nodes):
            # Find the cycle
            remaining = set(self._nodes.keys()) - set(result)
            raise PipelineError(
                f"Circular dependency detected involving nodes: {remaining}"
            )
        
        self._execution_order = result
        return result
    
    def add_hook(self, event: str, callback: Callable) -> Pipeline:
        """
        Add a hook callback for pipeline events.
        
        Events:
            - before_run: Called before pipeline starts
            - after_run: Called after pipeline completes
            - before_node: Called before each node (receives node name)
            - after_node: Called after each node (receives node name, result)
            - on_error: Called when an error occurs
        """
        self._hooks[event].append(callback)
        return self
    
    def _trigger_hooks(self, event: str, *args, **kwargs):
        """Trigger all hooks for an event."""
        for callback in self._hooks[event]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Hook callback failed for event '{event}': {e}")
    
    def run(
        self,
        context: PipelineContext,
        stop_on_failure: bool = True,
    ) -> dict[str, NodeResult]:
        """
        Execute the pipeline.
        
        Args:
            context: The pipeline context with input data
            stop_on_failure: Whether to stop execution if a node fails
            
        Returns:
            Dictionary mapping node names to their results
        """
        logger.info(f"Starting pipeline: {self.name}")
        self._trigger_hooks("before_run", context)
        
        execution_order = self._resolve_execution_order()
        results: dict[str, NodeResult] = {}
        
        # #region agent log
        _debug_log("F", "pipeline.py:run", "Execution order resolved", {"execution_order": execution_order, "nodes_count": len(self._nodes), "all_nodes": list(self._nodes.keys())})
        # #endregion
        
        logger.info(f"Execution order: {' -> '.join(execution_order)}")
        
        for node_name in execution_order:
            node = self._nodes[node_name]
            
            # #region agent log
            _debug_log("G", "pipeline.py:run_loop", "Processing node", {"node_name": node_name, "dependencies": node.dependencies})
            # #endregion
            
            # Check if dependencies succeeded
            deps_ok = all(
                results.get(dep, NodeResult.failure_result(Exception())).success
                for dep in node.dependencies
            )
            
            # #region agent log
            _debug_log("G", "pipeline.py:deps_check", "Dependencies check", {"node_name": node_name, "deps_ok": deps_ok, "deps": node.dependencies, "results_so_far": list(results.keys())})
            # #endregion
            
            if not deps_ok:
                logger.warning(f"Skipping node '{node_name}' due to failed dependencies")
                # #region agent log
                _debug_log("G", "pipeline.py:skipped", "SKIPPED node", {"node_name": node_name, "reason": "deps_failed"})
                # #endregion
                results[node_name] = NodeResult.skipped_result(
                    reason="Dependencies failed"
                )
                continue
            
            self._trigger_hooks("before_node", node_name, context)
            
            try:
                # #region agent log
                _debug_log("H", "pipeline.py:before_execute", "About to execute node", {"node_name": node_name})
                # #endregion
                
                result = node.execute(context)
                results[node_name] = result
                
                # #region agent log
                main_clip = context.get_main_clip()
                main_dur = float(main_clip.duration) if main_clip and hasattr(main_clip, "duration") else None
                _debug_log("H", "pipeline.py:after_execute", "Node executed", {"node_name": node_name, "success": result.success, "main_clip_duration_after": main_dur})
                # #endregion

                # Store output in context for other nodes to use
                if result.output is not None:
                    context.store_node_output(node_name, result.output)
                
                self._trigger_hooks("after_node", node_name, result, context)
                
                if not result.success and stop_on_failure:
                    logger.error(f"Pipeline stopped due to failure in node: {node_name}")
                    self._trigger_hooks("on_error", node_name, result.error, context)
                    break
                    
            except Exception as e:
                logger.exception(f"Unhandled exception in node: {node_name}")
                results[node_name] = NodeResult.failure_result(e)
                self._trigger_hooks("on_error", node_name, e, context)
                
                if stop_on_failure:
                    break
        
        self._trigger_hooks("after_run", results, context)
        
        # Log summary
        completed = sum(1 for r in results.values() if r.status == NodeStatus.COMPLETED)
        failed = sum(1 for r in results.values() if r.status == NodeStatus.FAILED)
        skipped = sum(1 for r in results.values() if r.status == NodeStatus.SKIPPED)
        
        logger.info(
            f"Pipeline complete - Completed: {completed}, Failed: {failed}, Skipped: {skipped}"
        )
        
        return results
    
    def dry_run(self, context: PipelineContext) -> list[str]:
        """
        Perform a dry run showing what would be executed.
        
        Returns:
            List of node names in execution order
        """
        order = self._resolve_execution_order()
        
        print(f"\nDry run for pipeline: {self.name}")
        print("=" * 50)
        
        for i, node_name in enumerate(order, 1):
            node = self._nodes[node_name]
            deps = f" (depends on: {', '.join(node.dependencies)})" if node.dependencies else ""
            print(f"  {i}. {node_name}{deps}")
        
        print("=" * 50)
        return order
    
    @classmethod
    def from_config(
        cls,
        config: PipelineConfig,
        node_registry: Optional[dict[str, type[Node]]] = None,
    ) -> Pipeline:
        """
        Create a pipeline from a configuration.
        
        Args:
            config: Pipeline configuration
            node_registry: Dictionary mapping stage names to Node classes
            
        Returns:
            Configured Pipeline instance
        """
        from videopipe.plugins.registry import get_registry
        
        pipeline = cls(name="ConfiguredPipeline")
        registry = node_registry or get_registry().get_all_nodes()
        
        for stage_name in config.pipeline_stages:
            if stage_name not in registry:
                raise PipelineError(f"Unknown pipeline stage: {stage_name}")
            
            node_class = registry[stage_name]
            node = node_class(config=config.to_dict())
            pipeline.add_node(node)
        
        return pipeline
    
    def __repr__(self) -> str:
        return f"Pipeline(name='{self.name}', nodes={len(self._nodes)})"
