"""
Base Node class for pipeline stages.

Nodes are the fundamental building blocks of the pipeline. Each node represents
a discrete processing step that can be composed with other nodes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from videopipe.core.context import PipelineContext

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a node execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class NodeResult:
    """Result of a node execution."""
    status: NodeStatus
    output: Any = None
    error: Optional[Exception] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status == NodeStatus.COMPLETED
    
    @classmethod
    def success_result(cls, output: Any = None, **metadata) -> NodeResult:
        return cls(status=NodeStatus.COMPLETED, output=output, metadata=metadata)
    
    @classmethod
    def failure_result(cls, error: Exception, **metadata) -> NodeResult:
        return cls(status=NodeStatus.FAILED, error=error, metadata=metadata)
    
    @classmethod
    def skipped_result(cls, reason: str = "") -> NodeResult:
        return cls(status=NodeStatus.SKIPPED, metadata={"reason": reason})


class Node(ABC):
    """
    Abstract base class for all pipeline nodes.
    
    Each node should:
    - Define a unique name
    - Implement the process() method
    - Optionally implement validate() for input validation
    - Optionally define dependencies on other nodes
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        dependencies: Optional[list[str]] = None,
    ):
        self._name = name or self.__class__.__name__
        self._config = config or {}
        self._dependencies = dependencies or []
        self._status = NodeStatus.PENDING
        
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def config(self) -> dict[str, Any]:
        return self._config
    
    @property
    def dependencies(self) -> list[str]:
        return self._dependencies
    
    @property
    def status(self) -> NodeStatus:
        return self._status
    
    def validate(self, context: PipelineContext) -> bool:
        """
        Validate that the node can be executed with the current context.
        Override this method to add custom validation logic.
        
        Returns:
            True if validation passes, False otherwise.
        """
        return True
    
    @abstractmethod
    def process(self, context: PipelineContext) -> NodeResult:
        """
        Execute the node's processing logic.
        
        Args:
            context: The pipeline context containing shared state and data.
            
        Returns:
            NodeResult indicating success/failure and any output data.
        """
        pass
    
    def execute(self, context: PipelineContext) -> NodeResult:
        """
        Execute the node with proper status tracking and error handling.
        
        This is the main entry point for node execution. It handles:
        - Status tracking
        - Validation
        - Error handling
        - Logging
        """
        logger.info(f"Executing node: {self.name}")
        self._status = NodeStatus.RUNNING
        
        try:
            # Validate inputs
            if not self.validate(context):
                self._status = NodeStatus.FAILED
                return NodeResult.failure_result(
                    ValueError(f"Validation failed for node: {self.name}")
                )
            
            # Execute processing
            result = self.process(context)
            self._status = result.status
            
            if result.success:
                logger.info(f"Node {self.name} completed successfully")
            else:
                logger.error(f"Node {self.name} failed: {result.error}")
                
            return result
            
        except Exception as e:
            self._status = NodeStatus.FAILED
            logger.exception(f"Node {self.name} raised an exception")
            return NodeResult.failure_result(e)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status={self.status.value})"


class CompositeNode(Node):
    """
    A node that contains and executes multiple child nodes in sequence.
    Useful for grouping related operations.
    """
    
    def __init__(
        self,
        name: str,
        nodes: list[Node],
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self._nodes = nodes
        
    @property
    def nodes(self) -> list[Node]:
        return self._nodes
    
    def process(self, context: PipelineContext) -> NodeResult:
        results = []
        
        for node in self._nodes:
            result = node.execute(context)
            results.append(result)
            
            if not result.success:
                return NodeResult.failure_result(
                    result.error or Exception(f"Child node {node.name} failed"),
                    child_results=results
                )
        
        return NodeResult.success_result(
            output=results[-1].output if results else None,
            child_results=results
        )
