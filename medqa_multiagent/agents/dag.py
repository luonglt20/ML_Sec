"""Pure Python Directed Acyclic Graph (DAG) Execution Engine for Multi-Agent Workflows.

Provides DAGNode, DAGGraph, and DAGRunner supporting topological ordering, cycle detection,
and concurrent parallel execution of independent AI agents without external frameworks.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class DAGNode:
    """A node in the DAG representing an agent action or computation step."""

    name: str
    action_func: Callable[[Dict[str, Any]], Any]
    dependencies: Set[str] = field(default_factory=set)

    def is_ready(self, executed_nodes: Set[str]) -> bool:
        """Check if all parent dependencies have been executed."""
        return self.dependencies.issubset(executed_nodes)


class DAGGraph:
    """Directed Acyclic Graph managing nodes, dependency resolution, and topological sorting."""

    def __init__(self) -> None:
        self.nodes: Dict[str, DAGNode] = {}

    def add_node(self, node: DAGNode) -> None:
        """Add a DAGNode to the graph."""
        self.nodes[node.name] = node

    def detect_cycles(self) -> bool:
        """Detect if the graph contains any cyclic dependencies (DFS)."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(name: str) -> bool:
            visited.add(name)
            rec_stack.add(name)
            node = self.nodes.get(name)
            if node:
                for dep in node.dependencies:
                    if dep not in visited:
                        if dfs(dep):
                            return True
                    elif dep in rec_stack:
                        return True
            rec_stack.remove(name)
            return False

        for node_name in self.nodes:
            if node_name not in visited:
                if dfs(node_name):
                    return True
        return False

    def topological_sort(self) -> List[str]:
        """Return topological ordering of node names."""
        if self.detect_cycles():
            raise ValueError("DAGGraph contains a cycle! Cannot perform topological sort.")

        in_degree = {name: len(node.dependencies) for name, node in self.nodes.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for name, node in self.nodes.items():
                if curr in node.dependencies:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)

        if len(order) != len(self.nodes):
            raise ValueError("DAGGraph topological sort incomplete.")
        return order


class DAGRunner:
    """Executes a DAGGraph concurrently using pure Python ThreadPoolExecutor."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def run(self, graph: DAGGraph, initial_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute all nodes in the DAG respecting dependencies and running ready nodes concurrently."""
        if graph.detect_cycles():
            raise ValueError("Cannot execute graph with cycles.")

        context: Dict[str, Any] = dict(initial_context or {})
        executed_nodes: Set[str] = set()
        pending_nodes: Set[str] = set(graph.nodes.keys())

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while pending_nodes:
                # Find all nodes that are ready to execute
                ready_nodes = [
                    graph.nodes[name]
                    for name in pending_nodes
                    if graph.nodes[name].is_ready(executed_nodes)
                ]

                if not ready_nodes and pending_nodes:
                    raise RuntimeError("Deadlock encountered in DAG execution.")

                # Submit ready nodes concurrently
                future_to_name = {
                    executor.submit(node.action_func, context): node.name
                    for node in ready_nodes
                }

                for future in as_completed(future_to_name):
                    name = future_to_name[future]
                    result = future.result()
                    context[name] = result
                    executed_nodes.add(name)
                    pending_nodes.remove(name)

        return context
