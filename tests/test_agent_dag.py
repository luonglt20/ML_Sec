import time
import pytest
from medqa_multiagent.agents import DAGGraph, DAGNode, DAGRunner


def test_dag_topological_sort_and_cycle_detection():
    graph = DAGGraph()
    n1 = DAGNode("Router", lambda ctx: "query_1")
    n2 = DAGNode("Memory", lambda ctx: "memory_brief")
    n3 = DAGNode("Researcher", lambda ctx: f"passages for {ctx['Router']}", dependencies={"Router"})
    n4 = DAGNode("Reasoner", lambda ctx: f"reason with {ctx['Researcher']} and {ctx['Memory']}", dependencies={"Researcher", "Memory"})

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)
    graph.add_node(n4)

    assert graph.detect_cycles() is False
    order = graph.topological_sort()
    assert len(order) == 4
    assert order.index("Router") < order.index("Researcher")
    assert order.index("Researcher") < order.index("Reasoner")
    assert order.index("Memory") < order.index("Reasoner")


def test_dag_cycle_detection():
    graph = DAGGraph()
    n1 = DAGNode("A", lambda ctx: "A", dependencies={"B"})
    n2 = DAGNode("B", lambda ctx: "B", dependencies={"A"})
    graph.add_node(n1)
    graph.add_node(n2)

    assert graph.detect_cycles() is True
    with pytest.raises(ValueError, match="contains a cycle"):
        graph.topological_sort()


def test_dag_concurrent_execution():
    graph = DAGGraph()

    def slow_task_1(ctx):
        time.sleep(0.05)
        return "Task 1 Done"

    def slow_task_2(ctx):
        time.sleep(0.05)
        return "Task 2 Done"

    def combine_task(ctx):
        return f"{ctx['Task1']} + {ctx['Task2']}"

    # Task1 and Task2 can run concurrently in parallel!
    graph.add_node(DAGNode("Task1", slow_task_1))
    graph.add_node(DAGNode("Task2", slow_task_2))
    graph.add_node(DAGNode("Combine", combine_task, dependencies={"Task1", "Task2"}))

    runner = DAGRunner(max_workers=4)
    t0 = time.monotonic()
    result = runner.run(graph)
    elapsed = time.monotonic() - t0

    # Total time should be ~0.05s (parallel) rather than 0.10s (sequential)
    assert elapsed < 0.09
    assert result["Combine"] == "Task 1 Done + Task 2 Done"
