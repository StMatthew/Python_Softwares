import collections
import time

def breadth_first_search(graph, start, goal):
    """Performs Breadth-First Search and measures performance."""
    start_time = time.time()
    queue = collections.deque([(start, [start])])
    visited = {start}
    nodes_expanded = 0

    while queue:
        node, path = queue.popleft()
        nodes_expanded += 1
        if node == goal:
            end_time = time.time()
            execution_time = end_time - start_time
            return path, nodes_expanded, execution_time

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    end_time = time.time()
    execution_time = end_time - start_time
    return None, nodes_expanded, execution_time

def analyze_bfs_performance(graph, start, goal):
    """Analyzes BFS performance and prints results."""
    path, nodes_expanded, execution_time = breadth_first_search(graph, start, goal)

    print("--- Breadth-First Search Performance ---")
    print(f"Nodes Expanded: {nodes_expanded}")
    print(f"Execution Time: {execution_time} seconds")

    if path:
        print(f"Path Found: {path}")
    else:
        print("Path not found.")

# Example Graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

start_node = 'A'
goal_node = 'F'

analyze_bfs_performance(graph, start_node, goal_node)