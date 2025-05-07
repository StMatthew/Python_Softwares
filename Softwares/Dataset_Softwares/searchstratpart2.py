import heapq
import time

def a_star_search(graph, start, goal, heuristic):
    """Performs A* Search and measures performance."""
    start_time = time.time()
    open_list = [(0 + heuristic[start], 0, start, [start])]
    closed_set = set()
    nodes_expanded = 0

    while open_list:
        f, g, current, path = heapq.heappop(open_list)
        nodes_expanded += 1

        if current == goal:
            end_time = time.time()
            execution_time = end_time - start_time
            return path, nodes_expanded, execution_time

        if current in closed_set:
            continue

        closed_set.add(current)

        for neighbor, cost in graph[current].items():
            if neighbor not in closed_set:
                new_g = g + cost
                new_f = new_g + heuristic[neighbor]
                heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))

    end_time = time.time()
    execution_time = end_time - start_time
    return None, nodes_expanded, execution_time

def analyze_a_star_performance(graph, start, goal, heuristic):
    """Analyzes A* performance and prints results."""
    path, nodes_expanded, execution_time = a_star_search(graph, start, goal, heuristic)

    print("--- A* Search Performance ---")
    print(f"Nodes Expanded: {nodes_expanded}")
    print(f"Execution Time: {execution_time} seconds")

    if path:
        print(f"Path Found: {path}")
    else:
        print("Path not found.")

# Example Graph (Weighted)
graph_a_star = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'D': 5, 'E': 12},
    'C': {'A': 4, 'F': 15},
    'D': {'B': 5},
    'E': {'B': 12, 'F': 1},
    'F': {'C': 15, 'E': 1}
}

# Heuristic (Example - Straight-line distance)
heuristic = {
    'A': 10,
    'B': 8,
    'C': 5,
    'D': 7,
    'E': 3,
    'F': 0
}

start_node_a_star = 'A'
goal_node_a_star = 'F'

analyze_a_star_performance(graph_a_star, start_node_a_star, goal_node_a_star, heuristic)