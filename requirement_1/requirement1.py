import heapq
from collections import deque
import time
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class PathFinder:
    def __init__(self, graph, heuristic=None):
        self.graph = graph
        self.heuristic = heuristic or {node: 0 for node in graph}

    def compare_algorithms_with_accuracy(self, start, goal):
        """Compare all algorithms and generate performance metrics including accuracy."""
        algorithms = [
            ("BFS", self.bfs),
            ("DFS", self.dfs),
            ("GBFS", self.gbfs),
            ("A*", self.a_star)
        ]

        results = []

        # Get the optimal path cost using A* (Ground Truth)
        optimal_path, optimal_cost = self.a_star(start, goal)

        for name, algo in algorithms:
            start_time = time.time()
            path, cost = algo(start, goal)
            execution_time = time.time() - start_time

            # Calculate Accuracy
            if optimal_cost and cost:  # Avoid division by zero
                accuracy = (optimal_cost / cost) * 100
            else:
                accuracy = 0  # If no valid path, accuracy is 0%

            results.append({
                "Algorithm": name,
                "Path": path if path else "No Path Found",
                "Cost": cost if cost else "N/A",
                "Time (ms)": round(execution_time * 1000, 6),  # Convert to milliseconds
                "Path Length": len(path) if path else 0,
                "Accuracy (%)": round(accuracy, 2)
            })

        return pd.DataFrame(results)

        
    def calculate_path_cost(self, path):
        """Calculate the total cost of a path."""
        if not path or len(path) < 2:
            return 0
        cost = 0
        for i in range(len(path) - 1):
            for neighbor, weight in self.graph[path[i]]:
                if neighbor == path[i + 1]:
                    cost += weight
                    break
        return cost

    def bfs(self, start, goal):
        """Improved BFS with early stopping and path tracking."""
        visited = {start: None}  # Store parent nodes
        queue = deque([start])
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                break
                
            for neighbor, _ in self.graph[current]:
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
        
        # Reconstruct path
        if goal not in visited:
            return None, None
            
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = visited[current]
            
        path.reverse()
        return path, self.calculate_path_cost(path)

    def dfs(self, start, goal):
        """Improved DFS with iterative implementation to prevent stack overflow."""
        visited = {start: None}
        stack = [(start, [start])]
        
        while stack:
            current, path = stack.pop()
            
            if current == goal:
                return path, self.calculate_path_cost(path)
                
            for neighbor, _ in self.graph[current]:
                if neighbor not in visited:
                    visited[neighbor] = current
                    stack.append((neighbor, path + [neighbor]))
        
        return None, None

    def gbfs(self, start, goal):
        """Improved GBFS with better priority queue handling."""
        visited = set()
        priority_queue = [(self.heuristic[start], start, [start])]
        
        while priority_queue:
            _, current, path = heapq.heappop(priority_queue)
            
            if current == goal:
                return path, self.calculate_path_cost(path)
                
            if current not in visited:
                visited.add(current)
                for neighbor, _ in self.graph[current]:
                    if neighbor not in visited:
                        heapq.heappush(priority_queue, 
                                     (self.heuristic[neighbor], neighbor, path + [neighbor]))
        
        return None, None

    def a_star(self, start, goal):
        """Improved A* with better memory management and path reconstruction."""
        visited = set()
        came_from = {start: None}
        g_score = {start: 0}
        f_score = {start: self.heuristic[start]}
        
        open_set = [(f_score[start], start)]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path, g_score[goal]
                
            visited.add(current)
            
            for neighbor, weight in self.graph[current]:
                if neighbor in visited:
                    continue
                    
                tentative_g_score = g_score[current] + weight
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic[neighbor]
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None, None

    def visualize_graph(self):
        """Visualize the graph using networkx."""
        G = nx.Graph()
        for node in self.graph:
            for neighbor, weight in self.graph[node]:
                G.add_edge(node, neighbor, weight=weight)
        
        pos = nx.spring_layout(G)
        plt.figure(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=12, font_weight='bold')
        
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        plt.title("Graph Visualization")
        plt.show()


def calculate_realistic_heuristic(graph, goal):
    """Calculate more realistic heuristic values using shortest path distances."""
    distances = {node: float('inf') for node in graph}
    distances[goal] = 0
    queue = [(0, goal)]
    
    while queue:
        dist, current = heapq.heappop(queue)
        
        if dist > distances[current]:
            continue
            
        for neighbor, weight in graph[current]:
            distance = dist + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))
    
    return distances

def main():
    # Define the graph
    graph = {
        "WL": [("Y", 1), ("PG", 6), ("YCK", 4)],
        "Y": [("WL", 1), ("YCK", 2)],
        "PG": [("WL", 6), ("PS", 2)],
        "YCK": [("WL", 4), ("Y", 2), ("PG", 5), ("TP", 3)],
        "TP": [("YCK", 3), ("CA", 1)],
        "PS": [("PG", 2), ("YCK", 5), ("CA", 2)],
        "CA": [("PS", 2), ("TP", 1)]
    }

    # Create PathFinder instance
    pathfinder = PathFinder(graph)
    
    while True:
        print("\n=== Pathfinding Algorithm Testing ===")
        print("\nAvailable locations:", ", ".join(sorted(graph.keys())))
        print("\nOptions:")
        print("1. Test all algorithms")
        print("2. Test specific algorithm")
        print("3. Visualize graph")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
            # Get start and goal nodes
            while True:
                start = input("\nEnter starting location (e.g., WL): ").upper()
                if start in graph:
                    break
                print("Invalid location! Please choose from available locations.")
            
            while True:
                goal = input("Enter destination location (e.g., CA): ").upper()
                if goal in graph:
                    break
                print("Invalid location! Please choose from available locations.")
            
            # Calculate heuristic and run comparison
            heuristic = calculate_realistic_heuristic(graph, goal)
            pathfinder.heuristic = heuristic
            results_df = pathfinder.compare_algorithms_with_accuracy(start, goal)
            
            print("\nResults:")
            print(results_df)

        elif choice == "2":
            print("\nAvailable algorithms:")
            print("1. Breadth-First Search (BFS)")
            print("2. Depth-First Search (DFS)")
            print("3. Greedy Best-First Search (GBFS)")
            print("4. A* Search")
            
            algo_choice = input("\nEnter algorithm number (1-4): ")

            start = input("\nEnter starting location: ").upper()
            goal = input("Enter destination location: ").upper()

            if start not in graph or goal not in graph:
                print("Invalid locations. Please try again.")
                continue

            heuristic = calculate_realistic_heuristic(graph, goal)
            pathfinder.heuristic = heuristic

            # Run selected algorithm
            algo_mapping = {"1": ("BFS", pathfinder.bfs),
                            "2": ("DFS", pathfinder.dfs),
                            "3": ("GBFS", pathfinder.gbfs),
                            "4": ("A*", pathfinder.a_star)}

            if algo_choice in algo_mapping:
                algo_name, algo_func = algo_mapping[algo_choice]
                path, cost = algo_func(start, goal)

                if path:
                    print(f"\n{algo_name} Results:")
                    print("Path:", " -> ".join(path))
                    print("Cost:", cost)
                else:
                    print("\nNo path found!")
            else:
                print("Invalid algorithm choice.")

        elif choice == "3":
            pathfinder.visualize_graph()

        elif choice == "4":
            print("\nExiting program...")
            break

        else:
            print("\nInvalid choice! Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()