import math
import random
import copy
import matplotlib.pyplot as plt


def generate_nodes(num_nodes):
    nodes = {}
    for i in range(num_nodes):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        nodes[i] = (x, y)
    return nodes

def create_distance_matrix(nodes):
    n = len(nodes)
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                d = math.sqrt((nodes[i][0] - nodes[j][0])**2 +
                              (nodes[i][1] - nodes[j][1])**2)
                traffic_factor = random.uniform(0.8, 1.2)
                matrix[i][j] = d * traffic_factor
            else:
                matrix[i][j] = 0
    return matrix

def generate_initial_solution(num_nodes, num_vehicles):
    deliveries = list(range(1, num_nodes))
    random.shuffle(deliveries)
    solution = [[] for _ in range(num_vehicles)]
    for i, d in enumerate(deliveries):
        solution[i % num_vehicles].append(d)
    return solution

def evaluate_solution(solution, distance_matrix):
    total_cost = 0
    for route in solution:
        full_route = [0] + route + [0]
        for i in range(len(full_route) - 1):
            total_cost += distance_matrix[full_route[i]][full_route[i+1]]
    return total_cost

def plot_routes(solution, nodes, title):
    plt.figure(figsize=(8, 6))
    
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
    
    startend = nodes[0]

    for i, route in enumerate(solution):
        route_nodes = [0] + route + [0]
        x = [nodes[node][0] for node in route_nodes]
        y = [nodes[node][1] for node in route_nodes]
        
        plt.plot(x, y, marker='o', color=colors[i % len(colors)], label=f'Vehicle {i+1}')
        for node in route_nodes:
            plt.text(nodes[node][0]+0.5, nodes[node][1]+0.5, str(node), fontsize=9, color=colors[i % len(colors)])
    
    plt.scatter(startend[0], startend[1], c='red', s=100, marker='s', label='Startend (0)')
    
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.grid(True)
    plt.show()

def get_random_neighbor(solution):
    neighbor = copy.deepcopy(solution)
    move_type = random.choice(['intra', 'inter'])
    
    if move_type == 'intra':
        valid_routes = [i for i, r in enumerate(neighbor) if len(r) >= 2]
        if valid_routes:
            route_idx = random.choice(valid_routes)
            route = neighbor[route_idx]
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
    else:
        routes_with_nodes = [i for i, r in enumerate(neighbor) if len(r) > 0]
        if len(routes_with_nodes) >= 2:
            route_idx1, route_idx2 = random.sample(routes_with_nodes, 2)
            route1 = neighbor[route_idx1]
            route2 = neighbor[route_idx2]
            i = random.randrange(len(route1))
            j = random.randrange(len(route2))
            route1[i], route2[j] = route2[j], route1[i]
    
    return neighbor

def local_search(initial_solution, distance_matrix, max_iter=1000):
    current_solution = copy.deepcopy(initial_solution)
    current_cost = evaluate_solution(current_solution, distance_matrix)
    
    for _ in range(max_iter):
        neighbor = get_random_neighbor(current_solution)
        neighbor_cost = evaluate_solution(neighbor, distance_matrix)
        if neighbor_cost < current_cost:
            current_solution = neighbor
            current_cost = neighbor_cost
    return current_solution, current_cost

def hill_climbing(initial_solution, distance_matrix, max_iter=1000):
    current_solution = copy.deepcopy(initial_solution)
    current_cost = evaluate_solution(current_solution, distance_matrix)
    
    for _ in range(max_iter):
        best_neighbor = None
        best_neighbor_cost = current_cost
        # evaluate 50 random neighbours at each iteration
        for _ in range(50):
            neighbor = get_random_neighbor(current_solution)
            neighbor_cost = evaluate_solution(neighbor, distance_matrix)
            if neighbor_cost < best_neighbor_cost:
                best_neighbor = neighbor
                best_neighbor_cost = neighbor_cost
        if best_neighbor is not None and best_neighbor_cost < current_cost:
            current_solution = best_neighbor
            current_cost = best_neighbor_cost
        else:
            # no improvement found
            break
    return current_solution, current_cost

def simulated_annealing(initial_solution, distance_matrix, initial_temp=1000, cooling_rate=0.995, max_iter=10000):
    current_solution = copy.deepcopy(initial_solution)
    current_cost = evaluate_solution(current_solution, distance_matrix)
    best_solution = current_solution
    best_cost = current_cost
    temp = initial_temp
    
    for _ in range(max_iter):
        neighbor = get_random_neighbor(current_solution)
        neighbor_cost = evaluate_solution(neighbor, distance_matrix)
        delta = neighbor_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_solution = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_solution = current_solution
                best_cost = current_cost
        
        temp *= cooling_rate
        if temp < 1e-3:
            break
    return best_solution, best_cost

def main():
    num_nodes = 21
    num_vehicles = 3
    nodes = generate_nodes(num_nodes)
    distance_matrix = create_distance_matrix(nodes)
    initial_solution = generate_initial_solution(num_nodes, num_vehicles)
    initial_cost = evaluate_solution(initial_solution, distance_matrix)
    
    print("Initial Routes:")
    for idx, route in enumerate(initial_solution):
        print(f"  Vehicle {idx+1}: {route}")
    print("Initial Total Travel Time:", initial_cost)
    
    plot_routes(initial_solution, nodes, f"Initial Routes (Cost: {initial_cost:.2f})")
    
    ls_solution, ls_cost = local_search(initial_solution, distance_matrix, max_iter=1000)
    print("\n--- Local Search ---")
    for idx, route in enumerate(ls_solution):
        print(f"  Vehicle {idx+1}: {route}")
    print("Local Search Total Travel Time:", ls_cost)
    plot_routes(ls_solution, nodes, f"Local Search Routes (Cost: {ls_cost:.2f})")

    hc_solution, hc_cost = hill_climbing(initial_solution, distance_matrix, max_iter=1000)
    print("\n--- Hill Climbing ---")
    for idx, route in enumerate(hc_solution):
        print(f"  Vehicle {idx+1}: {route}")
    print("Hill Climbing Total Travel Time:", hc_cost)
    plot_routes(hc_solution, nodes, f"Hill Climbing Routes (Cost: {hc_cost:.2f})")
    
    sa_solution, sa_cost = simulated_annealing(initial_solution, distance_matrix,
                                               initial_temp=1000, cooling_rate=0.995, max_iter=10000)
    print("\n--- Simulated Annealing ---")
    for idx, route in enumerate(sa_solution):
        print(f"  Vehicle {idx+1}: {route}")
    print("Simulated Annealing Total Travel Time:", sa_cost)
    plot_routes(sa_solution, nodes, f"Simulated Annealing Routes (Cost: {sa_cost:.2f})")

if __name__ == "__main__":
    main()
