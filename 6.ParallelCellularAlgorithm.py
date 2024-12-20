import numpy as np

# Problem Parameters
num_servers = 10  # Number of servers (cells)
num_tasks = 100  # Total number of tasks to distribute
max_iterations = 50  # Maximum number of iterations
neighborhood_size = 2  # Each server interacts with its 2 nearest neighbors


print("Name:Bharath C")
print("USN:1BM22CS068")
# Initialize server loads randomly
server_loads = np.random.randint(1, 10, size=num_servers)

# Total tasks distributed initially
total_load = sum(server_loads)

# Ensure all tasks are distributed
while total_load < num_tasks:
    idx = np.random.randint(0, num_servers)
    server_loads[idx] += 1
    total_load += 1

print("Initial server loads:", server_loads)

# Fitness Function: Measures load imbalance
def fitness(loads):
    return np.max(loads) - np.min(loads)  # Goal: Minimize this difference

# Cellular Update Rule
def update_loads(server_loads, neighborhood_size):
    new_loads = server_loads.copy()
    for i in range(len(server_loads)):
        # Determine neighbors (circular grid)
        neighbors = [
            server_loads[(i + offset) % len(server_loads)]
            for offset in range(-neighborhood_size, neighborhood_size + 1)
            if offset != 0
        ]
        
        # Compute average load of neighbors
        avg_neighbor_load = sum(neighbors) // len(neighbors)
        
        # Adjust current server's load towards neighbor's average
        if server_loads[i] > avg_neighbor_load:
            new_loads[i] -= 1
        elif server_loads[i] < avg_neighbor_load:
            new_loads[i] += 1
        
        # Ensure no negative loads
        new_loads[i] = max(0, new_loads[i])
    
    return new_loads

# Parallel Cellular Algorithm Execution
best_solution = server_loads
best_fitness = fitness(server_loads)

for iteration in range(max_iterations):
    # Update server loads
    server_loads = update_loads(server_loads, neighborhood_size)
    
    # Evaluate fitness
    current_fitness = fitness(server_loads)
    
    # Track the best solution
    if current_fitness < best_fitness:
        best_solution = server_loads
        best_fitness = current_fitness
    
    print(f"Iteration {iteration+1}: Server Loads = {server_loads}, Fitness = {current_fitness}")

print("\nFinal optimized server loads:", best_solution)
print("Final load imbalance (fitness):", best_fitness)