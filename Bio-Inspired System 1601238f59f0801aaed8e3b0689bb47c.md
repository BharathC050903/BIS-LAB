# Bio-Inspired System.

<aside>
ℹ️

A **bio-inspired system** is a technological or computational system designed by emulating principles, structures, or behaviors observed in biological systems. These systems aim to mimic nature's efficiency, adaptability, and resilience to solve real-world problems.

</aside>

## 1.Genetic Algorithm

### Pseudo Code

```
Function GeneticAlgorithm:
    # Step 1: Define the Problem
    Define fitness_function(x)  # Function to evaluate the fitness of an individual
    population_size = N         # Number of individuals in the population
    generations = G             # Number of generations to run
    mutation_rate = M           # Probability of mutation
    crossover_rate = C          # Probability of crossover
    gene_length = L             # Length of each individual (solution representation)

    # Step 2: Create Initial Population
    Function initialize_population():
        population = []
        For i = 1 to population_size:
            individual = RandomBinaryString(gene_length)  # Generate random binary string
            population.append(individual)
        Return population

    # Step 3: Evaluate Fitness
    Function evaluate_population(population):
        fitness_values = []
        For each individual in population:
            fitness = fitness_function(individual)
            fitness_values.append(fitness)
        Return fitness_values

    # Step 4: Selection (Roulette Wheel Selection)
    Function select_parents(population, fitness_values):
        total_fitness = Sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        parents = RandomSelectionWithProbabilities(population, probabilities, 2)
        Return parents

    # Step 5: Crossover
    Function crossover(parent1, parent2):
        If Random(0, 1) < crossover_rate:
            point = RandomInteger(1, gene_length - 1)  # Choose crossover point
            offspring1 = parent1[:point] + parent2[point:]
            offspring2 = parent2[:point] + parent1[point:]
            Return offspring1, offspring2
        Else:
            Return parent1, parent2  # No crossover

    # Step 6: Mutation
    Function mutate(individual):
        For i = 0 to gene_length - 1:
            If Random(0, 1) < mutation_rate:
                FlipBit(individual[i])  # Flip the binary bit
        Return individual

    # Step 7: Main Loop for Generations
    population = initialize_population()
    For generation = 1 to generations:
        fitness_values = evaluate_population(population)

        # Track the best solution
        best_individual = population[MaxIndex(fitness_values)]
        best_fitness = Max(fitness_values)

        # Create new population
        new_population = []
        While Length(new_population) < population_size:
            parents = select_parents(population, fitness_values)
            offspring1, offspring2 = crossover(parents[0], parents[1])
            offspring1 = mutate(offspring1)
            offspring2 = mutate(offspring2)
            new_population.append(offspring1)
            new_population.append(offspring2)

        # Update population
        population = new_population

    # Step 8: Return the Best Solution
    Return best_individual, best_fitness

```

## 2.**Particle Swarm Optimization (PSO)**

```
Function ParticleSwarmOptimization:
    # Step 1: Initialize Parameters
    Define fitness_function(x)  # The function to optimize
    swarm_size = N              # Number of particles in the swarm
    dimensions = D              # Number of variables in the function
    max_iterations = T          # Maximum number of iterations
    inertia_weight = w          # Controls particle velocity
    cognitive_const = c1        # Personal influence constant
    social_const = c2           # Global influence constant

    # Step 2: Initialize the Swarm
    For each particle in the swarm:
        particle.position = RandomVector(dimensions)   # Random initial position
        particle.velocity = RandomVector(dimensions)   # Random initial velocity
        particle.best_position = particle.position     # Initialize personal best
        particle.best_fitness = fitness_function(particle.position)

    global_best_position = BestPositionAmongParticles(swarm)
    global_best_fitness = fitness_function(global_best_position)

    # Step 3: Iterative Optimization
    For iteration = 1 to max_iterations:
        For each particle in the swarm:
            # Evaluate fitness
            current_fitness = fitness_function(particle.position)

            # Update personal best
            If current_fitness < particle.best_fitness:
                particle.best_position = particle.position
                particle.best_fitness = current_fitness

            # Update global best
            If current_fitness < global_best_fitness:
                global_best_position = particle.position
                global_best_fitness = current_fitness

            # Update velocity
            For each dimension d in particle.velocity:
                cognitive_term = cognitive_const * Random(0, 1) * (particle.best_position[d] - particle.position[d])
                social_term = social_const * Random(0, 1) * (global_best_position[d] - particle.position[d])
                particle.velocity[d] = inertia_weight * particle.velocity[d] + cognitive_term + social_term

            # Update position
            For each dimension d in particle.position:
                particle.position[d] += particle.velocity[d]

    # Step 4: Return the Best Solution
    Return global_best_position, global_best_fitness

```

---

## 3.Ant Colony Optimisation

```
Function AntColonyOptimization:
    # Step 1: Define the Problem
    Define cities and their coordinates
    distance_matrix = ComputeDistancesBetweenCities(cities)

    # Step 2: Initialize Parameters
    num_ants = A                       # Number of ants
    alpha = 1                          # Importance of pheromone
    beta = 2                           # Importance of heuristic information
    rho = 0.5                          # Evaporation rate
    pheromone = InitializePheromoneMatrix(size=NumCities, value=InitialPheromoneValue)

    max_iterations = T                 # Maximum number of iterations
    best_solution = None
    best_cost = Infinity

    # Step 3: Iterate
    For iteration = 1 to max_iterations:
        solutions = []                  # Store solutions for this iteration

        For each ant:
            # Step 3.1: Construct a solution
            current_city = RandomStartingCity()
            solution = [current_city]

            While not all cities are visited:
                probabilities = ComputeTransitionProbabilities(current_city, pheromone, distance_matrix, alpha, beta)
                next_city = SelectNextCity(probabilities)
                solution.append(next_city)
                current_city = next_city

            solutions.append(solution)

        # Step 3.2: Evaluate Solutions
        For each solution in solutions:
            cost = ComputeSolutionCost(solution, distance_matrix)
            If cost < best_cost:
                best_solution = solution
                best_cost = cost

        # Step 3.3: Update Pheromones
        pheromone *= (1 - rho)         # Evaporate pheromones
        For each solution in solutions:
            cost = ComputeSolutionCost(solution, distance_matrix)
            UpdatePheromoneTrails(pheromone, solution, cost)

    # Step 4: Output the Best Solution
    Return best_solution, best_cost

```

---

## 4. Cuckoo Search

```
Function CuckooSearch:
    # Step 1: Define the Problem
    Define the optimization problem: minimize or maximize a function f(x)
    Initialize the search space and boundaries

    # Step 2: Initialize Parameters
    num_nests = N                        # Number of nests (potential solutions)
    pa = 0.25                            # Probability of discovering alien eggs
    max_iterations = T                   # Maximum number of iterations
    nests = RandomlyGenerateInitialSolutions(num_nests)
    best_solution = FindBestSolution(nests)

    # Step 3: Iterate
    For iteration = 1 to max_iterations:
        For each cuckoo:
            # Step 3.1: Lévy Flight to Generate New Solution
            new_solution = LevyFlight(best_solution)

            # Step 3.2: Evaluate Fitness
            random_nest = SelectRandomNest(nests)
            If Fitness(new_solution) > Fitness(random_nest):
                Replace random_nest with new_solution

        # Step 3.3: Abandon Poor Solutions
        For each nest:
            If RandomNumber() < pa:
                nest = GenerateRandomSolution()

        # Step 3.4: Update the Best Solution
        current_best = FindBestSolution(nests)
        If Fitness(current_best) > Fitness(best_solution):
            best_solution = current_best

    # Step 4: Output the Best Solution
    Return best_solution

```

---

## **5.Grey Wolf Optimizer (GWO)**

```
Function GreyWolfOptimizer:
    # Step 1: Define the Problem
    Define the optimization function f(x)

    # Step 2: Initialize Parameters
    num_wolves = N                    # Number of wolves in the pack
    num_iterations = T                # Number of iterations
    search_space = [lower_bound, upper_bound]
    wolves_positions = RandomlyGenerateInitialPositions(num_wolves, search_space)

    # Step 3: Initialize the Hierarchy
    alpha, beta, delta = None, None, None  # Best, second-best, third-best wolves

    # Step 4: Main Loop
    For iteration = 1 to num_iterations:
        # Step 4.1: Evaluate Fitness
        For each wolf:
            fitness = EvaluateFitness(wolf_position)
            Update alpha, beta, delta based on fitness

        # Step 4.2: Update Wolf Positions
        For each wolf:
            For each dimension in search_space:
                D_alpha = |C1 * alpha_position - current_position|
                D_beta = |C2 * beta_position - current_position|
                D_delta = |C3 * delta_position - current_position|

                # Update position based on alpha, beta, delta guidance
                new_position = (A1 * alpha_position + A2 * beta_position + A3 * delta_position) / 3

            wolves_positions[wolf] = new_position

        # Reduce the coefficients A and C over iterations
        UpdateCoefficients(iteration)

    # Step 5: Return the Best Solution
    Return alpha_position  # Best solution found

```

---

## 6.Parallel Cellular Algorithm

```
Function ParallelCellularAlgorithm:
    # Step 1: Define the Problem
    Define the optimization function f(x)
    
    # Step 2: Initialize Parameters
    num_cells = N                    # Number of cells in the grid
    num_iterations = T               # Number of iterations
    grid_size = [rows, columns]      # Grid dimensions
    neighborhood_structure = DefineNeighborhood(grid_size)
    cell_positions = RandomlyGenerateInitialPositions(num_cells, grid_size)
    
    # Step 3: Initialize Population
    For each cell in grid:
        fitness[cell] = EvaluateFitness(cell_position)

    # Step 4: Main Loop
    For iteration = 1 to num_iterations:
        # Step 4.1: Update States
        For each cell in grid:
            neighbors = GetNeighbors(cell, neighborhood_structure)
            new_state = UpdateCellState(cell, neighbors, fitness)
            cell_positions[cell] = new_state

        # Step 4.2: Evaluate Fitness
        For each cell in grid:
            fitness[cell] = EvaluateFitness(cell_position)

    # Step 5: Return the Best Solution
    Return BestSolution(cell_positions, fitness)

```

## 7.Gene Expression Algorithm

```
Function GeneExpressionAlgorithm:
    # Step 1: Define the Problem
    Define the optimization function f(x)
    
    # Step 2: Initialize Parameters
    population_size = P               # Number of genetic sequences
    num_generations = G               # Number of generations
    mutation_rate = m                 # Probability of mutation
    crossover_rate = c                # Probability of crossover
    population = RandomlyGenerateInitialPopulation(population_size)

    # Step 3: Evaluate Initial Fitness
    For each genetic_sequence in population:
        fitness[genetic_sequence] = EvaluateFitness(genetic_sequence)

    # Step 4: Main Loop
    For generation = 1 to num_generations:
        # Step 4.1: Perform Selection
        selected_population = SelectBasedOnFitness(population, fitness)
        
        # Step 4.2: Perform Crossover
        offspring = PerformCrossover(selected_population, crossover_rate)
        
        # Step 4.3: Perform Mutation
        mutated_offspring = PerformMutation(offspring, mutation_rate)
        
        # Step 4.4: Gene Expression
        For each genetic_sequence in mutated_offspring:
            expressed_solution = ExpressGenes(genetic_sequence)
            fitness[genetic_sequence] = EvaluateFitness(expressed_solution)
        
        # Update Population
        population = CombinePopulation(selected_population, mutated_offspring)

    # Step 5: Return the Best Solution
    Return BestSolution(population, fitness)

```

## Common Applications For 7 Algorithms

### 1. **Parameter Optimization in Machine Learning and Deep Learning Models**

- **Genetic Algorithms (GA), Particle Swarm Optimization (PSO), Ant Colony Optimization (ACO), Cuckoo Search (CS), Grey Wolf Optimizer (GWO), Parallel Cellular Algorithms, and Gene Expression Algorithms (GEA) can be used** to optimize hyperparameters, feature selection, and structure of models like neural networks, SVMs, or ensemble methods.
- **Applications**: Improving accuracy, training efficiency, and generalization in machine learning models.

### 2. **Engineering Design and Optimization**

- **Genetic Algorithms, Particle Swarm Optimization, Ant Colony Optimization, Cuckoo Search, Grey Wolf Optimizer, Parallel Cellular Algorithms, and Gene Expression Algorithms** can be employed to optimize design parameters like weight, strength, and efficiency in structures, mechanical parts, and systems.
- **Applications**: Minimizing material usage, enhancing structural performance, and reducing manufacturing costs.

### 3. **Combinatorial Optimization Problems**

- **All seven algorithms** can be used to solve combinatorial problems such as the Traveling Salesman Problem (TSP), the Knapsack Problem, and job scheduling.
- **Applications**: Routing optimization, scheduling tasks in manufacturing, and resource allocation in supply chains.

### 4. **Feature Selection and Data Mining**

- **Genetic Algorithms, Particle Swarm Optimization, Ant Colony Optimization, Cuckoo Search, Grey Wolf Optimizer, Parallel Cellular Algorithms, and Gene Expression Algorithms** are used for selecting relevant features from datasets to improve the performance of machine learning models.
- **Applications**: Reducing dimensionality, enhancing predictive accuracy, and feature subset selection for big data analysis.

### 5. **Optimization in Control Systems and Operations Research**

- **All seven algorithms** can be utilized for optimizing control parameters, system states, and decision-making processes in operations research.
- **Applications**: Designing efficient controllers for dynamic systems, optimizing resource allocation in telecommunications, and managing complex logistical operations.