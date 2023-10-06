import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Define simulation parameters
num_users = 500
num_base_stations = 100
num_iterations = 400
population_size = 600
energy_threshold = 200

# Generate synthetic data for energy consumption and traffic load
max_energy = 500
max_traffic = num_users * 4

energy_consumption = np.random.uniform(0.1, max_energy, num_base_stations)
traffic_load = np.random.randint(1, max_traffic, num_users)

# Define optimization objectives (fitness functions)
def energy_objective(solution):
    solution_indices = solution.astype(int) - 1
    total_energy = np.sum(energy_consumption[solution_indices])
    energy_weight = 0.8
    energy_constraint = 500
    energy_penalty = max(0, total_energy - energy_constraint)
    return energy_weight * total_energy + (1 - energy_weight) * traffic_objective(solution) + energy_penalty

def traffic_objective(solution):
    solution_indices = solution.astype(int) - 1
    total_traffic = np.sum(traffic_load[solution_indices])
    return total_traffic


# Define Genetic Algorithm
def genetic_algorithm():
    # Initialize population with random solutions
    population = [np.random.randint(1, num_base_stations + 1, num_users) for i in range(population_size)]
    
    for i in range(num_iterations):
        # Evaluate fitness of each solution based on objectives
        fitness_values = [energy_objective(solution) + traffic_objective(solution) for solution in population]
        
        # Select parents based on fitness (roulette wheel selection)
        probabilities = fitness_values / np.sum(fitness_values)
        parent_indices = np.random.choice(range(population_size), size=population_size, p=probabilities)
        parents = [population[i] for i in parent_indices]
        
        # Apply crossover and mutation to create new generation
        new_population = []
        for i in range(population_size):
            parent1, parent2 = np.random.choice(parent_indices, size=2, replace=False)
            parent1 = parents[parent1]
            parent2 = parents[parent2]
            crossover_point = np.random.randint(1, num_users)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            mutation_rate = 0.5
            for i in range(num_users):
                if np.random.rand() < mutation_rate:
                    child[i] = np.random.randint(1, num_base_stations + 1)
            new_population.append(child)
        
        # Update population with new generation
        population = new_population
        
    # Return best solution
    best_solution_index = np.argmin(fitness_values)
    return population[best_solution_index]


def particle_swarm_optimization(initial_solution):
    # Initialize particles with the initial solution
    particles = [{"position": np.copy(initial_solution),
                  "velocity": np.zeros(num_users),
                  "best_position": np.copy(initial_solution),
                  "best_fitness": energy_objective(initial_solution) + traffic_objective(initial_solution)} for i in range(population_size)]

    global_best_position = np.copy(initial_solution)
    global_best_fitness = energy_objective(initial_solution) + traffic_objective(initial_solution)
    
    for i in range(num_iterations):
        for particle in particles:
            # Update particle velocity and position
            inertia_weight = 0.8
            cognitive_weight = 1.0
            social_weight = 1.0
            if particle["best_position"] is not None:
                velocity_update = (cognitive_weight * np.random.rand() * (particle["best_position"] - particle["position"]) +
                                  social_weight * np.random.rand() * (global_best_position - particle["position"]))
                particle["velocity"] = inertia_weight * particle["velocity"] + velocity_update
                # Clip particle positions to valid range
                particle["position"] = np.clip(particle["position"] + particle["velocity"], 1, num_base_stations)
            particle_fitness = energy_objective(particle["position"]) + traffic_objective(particle["position"])
            
            # Update particle's best position and fitness
            if particle_fitness < particle["best_fitness"]:
                particle["best_fitness"] = particle_fitness
                particle["best_position"] = particle["position"]
                
                # Update global best position and fitness
                if particle_fitness < global_best_fitness:
                    global_best_fitness = particle_fitness
                    global_best_position = particle["position"]
                
    # Return global best solution
    return global_best_position

# After running the optimizations, create a list indicating whether each channel is active or shut down
def channel_optimization(solution, energy_threshold):
    active_channels = [energy <= energy_threshold for energy in energy_consumption]
    return active_channels

# Run Genetic Algorithm and Particle Swarm Optimization
ga_solution = genetic_algorithm()
pso_solution = particle_swarm_optimization(ga_solution)

# Perform channel optimization
ga_channel_optimization = channel_optimization(ga_solution, energy_threshold)
pso_channel_optimization = channel_optimization(pso_solution, energy_threshold)

# Create plots
plt.figure(figsize=(10, 6))  # Adjust the figsize as needed

# Energy Consumption Bar Plot
plt.subplot(1, 1, 1)
bars = plt.bar(range(1, num_base_stations + 1), energy_consumption, color=['red' if energy > energy_threshold else 'blue' for energy in energy_consumption])

plt.xlabel("Base Station")
plt.ylabel("Energy Consumption (Joules)")
plt.title("Energy Consumption per Base Station")

# Traffic Load Histogram
plt.subplot(4, 2, 2)
plt.hist(traffic_load, bins=20, histtype='barstacked', color='blue', alpha=0.7)

plt.xlabel("Traffic Load (Kbps)")
plt.ylabel("Frequency")
plt.title("Distribution of Traffic Load")

# Calculate best-fit lines
combined_fit = np.polyfit(combined_energy, combined_traffic, 1)

# Find the minimum energy consumption and corresponding index for both GA and PSO
ga_best_index = np.argmin(ga_energy_values)
pso_best_index = np.argmin(pso_energy_values)

# Calculate annotation offsets based on index
ga_offset = -10 * (ga_best_index % 2)  # Adjust the multiplier to fine-tune the offset
pso_offset = 10 * (pso_best_index % 2)  # Adjust the multiplier to fine-tune the offset

# Energy Consumption vs. Traffic Load with best fit line and combined best solutions
plt.subplot(4, 2, 3)

# Plot Genetic Algorithm
plt.scatter(ga_energy_values, ga_traffic_values, c='blue', alpha=0.7, label='Genetic Algorithm', marker='o')

# Plot Particle Swarm Optimization
plt.scatter(pso_energy_values, pso_traffic_values, c='orange', alpha=0.7, label='Particle Swarm Optimization', marker='x')

# Plot combined best fit line
plt.plot(combined_energy, np.polyval(combined_fit, combined_energy), 'green', linestyle='dashed', label='Combined Fit')

# Annotate the combined best solutions with adjusted offsets
plt.annotate(f'GA: {ga_energy_values[ga_best_index]:.2f} J', 
             (ga_energy_values[ga_best_index], ga_traffic_values[ga_best_index]), 
             textcoords="offset points", xytext=(0, ga_offset), ha='center', fontsize=9)

plt.annotate(f'PSO: {pso_energy_values[pso_best_index]:.2f} J', 
             (pso_energy_values[pso_best_index], pso_traffic_values[pso_best_index]), 
             textcoords="offset points", xytext=(0, pso_offset), ha='center', fontsize=9)

plt.xlabel("Energy Consumption (Joules)")
plt.ylabel("Traffic Load (Kbps)")
plt.title("Energy Consumption vs. Traffic Load")
plt.legend()
plt.grid(True)

# Channel Optimization Plot
plt.subplot(4, 2, 4)
colors = ['green' if active else 'red' for active in ga_channel_optimization]
plt.bar(range(1, num_base_stations + 1), energy_consumption, color=colors)

plt.xlabel("Base Station")
plt.ylabel("Energy Consumption (Joules)")
plt.title("Channel Optimization: Active Channels (Green) vs. Shut Down Channels (Red)")

# Display plots
plt.tight_layout()
plt.show()
