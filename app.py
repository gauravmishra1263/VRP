from flask import Flask, render_template, request
import pandas as pd
import random

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    origin = request.form['origin']
    cities_str = request.form['cities']
    distance_matrix_file = request.files['distance_matrix']

    cities = [origin] + [city.strip() for city in cities_str.split(',')]

    # Read distance matrix from CSV file
    distance_matrix = pd.read_csv(distance_matrix_file, index_col=0).to_dict(orient='index')

    shortest_path, shortest_distance = genetic_algorithm(cities, origin, distance_matrix)
    return render_template('index.html', result=(shortest_path, shortest_distance))

def generate_chromosome(cities):
    return random.sample(cities, len(cities))

def fitness(chromosome, distance_matrix):
    total_distance = sum(distance_matrix[chromosome[i]][chromosome[i + 1]] for i in range(len(chromosome) - 1))
    return 1 / total_distance if total_distance != 0 else 0

def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    selected_parents = []
    for _ in range(len(population) // 2):
        random_value = random.uniform(0, total_fitness)
        current_fitness = 0
        for i, individual in enumerate(population):
            current_fitness += fitness_values[i]
            if current_fitness >= random_value:
                selected_parents.append(individual)
                break
    return selected_parents

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    offspring1 = parent1[:crossover_point] + [city for city in parent2 if city not in parent1[:crossover_point]]
    offspring2 = parent2[:crossover_point] + [city for city in parent1 if city not in parent2[:crossover_point]]
    return offspring1, offspring2

def mutation(chromosome):
    i, j = random.sample(range(len(chromosome)), 2)
    chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
    return chromosome

def get_shortest_path(chromosome):
    return chromosome[1:]

def initialize_population(cities, population_size):
    return [generate_chromosome(cities) for _ in range(population_size)]

def genetic_algorithm(cities, origin, distance_matrix, population_size=100, max_iterations=100, mutation_rate=0.05, crossover_rate=0.7):
    population = initialize_population(cities, population_size)

    for _ in range(max_iterations):
        fitness_values = [fitness(chromosome, distance_matrix) for chromosome in population]
        parents = selection(population, fitness_values)

        offspring = []
        for i in range(0, len(parents), 2):
            offspring1, offspring2 = crossover(parents[i], parents[i + 1])
            offspring.append(mutation(offspring1))
            offspring.append(mutation(offspring2))

        # Select new individuals for the population
        population = offspring + [generate_chromosome(cities) for _ in range(population_size - len(offspring))]

    best_chromosome = max(population, key=lambda x: fitness(x, distance_matrix))
    shortest_path = get_shortest_path(best_chromosome)
    shortest_distance = sum(distance_matrix[origin][shortest_path[0]] + distance_matrix[shortest_path[i]][shortest_path[i + 1]] for i in range(len(shortest_path) - 1))

    return shortest_path, shortest_distance

if __name__ == '__main__':
    app.run(debug=True)
