import numpy as np
import time
import random
from typing import List, Tuple, Dict, Callable
import sys
sys.path.append("..")
from utils.graph import Graph

class SimulatedAnnealing:
    """
    Simulated Annealing algorithm for the Traveling Salesman Problem.
    A metaheuristic approach that mimics the physical process of annealing.
    Time Complexity: O(n² * iterations)
    Space Complexity: O(n)
    """
    
    def __init__(self, 
                 initial_temp: float = 1000.0, 
                 cooling_rate: float = 0.995, 
                 iterations: int = 10000,
                 min_temp: float = 1e-10):
        """
        Initialize the Simulated Annealing algorithm.
        
        Args:
            initial_temp: Initial temperature
            cooling_rate: Cooling rate for temperature reduction
            iterations: Maximum number of iterations
            min_temp: Minimum temperature to stop annealing
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.iterations = iterations
        self.min_temp = min_temp
        self.best_route = []
        self.best_distance = float('inf')
        self.execution_time = 0
    
    def _calculate_route_distance(self, route: List[int], graph: Graph) -> float:
        """Calculate the total distance of a route."""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += graph.get_weight(route[i], route[i + 1])
        # Add the distance from the last vertex back to the start
        total_distance += graph.get_weight(route[-1], route[0])
        return total_distance
    
    def _get_random_neighbor(self, route: List[int]) -> List[int]:
        """
        Generate a random neighboring solution by swapping two vertices.
        Uses the 2-opt move: breaks two edges and reconnects them in the other way.
        """
        n = len(route)
        # Choose two random indices to swap (excluding the start vertex)
        i, j = sorted(random.sample(range(1, n), 2))
        # Create a new route by reversing the segment between i and j
        neighbor = route[:i] + route[i:j+1][::-1] + route[j+1:]
        return neighbor
    
    def solve(self, graph: Graph, start_vertex: int = 0) -> Dict:
        """
        Solve the TSP using Simulated Annealing.
        
        Args:
            graph: Graph object representing the problem
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        start_time = time.time()
        
        n = graph.num_vertices
        
        # Generate an initial solution (greedy nearest neighbor)
        current_vertex = start_vertex
        unvisited = set(range(n))
        unvisited.remove(current_vertex)
        
        current_route = [current_vertex]
        
        while unvisited:
            # Find the nearest unvisited vertex
            min_distance = float('inf')
            nearest_vertex = None
            
            for next_vertex in unvisited:
                distance = graph.get_weight(current_vertex, next_vertex)
                if distance < min_distance:
                    min_distance = distance
                    nearest_vertex = next_vertex
            
            if nearest_vertex is None:
                # No reachable unvisited vertex
                break
            
            # Move to the nearest vertex
            current_vertex = nearest_vertex
            current_route.append(current_vertex)
            unvisited.remove(current_vertex)
        
        current_distance = self._calculate_route_distance(current_route, graph)
        
        # Initialize best solution
        best_route = current_route.copy()
        best_distance = current_distance
        
        # Simulated annealing process
        temp = self.initial_temp
        
        for iteration in range(self.iterations):
            if temp < self.min_temp:
                break
                
            # Generate a neighboring solution
            neighbor_route = self._get_random_neighbor(current_route)
            neighbor_distance = self._calculate_route_distance(neighbor_route, graph)
            
            # Calculate the acceptance probability
            delta = neighbor_distance - current_distance
            
            # Accept if better, or accept with a probability if worse
            if delta < 0 or random.random() < np.exp(-delta / temp):
                current_route = neighbor_route
                current_distance = neighbor_distance
                
                # Update best solution if current is better
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
            
            # Cool down the temperature
            temp *= self.cooling_rate
        
        self.best_route = best_route
        self.best_distance = best_distance
        self.execution_time = time.time() - start_time
        
        return {
            "route": self.best_route,
            "distance": self.best_distance,
            "execution_time": self.execution_time
        }
    
    def solve_from_matrix(self, distance_matrix: np.ndarray, start_vertex: int = 0) -> Dict:
        """
        Solve TSP using Simulated Annealing from a distance matrix.
        
        Args:
            distance_matrix: Square matrix of distances between vertices
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        # Create a graph from the distance matrix
        graph = Graph(num_vertices=distance_matrix.shape[0])
        graph.from_complete_distance_matrix(distance_matrix)
        
        return self.solve(graph, start_vertex)
    
    def solve_from_coordinates(self, coordinates: List[Tuple[float, float]], 
                              start_vertex: int = 0, 
                              metric: str = 'euclidean') -> Dict:
        """
        Solve TSP using Simulated Annealing from a set of coordinates.
        
        Args:
            coordinates: List of (x, y) coordinates
            start_vertex: Starting vertex for the tour
            metric: Distance metric ('euclidean' or 'manhattan')
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        # Create a graph from the coordinates
        graph = Graph(num_vertices=len(coordinates))
        graph.from_coordinates(coordinates, metric)
        
        return self.solve(graph, start_vertex)


class GeneticAlgorithm:
    """
    Genetic Algorithm for the Traveling Salesman Problem.
    A metaheuristic approach inspired by natural selection.
    Uses evolution-inspired operators like crossover and mutation.
    Time Complexity: O(n * population_size * generations)
    Space Complexity: O(population_size * n)
    """
    
    def __init__(self, 
                 population_size: int = 100,
                 generations: int = 500,
                 mutation_rate: float = 0.01,
                 elite_size: int = 20):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            population_size: Size of the population
            generations: Number of generations
            mutation_rate: Probability of mutation
            elite_size: Number of elite individuals to pass to next generation
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_route = []
        self.best_distance = float('inf')
        self.execution_time = 0
    
    def _calculate_route_distance(self, route: List[int], graph: Graph) -> float:
        """Calculate the total distance of a route."""
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += graph.get_weight(route[i], route[i + 1])
        # Add the distance from the last vertex back to the start
        total_distance += graph.get_weight(route[-1], route[0])
        return total_distance
    
    def _create_initial_population(self, n: int, start_vertex: int) -> List[List[int]]:
        """
        Create an initial population of random routes.
        
        Args:
            n: Number of vertices
            start_vertex: Starting vertex for each route
            
        Returns:
            List of routes
        """
        population = []
        
        for _ in range(self.population_size):
            # Create a random route starting with start_vertex
            other_vertices = list(range(n))
            other_vertices.remove(start_vertex)
            random.shuffle(other_vertices)
            route = [start_vertex] + other_vertices
            population.append(route)
        
        return population
    
    def _rank_routes(self, population: List[List[int]], graph: Graph) -> List[Tuple[List[int], float]]:
        """
        Rank routes by their fitness (inverse of distance).
        
        Args:
            population: Population of routes
            graph: Graph representing the problem
            
        Returns:
            List of (route, fitness) tuples sorted by fitness
        """
        fitness_results = []
        
        for route in population:
            distance = self._calculate_route_distance(route, graph)
            # Use inverse distance as fitness (higher is better)
            fitness = 1 / distance if distance > 0 else float('inf')
            fitness_results.append((route, fitness))
        
        # Sort by fitness in descending order
        fitness_results.sort(key=lambda x: x[1], reverse=True)
        
        return fitness_results
    
    def _selection(self, ranked_population: List[Tuple[List[int], float]]) -> List[List[int]]:
        """
        Select routes for the mating pool using fitness-proportionate selection.
        
        Args:
            ranked_population: List of (route, fitness) tuples
            
        Returns:
            List of selected routes
        """
        # Directly select the elite routes
        selection_results = [route for route, _ in ranked_population[:self.elite_size]]
        
        # Calculate cumulative fitness
        fitness_sum = sum(fitness for _, fitness in ranked_population)
        probabilities = [fitness/fitness_sum for _, fitness in ranked_population]
        cumulative_probabilities = np.cumsum(probabilities)
        
        # Perform roulette wheel selection for the rest
        while len(selection_results) < self.population_size:
            rand = random.random()
            for i, prob in enumerate(cumulative_probabilities):
                if rand <= prob:
                    route = ranked_population[i][0]
                    if route not in selection_results:  # Avoid duplicates
                        selection_results.append(route)
                    break
        
        return selection_results
    
    def _ordered_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Perform ordered crossover between two parent routes.
        Preserves the relative order of elements from both parents.
        
        Args:
            parent1: First parent route
            parent2: Second parent route
            
        Returns:
            Offspring route
        """
        # Ensure parents are valid routes
        if not parent1 or not parent2 or len(parent1) != len(parent2):
            raise ValueError("Parents must be valid routes of the same length")
        
        # Get the start and end indices for the crossover segment
        n = len(parent1)
        start_idx = random.randint(0, n-2)
        end_idx = random.randint(start_idx+1, n-1)
        
        # Create the child route
        child = [-1] * n
        
        # Copy the segment from parent1
        for i in range(start_idx, end_idx+1):
            child[i] = parent1[i]
        
        # Fill the rest of the child in the order they appear in parent2
        j = 0
        for i in range(n):
            if j == start_idx:
                j = end_idx + 1
            
            if child[j] == -1:
                for gene in parent2:
                    if gene not in child:
                        child[j] = gene
                        break
                j += 1
        
        return child
    
    def _mutate(self, route: List[int]) -> List[int]:
        """
        Mutate a route by swapping two vertices (excluding the start vertex).
        
        Args:
            route: Route to mutate
            
        Returns:
            Mutated route
        """
        # Create a copy of the route
        mutated_route = route.copy()
        
        # Only mutate with probability mutation_rate
        for i in range(1, len(route)):
            if random.random() < self.mutation_rate:
                # Choose another vertex to swap with (excluding the start vertex)
                j = random.randint(1, len(route) - 1)
                mutated_route[i], mutated_route[j] = mutated_route[j], mutated_route[i]
        
        return mutated_route
    
    def _breed_population(self, mating_pool: List[List[int]]) -> List[List[int]]:
        """
        Breed the next generation from the mating pool.
        
        Args:
            mating_pool: List of routes selected for breeding
            
        Returns:
            List of routes for the next generation
        """
        # The elite routes pass directly to the next generation
        children = mating_pool[:self.elite_size].copy()
        
        # Calculate how many children we need to produce
        pool_size = len(mating_pool)
        num_children = self.population_size - self.elite_size
        
        # Breed to fill the rest of the population
        for i in range(num_children):
            # Select two random parents
            parent1_idx = random.randint(0, pool_size - 1)
            parent2_idx = random.randint(0, pool_size - 1)
            
            # Ensure parents are different
            while parent1_idx == parent2_idx:
                parent2_idx = random.randint(0, pool_size - 1)
            
            parent1 = mating_pool[parent1_idx]
            parent2 = mating_pool[parent2_idx]
            
            # Create and mutate the child
            child = self._ordered_crossover(parent1, parent2)
            mutated_child = self._mutate(child)
            
            children.append(mutated_child)
        
        return children
    
    def solve(self, graph: Graph, start_vertex: int = 0) -> Dict:
        """
        Solve the TSP using Genetic Algorithm.
        
        Args:
            graph: Graph object representing the problem
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        start_time = time.time()
        
        n = graph.num_vertices
        
        # Create initial population
        population = self._create_initial_population(n, start_vertex)
        
        # Initial ranking
        ranked_population = self._rank_routes(population, graph)
        best_route, best_fitness = ranked_population[0]
        best_distance = 1 / best_fitness if best_fitness > 0 else float('inf')
        
        # Evolution process
        for generation in range(self.generations):
            # Select routes for mating
            mating_pool = self._selection(ranked_population)
            
            # Breed the next generation
            population = self._breed_population(mating_pool)
            
            # Rank the new population
            ranked_population = self._rank_routes(population, graph)
            
            # Check if we have a new best route
            current_best = ranked_population[0]
            current_distance = 1 / current_best[1] if current_best[1] > 0 else float('inf')
            
            if current_distance < best_distance:
                best_route = current_best[0]
                best_distance = current_distance
        
        self.best_route = best_route
        self.best_distance = best_distance
        self.execution_time = time.time() - start_time
        
        return {
            "route": self.best_route,
            "distance": self.best_distance,
            "execution_time": self.execution_time
        }
    
    def solve_from_matrix(self, distance_matrix: np.ndarray, start_vertex: int = 0) -> Dict:
        """
        Solve TSP using Genetic Algorithm from a distance matrix.
        
        Args:
            distance_matrix: Square matrix of distances between vertices
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        # Create a graph from the distance matrix
        graph = Graph(num_vertices=distance_matrix.shape[0])
        graph.from_complete_distance_matrix(distance_matrix)
        
        return self.solve(graph, start_vertex)
    
    def solve_from_coordinates(self, coordinates: List[Tuple[float, float]], 
                              start_vertex: int = 0, 
                              metric: str = 'euclidean') -> Dict:
        """
        Solve TSP using Genetic Algorithm from a set of coordinates.
        
        Args:
            coordinates: List of (x, y) coordinates
            start_vertex: Starting vertex for the tour
            metric: Distance metric ('euclidean', 'manhattan', or 'haversine')
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        # Create a graph from the coordinates
        graph = Graph(num_vertices=len(coordinates))
        graph.from_coordinates(coordinates, metric)
        
        return self.solve(graph, start_vertex)
