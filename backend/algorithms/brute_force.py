import numpy as np
import time
from itertools import permutations
from typing import List, Tuple, Dict
import sys
sys.path.append("..")
from utils.graph import Graph

class BruteForce:
    """
    Brute Force algorithm for solving the Traveling Salesman Problem.
    Examines all possible permutations of vertices to find the optimal route.
    Time Complexity: O(n!)
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.best_route = []
        self.best_distance = float('inf')
        self.execution_time = 0
    
    def solve(self, graph: Graph, start_vertex: int = 0) -> Dict:
        """
        Solve the TSP using brute force approach.
        
        Args:
            graph: Graph object representing the problem
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        start_time = time.time()
        
        n = graph.num_vertices
        
        if n > 12:
            raise ValueError("Brute force approach is not feasible for graphs with more than 12 vertices due to factorial complexity!")
        
        vertices = list(range(n))
        
        # If start_vertex is specified, remove it from the list and handle it separately
        if start_vertex != 0:
            vertices.remove(start_vertex)
            vertices.insert(0, start_vertex)

        # Generate all possible permutations of vertices excluding the start vertex
        all_permutations = permutations(vertices[1:])
        
        self.best_distance = float('inf')
        self.best_route = []
        
        # Evaluate each permutation
        for perm in all_permutations:
            # Complete route: start -> permutation -> start
            current_route = [vertices[0]] + list(perm) + [vertices[0]]
            
            # Calculate the total distance
            total_distance = 0
            for i in range(len(current_route) - 1):
                u, v = current_route[i], current_route[i + 1]
                edge_weight = graph.get_weight(u, v)
                
                if edge_weight == float('inf'):
                    # Edge doesn't exist
                    total_distance = float('inf')
                    break
                
                total_distance += edge_weight
            
            # Update best route if current is better
            if total_distance < self.best_distance:
                self.best_distance = total_distance
                self.best_route = current_route[:-1]  # Remove the last vertex (back to start) for consistency
        
        self.execution_time = time.time() - start_time
        
        return {
            "route": self.best_route,
            "distance": self.best_distance,
            "execution_time": self.execution_time
        }
    
    def solve_from_matrix(self, distance_matrix: np.ndarray, start_vertex: int = 0) -> Dict:
        """
        Solve TSP using brute force from a distance matrix.
        
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
        Solve TSP using brute force from a set of coordinates.
        
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
