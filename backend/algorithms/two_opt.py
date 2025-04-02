import numpy as np
import time
from typing import List, Tuple, Dict
import sys
sys.path.append("..")
from utils.graph import Graph

class TwoOpt:
    """
    Two-Opt optimization algorithm for the Traveling Salesman Problem.
    A local search algorithm that improves a route by swapping edges.
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    
    def __init__(self, max_iterations: int = 1000):
        """
        Initialize the Two-Opt algorithm.
        
        Args:
            max_iterations: Maximum number of iterations to perform
        """
        self.max_iterations = max_iterations
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
    
    def _two_opt_swap(self, route: List[int], i: int, j: int) -> List[int]:
        """
        Perform a 2-opt swap by reversing the segment between positions i and j.
        
        Args:
            route: Current route
            i: First position to swap
            j: Second position to swap
            
        Returns:
            New route after swap
        """
        # Create a new route with the segment between i and j reversed
        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
        return new_route
    
    def solve(self, graph: Graph, start_vertex: int = 0) -> Dict:
        """
        Solve the TSP using the Two-Opt algorithm.
        
        Args:
            graph: Graph object representing the problem
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        start_time = time.time()
        
        n = graph.num_vertices
        
        # Generate an initial solution (nearest neighbor)
        current_vertex = start_vertex
        unvisited = set(range(n))
        unvisited.remove(current_vertex)
        
        route = [current_vertex]
        
        # Create initial route using nearest neighbor
        while unvisited:
            min_distance = float('inf')
            nearest_vertex = None
            
            for v in unvisited:
                distance = graph.get_weight(current_vertex, v)
                if distance < min_distance:
                    min_distance = distance
                    nearest_vertex = v
            
            current_vertex = nearest_vertex
            route.append(current_vertex)
            unvisited.remove(current_vertex)
        
        # Apply 2-opt improvement
        improved = True
        iteration = 0
        current_distance = self._calculate_route_distance(route, graph)
        
        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1
            
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Skip adjacent edges and cases where endpoints are 0
                    if j - i == 1 or (i == 0 and j == n - 1):
                        continue
                    
                    # Calculate the change in distance if we perform this swap
                    # Current edges: (i-1, i) and (j, j+1)
                    # New edges: (i-1, j) and (i, j+1)
                    # Note: j+1 might overflow, so we use modulo
                    current_edges_dist = graph.get_weight(route[i-1], route[i]) + graph.get_weight(route[j], route[(j+1) % n])
                    new_edges_dist = graph.get_weight(route[i-1], route[j]) + graph.get_weight(route[i], route[(j+1) % n])
                    
                    if new_edges_dist < current_edges_dist:
                        # Perform the swap if it improves the route
                        route = self._two_opt_swap(route, i, j)
                        current_distance = self._calculate_route_distance(route, graph)
                        improved = True
                        break
                
                if improved:
                    break
        
        self.best_route = route
        self.best_distance = current_distance
        self.execution_time = time.time() - start_time
        
        return {
            "route": self.best_route,
            "distance": self.best_distance,
            "execution_time": self.execution_time
        }
    
    def solve_from_matrix(self, distance_matrix: np.ndarray, start_vertex: int = 0) -> Dict:
        """
        Solve TSP using Two-Opt from a distance matrix.
        
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
        Solve TSP using Two-Opt from a set of coordinates.
        
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