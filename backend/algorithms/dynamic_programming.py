import numpy as np
import time
from typing import List, Tuple, Dict
import sys
sys.path.append("..")
from utils.graph import Graph

class DynamicProgramming:
    """
    Dynamic Programming (Held-Karp) algorithm for solving the Traveling Salesman Problem.
    Uses a bottom-up approach with memoization to find the optimal route.
    Time Complexity: O(n^2 * 2^n)
    Space Complexity: O(n * 2^n)
    """
    
    def __init__(self):
        self.best_route = []
        self.best_distance = float('inf')
        self.execution_time = 0
        self.dp_table = {}  # Memoization table
        self.parent = {}    # Parent table for reconstructing the path
    
    def _get_path(self, mask: int, pos: int, n: int, start: int) -> List[int]:
        """
        Reconstruct the path from the parent table.
        
        Args:
            mask: Bit mask representing visited vertices
            pos: Current position
            n: Number of vertices
            start: Starting vertex
            
        Returns:
            Reconstructed path
        """
        path = [pos]
        
        # While there are unvisited vertices
        while mask != 0:
            pos = self.parent.get((mask, pos))
            path.append(pos)
            # Mark the vertex as unvisited in the mask
            mask &= ~(1 << pos)
        
        # Add the starting vertex at the beginning
        if path[-1] != start:
            path.append(start)
        path.reverse()
        
        return path
    
    def solve(self, graph: Graph, start_vertex: int = 0) -> Dict:
        """
        Solve the TSP using dynamic programming (Held-Karp algorithm).
        
        Args:
            graph: Graph object representing the problem
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        start_time = time.time()
        
        n = graph.num_vertices
        
        if n > 25:
            raise ValueError("Dynamic programming approach may be inefficient for graphs with more than 25 vertices!")
        
        # Initialize memoization and parent tables
        self.dp_table = {}
        self.parent = {}
        
        # Base case: When only one vertex is visited (the starting vertex)
        for i in range(n):
            if i != start_vertex:
                self.dp_table[(1 << start_vertex | 1 << i, i)] = graph.get_weight(start_vertex, i)
                self.parent[(1 << start_vertex | 1 << i, i)] = start_vertex
        
        # For each subset size
        for subset_size in range(3, n + 1):
            # For each possible subset of vertices of size subset_size
            for subset in self._generate_combinations(n, subset_size, start_vertex):
                bits = subset
                # Make sure the start vertex is included
                subset_with_start = bits | (1 << start_vertex)
                
                # For each possible last vertex in the subset
                for last in range(n):
                    if last == start_vertex or (bits & (1 << last)) == 0:
                        continue
                    
                    # Find the minimum distance to reach 'last' after visiting all vertices in the subset
                    min_distance = float('inf')
                    min_prev = -1
                    
                    # Try all possible vertices before 'last'
                    prev_subset = subset_with_start & ~(1 << last)
                    for prev in range(n):
                        if prev == start_vertex or prev == last or (prev_subset & (1 << prev)) == 0:
                            continue
                        
                        # Calculate the distance through 'prev'
                        current_distance = self.dp_table.get((prev_subset, prev), float('inf')) + graph.get_weight(prev, last)
                        
                        if current_distance < min_distance:
                            min_distance = current_distance
                            min_prev = prev
                    
                    # Update the memoization and parent tables
                    if min_distance != float('inf'):
                        self.dp_table[(subset_with_start, last)] = min_distance
                        self.parent[(subset_with_start, last)] = min_prev
        
        # Find the optimal route back to the starting vertex
        min_distance = float('inf')
        min_last = -1
        all_vertices = (1 << n) - 1
        
        for last in range(n):
            if last != start_vertex:
                current_distance = self.dp_table.get((all_vertices, last), float('inf')) + graph.get_weight(last, start_vertex)
                if current_distance < min_distance:
                    min_distance = current_distance
                    min_last = last
        
        # Reconstruct the optimal route
        self.best_route = self._get_path(all_vertices & ~(1 << start_vertex), min_last, n, start_vertex)
        self.best_distance = min_distance
        self.execution_time = time.time() - start_time
        
        return {
            "route": self.best_route,
            "distance": self.best_distance,
            "execution_time": self.execution_time
        }
    
    def _generate_combinations(self, n: int, size: int, start_vertex: int) -> List[int]:
        """
        Generate all combinations of size 'size' from n vertices, excluding the start vertex.
        
        Args:
            n: Number of vertices
            size: Size of the combination
            start_vertex: Starting vertex to exclude
            
        Returns:
            List of combinations represented as bit masks
        """
        combinations = []
        self._generate_combinations_util(0, 0, n, size, combinations, start_vertex, 0)
        return combinations
    
    def _generate_combinations_util(self, idx: int, curr_size: int, n: int, size: int, 
                                   combinations: List[int], start_vertex: int, mask: int) -> None:
        """
        Utility function to generate combinations recursively.
        
        Args:
            idx: Current index
            curr_size: Current size of the combination
            n: Number of vertices
            size: Target size of the combination
            combinations: List to store the combinations
            start_vertex: Starting vertex to exclude
            mask: Current bit mask
        """
        if curr_size == size:
            combinations.append(mask)
            return
        
        if idx == n:
            return
        
        if idx != start_vertex:
            # Include current vertex
            self._generate_combinations_util(idx + 1, curr_size + 1, n, size, 
                                           combinations, start_vertex, mask | (1 << idx))
        
        # Exclude current vertex
        self._generate_combinations_util(idx + 1, curr_size, n, size, 
                                       combinations, start_vertex, mask)
    
    def solve_from_matrix(self, distance_matrix: np.ndarray, start_vertex: int = 0) -> Dict:
        """
        Solve TSP using dynamic programming from a distance matrix.
        
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
        Solve TSP using dynamic programming from a set of coordinates.
        
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
