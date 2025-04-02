import numpy as np
import time
import networkx as nx
from typing import List, Tuple, Dict
import sys
sys.path.append("..")
from utils.graph import Graph

class NearestNeighbor:
    """
    Nearest Neighbor approximation algorithm for the Traveling Salesman Problem.
    A greedy approach that always visits the nearest unvisited vertex.
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.best_route = []
        self.best_distance = float('inf')
        self.execution_time = 0
    
    def solve(self, graph: Graph, start_vertex: int = 0) -> Dict:
        """
        Solve the TSP using the Nearest Neighbor approach.
        
        Args:
            graph: Graph object representing the problem
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        start_time = time.time()
        
        n = graph.num_vertices
        
        # Initialize variables
        current_vertex = start_vertex
        unvisited = set(range(n))
        unvisited.remove(current_vertex)
        
        route = [current_vertex]
        total_distance = 0
        
        # While there are unvisited vertices
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
            route.append(current_vertex)
            unvisited.remove(current_vertex)
            total_distance += min_distance
        
        # Complete the tour by returning to the start
        total_distance += graph.get_weight(route[-1], start_vertex)
        
        self.best_route = route
        self.best_distance = total_distance
        self.execution_time = time.time() - start_time
        
        return {
            "route": self.best_route,
            "distance": self.best_distance,
            "execution_time": self.execution_time
        }
    
    def solve_from_matrix(self, distance_matrix: np.ndarray, start_vertex: int = 0) -> Dict:
        """
        Solve TSP using Nearest Neighbor from a distance matrix.
        
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
        Solve TSP using Nearest Neighbor from a set of coordinates.
        
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


class MST2Approximation:
    """
    Minimum Spanning Tree 2-approximation algorithm for the Traveling Salesman Problem.
    Uses a minimum spanning tree to find a route with a cost at most twice the optimal.
    Time Complexity: O(n² log n)
    Space Complexity: O(n)
    """
    
    def __init__(self):
        self.best_route = []
        self.best_distance = float('inf')
        self.execution_time = 0
    
    def solve(self, graph: Graph, start_vertex: int = 0) -> Dict:
        """
        Solve the TSP using the MST 2-approximation approach.
        
        Args:
            graph: Graph object representing the problem
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        start_time = time.time()
        
        # Convert the graph to a NetworkX graph for MST computation
        nx_graph = graph.to_networkx()
        
        # Compute the minimum spanning tree
        mst = nx.minimum_spanning_tree(nx_graph)
        
        # Perform a pre-order traversal of the MST
        route = list(nx.dfs_preorder_nodes(mst, source=start_vertex))
        
        # Calculate the total distance of the tour
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += graph.get_weight(route[i], route[i + 1])
        # Add the distance from the last vertex back to the start
        total_distance += graph.get_weight(route[-1], route[0])
        
        self.best_route = route
        self.best_distance = total_distance
        self.execution_time = time.time() - start_time
        
        return {
            "route": self.best_route,
            "distance": self.best_distance,
            "execution_time": self.execution_time
        }
    
    def solve_from_matrix(self, distance_matrix: np.ndarray, start_vertex: int = 0) -> Dict:
        """
        Solve TSP using MST 2-approximation from a distance matrix.
        
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
        Solve TSP using MST 2-approximation from a set of coordinates.
        
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


class Christofides:
    """
    Christofides algorithm for the Traveling Salesman Problem.
    A 3/2-approximation algorithm for metric TSP instances.
    Time Complexity: O(n³)
    Space Complexity: O(n²)
    """
    
    def __init__(self):
        self.best_route = []
        self.best_distance = float('inf')
        self.execution_time = 0
    
    def solve(self, graph: Graph, start_vertex: int = 0) -> Dict:
        """
        Solve the TSP using the Christofides algorithm.
        
        Args:
            graph: Graph object representing the problem
            start_vertex: Starting vertex for the tour
            
        Returns:
            Dict containing the best route, distance, and execution time
        """
        start_time = time.time()
        
        # Convert the graph to a NetworkX graph
        nx_graph = graph.to_networkx()
        
        # 1. Compute the minimum spanning tree
        mst = nx.minimum_spanning_tree(nx_graph)
        
        # 2. Find vertices with odd degree
        odd_vertices = [v for v, degree in mst.degree() if degree % 2 == 1]
        
        # 3. Find minimum weight perfect matching on odd vertices
        # Create a subgraph with only the odd vertices
        odd_subgraph = nx.subgraph(nx_graph, odd_vertices)
        
        # Invert edge weights for minimum weight matching
        for u, v, d in odd_subgraph.edges(data=True):
            d['weight'] = -d['weight']
        
        # Find the matching
        matching = nx.algorithms.matching.max_weight_matching(odd_subgraph, maxcardinality=True)
        
        # 4. Combine the MST and the matching
        eulerian_graph = nx.MultiGraph(mst)
        for u, v in matching:
            eulerian_graph.add_edge(u, v, weight=nx_graph[u][v]['weight'])
        
        # 5. Find an Eulerian tour
        eulerian_circuit = list(nx.eulerian_circuit(eulerian_graph, source=start_vertex))
        
        # 6. Shortcut the Eulerian tour to get a Hamiltonian cycle
        visited = set()
        route = []
        
        for u, v in eulerian_circuit:
            if u not in visited:
                route.append(u)
                visited.add(u)
        
        # Ensure the route starts at the specified vertex
        if route and route[0] != start_vertex:
            start_idx = route.index(start_vertex)
            route = route[start_idx:] + route[:start_idx]
        
        # Calculate the total distance
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += graph.get_weight(route[i], route[i + 1])
        # Add the distance from the last vertex back to the start
        total_distance += graph.get_weight(route[-1], route[0])
        
        self.best_route = route
        self.best_distance = total_distance
        self.execution_time = time.time() - start_time
        
        return {
            "route": self.best_route,
            "distance": self.best_distance,
            "execution_time": self.execution_time
        }
    
    def solve_from_matrix(self, distance_matrix: np.ndarray, start_vertex: int = 0) -> Dict:
        """
        Solve TSP using Christofides algorithm from a distance matrix.
        
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
        Solve TSP using Christofides algorithm from a set of coordinates.
        
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
