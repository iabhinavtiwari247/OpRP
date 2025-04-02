import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Union, Optional
import math

class Graph:
    """
    Graph class for representing TSP problems.
    Supports both adjacency matrix and adjacency list representations.
    """
    
    def __init__(self, 
                 num_vertices: int = 0, 
                 use_adjacency_list: bool = False,
                 coordinates: Optional[List[Tuple[float, float]]] = None):
        """
        Initialize a graph for the TSP problem.
        
        Args:
            num_vertices: Number of vertices/locations in the graph
            use_adjacency_list: If True, use adjacency list; otherwise use adjacency matrix
            coordinates: List of (x, y) coordinates for each location
        """
        self.num_vertices = num_vertices
        self.use_adjacency_list = use_adjacency_list
        self.coordinates = coordinates if coordinates else []
        
        # Initialize the appropriate data structure
        if use_adjacency_list:
            self.adjacency_list = [[] for _ in range(num_vertices)]
        else:
            # Use infinity for missing edges, 0 for self-loops
            self.adjacency_matrix = np.full((num_vertices, num_vertices), np.inf)
            np.fill_diagonal(self.adjacency_matrix, 0)
    
    def add_edge(self, u: int, v: int, weight: float) -> None:
        """
        Add an edge between vertices u and v with the given weight.
        
        Args:
            u: Source vertex
            v: Destination vertex
            weight: Edge weight/distance
        """
        if u < 0 or u >= self.num_vertices or v < 0 or v >= self.num_vertices:
            raise ValueError(f"Vertex indices must be between 0 and {self.num_vertices-1}")
            
        if self.use_adjacency_list:
            self.adjacency_list[u].append((v, weight))
            self.adjacency_list[v].append((u, weight))  # For undirected graph
        else:
            self.adjacency_matrix[u][v] = weight
            self.adjacency_matrix[v][u] = weight  # For undirected graph
    
    def get_weight(self, u: int, v: int) -> float:
        """Get the weight of the edge between vertices u and v."""
        if self.use_adjacency_list:
            for vertex, weight in self.adjacency_list[u]:
                if vertex == v:
                    return weight
            return float('inf')  # Edge doesn't exist
        else:
            return self.adjacency_matrix[u][v]
    
    def from_complete_distance_matrix(self, distance_matrix: np.ndarray) -> None:
        """
        Initialize the graph from a complete distance matrix.
        
        Args:
            distance_matrix: Square matrix of distances between all pairs of vertices
        """
        n = distance_matrix.shape[0]
        if distance_matrix.shape != (n, n):
            raise ValueError("Distance matrix must be square")
        
        self.num_vertices = n
        
        if self.use_adjacency_list:
            self.adjacency_list = [[] for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i != j:  # Skip self-loops
                        self.adjacency_list[i].append((j, distance_matrix[i][j]))
        else:
            self.adjacency_matrix = distance_matrix.copy()
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees) using the haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of point 1 (in degrees)
            lat2, lon2: Latitude and longitude of point 2 (in degrees)
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a)) 
        r = 6371  # Radius of Earth in kilometers
        
        return c * r
    
    def from_coordinates(self, coordinates: List[Tuple[float, float]], metric: str = 'euclidean') -> None:
        """
        Create a complete graph from a list of coordinates.
        
        Args:
            coordinates: List of (x, y) coordinates for each location
            metric: Distance metric to use ('euclidean', 'manhattan', 'haversine')
        """
        self.coordinates = coordinates
        self.num_vertices = len(coordinates)
        
        # Calculate distances between all pairs of coordinates
        if metric == 'euclidean':
            distance_fn = lambda p1, p2: np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        elif metric == 'manhattan':
            distance_fn = lambda p1, p2: abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
        elif metric == 'haversine':
            distance_fn = lambda p1, p2: self._haversine_distance(p1[0], p1[1], p2[0], p2[1])
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
        
        if self.use_adjacency_list:
            self.adjacency_list = [[] for _ in range(self.num_vertices)]
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    if i != j:  # Skip self-loops
                        distance = distance_fn(coordinates[i], coordinates[j])
                        self.adjacency_list[i].append((j, distance))
        else:
            self.adjacency_matrix = np.zeros((self.num_vertices, self.num_vertices))
            for i in range(self.num_vertices):
                for j in range(self.num_vertices):
                    if i != j:  # Skip self-loops
                        self.adjacency_matrix[i][j] = distance_fn(coordinates[i], coordinates[j])
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph for visualization and analysis."""
        G = nx.Graph()
        
        # Add nodes with position attributes if coordinates are available
        if self.coordinates:
            for i, (x, y) in enumerate(self.coordinates):
                G.add_node(i, pos=(x, y))
        else:
            G.add_nodes_from(range(self.num_vertices))
        
        # Add edges
        if self.use_adjacency_list:
            for u in range(self.num_vertices):
                for v, weight in self.adjacency_list[u]:
                    if u < v:  # Add each edge only once
                        G.add_edge(u, v, weight=weight)
        else:
            for u in range(self.num_vertices):
                for v in range(u + 1, self.num_vertices):  # Add each edge only once
                    weight = self.adjacency_matrix[u][v]
                    if weight != np.inf:
                        G.add_edge(u, v, weight=weight)
        
        return G
    
    def __str__(self) -> str:
        if self.use_adjacency_list:
            result = "Graph (Adjacency List):\n"
            for i, edges in enumerate(self.adjacency_list):
                result += f"{i}: {edges}\n"
        else:
            result = "Graph (Adjacency Matrix):\n"
            result += str(self.adjacency_matrix)
        return result 