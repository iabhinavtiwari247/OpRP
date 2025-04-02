import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Optional
import io
import base64

def plot_route(coordinates: List[Tuple[float, float]], 
               route: List[int], 
               title: str = "TSP Route", 
               node_labels: Optional[Dict[int, str]] = None) -> str:
    """
    Plot a TSP route given coordinates and the route order.
    
    Args:
        coordinates: List of (x, y) coordinates for each location
        route: List of indices representing the route order
        title: Title of the plot
        node_labels: Dictionary mapping node indices to labels
        
    Returns:
        base64 encoded string of the plot image
    """
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Extract x and y coordinates
    x = [coordinates[i][0] for i in range(len(coordinates))]
    y = [coordinates[i][1] for i in range(len(coordinates))]
    
    # Plot all nodes
    plt.scatter(x, y, c='blue', s=100, zorder=2)
    
    # Add node labels if provided
    if node_labels:
        for i, (x_i, y_i) in enumerate(coordinates):
            label = node_labels.get(i, str(i))
            plt.annotate(label, (x_i, y_i), xytext=(5, 5), textcoords='offset points')
    else:
        for i, (x_i, y_i) in enumerate(coordinates):
            plt.annotate(str(i), (x_i, y_i), xytext=(5, 5), textcoords='offset points')
    
    # Connect the route
    route_x = [coordinates[route[i]][0] for i in range(len(route))]
    route_y = [coordinates[route[i]][1] for i in range(len(route))]
    
    # Close the loop (return to start)
    route_x.append(route_x[0])
    route_y.append(route_y[0])
    
    # Plot the route
    plt.plot(route_x, route_y, 'r-', zorder=1)
    
    # Add arrows to show direction
    for i in range(len(route)):
        next_i = (i + 1) % len(route)
        src_x, src_y = coordinates[route[i]]
        dst_x, dst_y = coordinates[route[next_i]]
        dx = dst_x - src_x
        dy = dst_y - src_y
        # Add arrow at the midpoint
        plt.arrow(src_x + dx*0.4, src_y + dy*0.4, dx*0.2, dy*0.2, 
                  head_width=0.03*max(abs(max(x)-min(x)), abs(max(y)-min(y))),
                  head_length=0.05*max(abs(max(x)-min(x)), abs(max(y)-min(y))), 
                  fc='black', ec='black', zorder=3)
    
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Encode the plot as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def plot_multiple_routes(coordinates: List[Tuple[float, float]], 
                         routes: List[List[int]], 
                         labels: List[str], 
                         title: str = "TSP Route Comparison") -> str:
    """
    Plot multiple TSP routes for comparison.
    
    Args:
        coordinates: List of (x, y) coordinates for each location
        routes: List of routes, where each route is a list of indices
        labels: Labels for each route (e.g., algorithm names)
        title: Title of the plot
        
    Returns:
        base64 encoded string of the plot image
    """
    # Determine the subplot grid dimensions
    n = len(routes)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    # Create a figure
    plt.figure(figsize=(5*cols, 4*rows))
    
    for i, (route, label) in enumerate(zip(routes, labels)):
        plt.subplot(rows, cols, i+1)
        
        # Extract x and y coordinates
        x = [coordinates[j][0] for j in range(len(coordinates))]
        y = [coordinates[j][1] for j in range(len(coordinates))]
        
        # Plot all nodes
        plt.scatter(x, y, c='blue', s=50, zorder=2)
        
        # Add node indices as labels
        for j, (x_j, y_j) in enumerate(coordinates):
            plt.annotate(str(j), (x_j, y_j), xytext=(3, 3), textcoords='offset points', fontsize=8)
        
        # Connect the route
        route_x = [coordinates[route[j]][0] for j in range(len(route))]
        route_y = [coordinates[route[j]][1] for j in range(len(route))]
        
        # Close the loop (return to start)
        route_x.append(route_x[0])
        route_y.append(route_y[0])
        
        # Plot the route
        plt.plot(route_x, route_y, 'r-', zorder=1)
        
        plt.title(f"{label}")
        plt.xlabel("X")
        plt.ylabel("Y")
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1.05)
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Encode the plot as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str

def plot_performance_comparison(algorithm_names: List[str], 
                               execution_times: List[float], 
                               route_distances: List[float]) -> str:
    """
    Create a performance comparison chart between different TSP algorithms.
    
    Args:
        algorithm_names: Names of the algorithms
        execution_times: Execution time for each algorithm
        route_distances: Route distance/cost for each algorithm
        
    Returns:
        base64 encoded string of the plot image
    """
    n = len(algorithm_names)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot execution times
    bars1 = ax1.bar(range(n), execution_times, color='skyblue')
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(algorithm_names, rotation=45, ha='right')
    
    # Add values on top of the bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(execution_times),
                f'{execution_times[i]:.4f}s',
                ha='center', va='bottom', rotation=0)
    
    # Plot route distances
    bars2 = ax2.bar(range(n), route_distances, color='lightgreen')
    ax2.set_xlabel('Algorithm')
    ax2.set_ylabel('Route Distance')
    ax2.set_title('Route Distance Comparison')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(algorithm_names, rotation=45, ha='right')
    
    # Add values on top of the bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02*max(route_distances),
                f'{route_distances[i]:.2f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.suptitle('Performance Comparison of TSP Algorithms', fontsize=16, y=1.05)
    
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    
    # Encode the plot as base64
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_str 