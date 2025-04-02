import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests
import time

# Import algorithm modules
from algorithms.brute_force import BruteForce
from algorithms.dynamic_programming import DynamicProgramming
from algorithms.approximation import NearestNeighbor, MST2Approximation, Christofides
from algorithms.metaheuristic import SimulatedAnnealing, GeneticAlgorithm
from algorithms.two_opt import TwoOpt

# Import utility modules
from utils.graph import Graph
from utils.visualization import plot_route, plot_multiple_routes, plot_performance_comparison

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Constants
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')

# Dictionary of available algorithms
ALGORITHMS = {
    'brute_force': BruteForce,
    'dynamic_programming': DynamicProgramming,
    'nearest_neighbor': NearestNeighbor,
    'mst_approximation': MST2Approximation,
    'christofides': Christofides,
    'simulated_annealing': SimulatedAnnealing,
    'genetic_algorithm': GeneticAlgorithm,
    # Add the additional algorithms from the frontend
    'two_opt': TwoOpt,  # New implementation
    'genetic': GeneticAlgorithm,  # Alias for genetic_algorithm
}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the API is running."""
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/solve', methods=['POST'])
def solve_tsp():
    """
    Solve the TSP problem using the specified algorithm.
    
    Expected JSON input:
    {
        "algorithm": "nearest_neighbor", // One of the available algorithms
        "locations": [
            {"lat": 37.7749, "lng": -122.4194, "name": "San Francisco"},
            {"lat": 34.0522, "lng": -118.2437, "name": "Los Angeles"}
            // ... more locations
        ],
        "start_location_index": 0,  // Optional, defaults to 0
        "distance_metric": "euclidean"  // Optional, defaults to 'euclidean'
    }
    
    Returns:
    {
        "route": [0, 2, 1, 3, ...],  // Indices of locations in optimal order
        "route_names": ["San Francisco", "Sacramento", ...],  // Names if provided
        "distance": 1234.56,  // Total route distance
        "execution_time": 0.123,  // Time taken to solve (seconds)
        "route_visualization": "base64_encoded_image"  // Base64 encoded image
    }
    """
    try:
        # Parse input data
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        algorithm_name = data.get('algorithm', 'nearest_neighbor')
        locations = data.get('locations', [])
        start_location_index = data.get('start_location_index', 0)
        distance_metric = data.get('distance_metric', 'euclidean')
        
        if not locations:
            return jsonify({'error': 'No locations provided'}), 400
        
        if algorithm_name not in ALGORITHMS:
            return jsonify({'error': f'Algorithm {algorithm_name} not supported'}), 400
        
        # Extract coordinates and names
        coordinates = [(loc['lat'], loc['lng']) for loc in locations]
        location_names = [loc.get('name', f'Location {i}') for i, loc in enumerate(locations)]
        
        # Create the algorithm instance
        algorithm = ALGORITHMS[algorithm_name]()
        
        # Solve the TSP
        result = algorithm.solve_from_coordinates(
            coordinates, 
            start_vertex=start_location_index,
            metric=distance_metric
        )
        
        # Generate route visualization
        route_visualization = plot_route(
            coordinates,
            result['route'],
            title=f'TSP Route using {algorithm_name.replace("_", " ").title()}',
            node_labels={i: name for i, name in enumerate(location_names)}
        )
        
        # Get the route names in order
        route_names = [location_names[i] for i in result['route']]
        
        # Prepare the response
        response = {
            'route': result['route'],
            'route_names': route_names,
            'distance': result['distance'],
            'execution_time': result['execution_time'],
            'route_visualization': route_visualization
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """
    Compare multiple TSP algorithms on the same problem.
    
    Expected JSON input:
    {
        "algorithms": ["nearest_neighbor", "genetic_algorithm"],
        "locations": [
            {"lat": 37.7749, "lng": -122.4194, "name": "San Francisco"},
            {"lat": 34.0522, "lng": -118.2437, "name": "Los Angeles"}
            // ... more locations
        ],
        "start_location_index": 0,  // Optional, defaults to 0
        "distance_metric": "euclidean"  // Optional, defaults to 'euclidean'
    }
    
    Returns:
    {
        "results": [
            {
                "algorithm": "nearest_neighbor",
                "route": [0, 2, 1, 3, ...],
                "distance": 1234.56,
                "execution_time": 0.123
            },
            // ... more algorithm results
        ],
        "comparison_visualization": "base64_encoded_image"
    }
    """
    try:
        # Parse input data
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        algorithm_names = data.get('algorithms', ['nearest_neighbor', 'genetic_algorithm'])
        locations = data.get('locations', [])
        start_location_index = data.get('start_location_index', 0)
        distance_metric = data.get('distance_metric', 'euclidean')
        
        if not locations:
            return jsonify({'error': 'No locations provided'}), 400
        
        # Validate algorithms
        for alg_name in algorithm_names:
            if alg_name not in ALGORITHMS:
                return jsonify({'error': f'Algorithm {alg_name} not supported'}), 400
        
        # Extract coordinates
        coordinates = [(loc['lat'], loc['lng']) for loc in locations]
        
        # Run each algorithm and collect results
        results = []
        routes = []
        algorithm_labels = []
        execution_times = []
        route_distances = []
        
        for alg_name in algorithm_names:
            # Create the algorithm instance
            algorithm = ALGORITHMS[alg_name]()
            
            # Solve the TSP
            result = algorithm.solve_from_coordinates(
                coordinates, 
                start_vertex=start_location_index,
                metric=distance_metric
            )
            
            # Store results
            results.append({
                'algorithm': alg_name,
                'route': result['route'],
                'distance': result['distance'],
                'execution_time': result['execution_time']
            })
            
            routes.append(result['route'])
            algorithm_labels.append(alg_name.replace('_', ' ').title())
            execution_times.append(result['execution_time'])
            route_distances.append(result['distance'])
        
        # Generate comparison visualizations
        route_comparison = plot_multiple_routes(
            coordinates,
            routes,
            algorithm_labels,
            title='Route Comparison of TSP Algorithms'
        )
        
        performance_comparison = plot_performance_comparison(
            algorithm_labels,
            execution_times,
            route_distances
        )
        
        # Prepare the response
        response = {
            'results': results,
            'route_comparison': route_comparison,
            'performance_comparison': performance_comparison
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/geocode', methods=['POST'])
def geocode_address():
    """
    Convert a physical address to latitude and longitude coordinates.
    
    Expected JSON input:
    {
        "address": "1600 Amphitheatre Parkway, Mountain View, CA"
    }
    
    Returns:
    {
        "lat": 37.4223878,
        "lng": -122.0841877,
        "formatted_address": "1600 Amphitheatre Pkwy, Mountain View, CA 94043, USA"
    }
    """
    try:
        # Parse input data
        data = request.get_json()
        
        if not data or 'address' not in data:
            return jsonify({'error': 'No address provided'}), 400
        
        address = data['address']
        
        # Check if Google Maps API key is configured
        if not GOOGLE_MAPS_API_KEY:
            return jsonify({'error': 'Google Maps API key not configured'}), 500
        
        # Make request to Google Maps Geocoding API
        response = requests.get(
            'https://maps.googleapis.com/maps/api/geocode/json',
            params={
                'address': address,
                'key': GOOGLE_MAPS_API_KEY
            }
        )
        
        response_data = response.json()
        
        if response_data['status'] != 'OK':
            return jsonify({'error': f'Geocoding failed: {response_data["status"]}'}), 400
        
        # Extract the location data
        location = response_data['results'][0]['geometry']['location']
        formatted_address = response_data['results'][0]['formatted_address']
        
        return jsonify({
            'lat': location['lat'],
            'lng': location['lng'],
            'formatted_address': formatted_address
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/directions', methods=['POST'])
def get_directions():
    """
    Get detailed directions between ordered locations.
    
    Expected JSON input:
    {
        "route": [0, 2, 1, 3],
        "locations": [
            {"lat": 37.7749, "lng": -122.4194},
            {"lat": 34.0522, "lng": -118.2437},
            {"lat": 32.7157, "lng": -117.1611},
            {"lat": 36.1699, "lng": -115.1398}
        ],
        "mode": "driving"  // Optional, defaults to 'driving'
    }
    
    Returns:
    {
        "routes": [
            {
                "origin": {"lat": 37.7749, "lng": -122.4194},
                "destination": {"lat": 32.7157, "lng": -117.1611},
                "distance": "500 mi",
                "duration": "8 hours 30 mins",
                "steps": [
                    // Directions from Google Maps
                ]
            },
            // ... more route segments
        ],
        "total_distance": "1200 mi",
        "total_duration": "20 hours 15 mins"
    }
    """
    try:
        # Parse input data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        route = data.get('route', [])
        locations = data.get('locations', [])
        mode = data.get('mode', 'driving')
        
        if not route or not locations:
            return jsonify({'error': 'Route or locations not provided'}), 400
        
        # Check if Google Maps API key is configured
        if not GOOGLE_MAPS_API_KEY:
            return jsonify({'error': 'Google Maps API key not configured'}), 500
        
        # Get directions for each segment of the route
        routes = []
        total_distance_meters = 0
        total_duration_seconds = 0
        
        # Close the loop by returning to the starting point
        full_route = route + [route[0]]
        
        for i in range(len(full_route) - 1):
            origin_idx = full_route[i]
            dest_idx = full_route[i + 1]
            
            # Check if indices are within bounds
            if origin_idx < 0 or origin_idx >= len(locations) or dest_idx < 0 or dest_idx >= len(locations):
                return jsonify({'error': f'Invalid location index in route: {origin_idx} or {dest_idx}'}), 400
            
            origin = locations[origin_idx]
            destination = locations[dest_idx]
            
            # Make request to Google Maps Directions API
            response = requests.get(
                'https://maps.googleapis.com/maps/api/directions/json',
                params={
                    'origin': f"{origin['lat']},{origin['lng']}",
                    'destination': f"{destination['lat']},{destination['lng']}",
                    'mode': mode,
                    'key': GOOGLE_MAPS_API_KEY
                }
            )
            
            response_data = response.json()
            
            if response_data['status'] != 'OK':
                return jsonify({'error': f'Directions failed: {response_data["status"]}'}), 400
            
            # Extract the route data
            route_data = response_data['routes'][0]['legs'][0]
            distance_text = route_data['distance']['text']
            duration_text = route_data['duration']['text']
            steps = route_data['steps']
            
            # Add to totals
            total_distance_meters += route_data['distance']['value']
            total_duration_seconds += route_data['duration']['value']
            
            # Simplify steps to reduce response size
            simplified_steps = []
            for step in steps:
                simplified_steps.append({
                    'instruction': step['html_instructions'],
                    'distance': step['distance']['text'],
                    'duration': step['duration']['text']
                })
            
            routes.append({
                'origin': origin,
                'destination': destination,
                'distance': distance_text,
                'duration': duration_text,
                'steps': simplified_steps
            })
        
        # Convert total distance and duration to human-readable format
        total_distance = f"{total_distance_meters / 1000:.2f} km"
        
        hours = total_duration_seconds // 3600
        minutes = (total_duration_seconds % 3600) // 60
        total_duration = f"{hours} hours {minutes} mins"
        
        return jsonify({
            'routes': routes,
            'total_distance': total_distance,
            'total_duration': total_duration
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
