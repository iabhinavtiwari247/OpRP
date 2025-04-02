# Optimized Route Planner Using Travelling Salesman Problem (TSP)

An advanced route optimization system that calculates the shortest possible route for visiting multiple destinations and returning to the starting point. This project implements several algorithms to solve the Travelling Salesman Problem (TSP) and provides performance comparisons between them.

## Features

- **Multiple TSP Algorithms**:
  - Brute Force (for small instances)
  - Dynamic Programming (Held-Karp)
  - Nearest Neighbor Approximation
  - Minimum Spanning Tree (MST) 2-Approximation
  - Christofides Algorithm (3/2-Approximation)
  - Simulated Annealing (Metaheuristic)
  - Genetic Algorithm (Metaheuristic)

- **Graph Visualization**: Visual representation of routes and algorithm performance
- **Algorithm Comparison**: Analyze execution time and solution quality across algorithms
- **Distance Metrics**: Support for Euclidean and Manhattan distances
- **Google Maps Integration**: Convert addresses to coordinates and get detailed directions
- **Performance Optimization**: Scalable to handle larger numbers of locations

## Project Structure

```
/
├── backend/                    # Python backend
│   ├── algorithms/             # TSP algorithm implementations
│   │   ├── brute_force.py      # Exhaustive search (O(n!))
│   │   ├── dynamic_programming.py # Held-Karp algorithm (O(n²2ⁿ))
│   │   ├── approximation.py    # Approximation algorithms
│   │   └── metaheuristic.py    # Simulated annealing & genetic algorithms
│   ├── utils/                  # Utility modules
│   │   ├── graph.py            # Graph representation
│   │   └── visualization.py    # Route visualization tools
│   ├── main.py                 # Flask API endpoints
│   └── requirements.txt        # Python dependencies
│
└── frontend/                   # React frontend
    ├── src/
    │   ├── components/         # React components
    │   │   ├── RouteMap.js     # Route visualization component
    │   │   ├── RouteMap.css
    │   │   ├── ControlPanel.js # User control interface 
    │   │   └── ControlPanel.css
    │   ├── App.js              # Main app component
    │   └── App.css             # Global styles
    └── public/                 # Static assets
```

## Getting Started

### Prerequisites

- Python 3.8+ with pip
- Node.js and npm

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/optimized-route-planner.git
   cd optimized-route-planner
   ```

2. Set up the backend
   ```
   cd backend
   pip install -r requirements.txt
   ```

3. Set up the frontend
   ```
   cd frontend
   npm install
   ```

4. Create a `.env` file in the backend directory with your Google Maps API key:
   ```
   GOOGLE_MAPS_API_KEY=your-api-key-here
   ```

### Running the Application

1. Start the backend server
   ```
   cd backend
   python main.py
   ```

2. Start the frontend development server
   ```
   cd frontend
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## API Endpoints

- `GET /health` - Health check endpoint
- `POST /api/solve` - Solve TSP with a specific algorithm
- `POST /api/compare` - Compare multiple TSP algorithms
- `POST /api/geocode` - Convert an address to coordinates
- `POST /api/directions` - Get detailed directions between locations

## Algorithm Performance

| Algorithm | Time Complexity | Space Complexity | Optimality | Best For |
|-----------|----------------|------------------|------------|----------|
| Brute Force | O(n!) | O(n) | Optimal | < 10 locations |
| Dynamic Programming | O(n²2ⁿ) | O(n2ⁿ) | Optimal | < 20 locations |
| Nearest Neighbor | O(n²) | O(n) | Approximate | Any size, quick solutions |
| MST 2-Approximation | O(n² log n) | O(n) | ≤ 2 * optimal | Medium-sized problems |
| Christofides | O(n³) | O(n²) | ≤ 1.5 * optimal | Metric TSPs |
| Simulated Annealing | O(n² * iterations) | O(n) | Near-optimal | Large problems |
| Genetic Algorithm | O(population * generations * n) | O(population * n) | Near-optimal | Large problems |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to all contributors
- Inspired by the traveling salesman problem in combinatorial optimization