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
   git clone https://github.com/iabhinavtiwari247/optimized-route-planner.git
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
   git clone https://github.com/iabhinavtiwari247/optimized-route-planner.git
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

## Deployment

### Docker Deployment (Recommended)

This project includes Docker and Docker Compose configurations for easy deployment.

1. Make sure you have Docker and Docker Compose installed on your system
   ```
   docker --version
   docker-compose --version
   ```

2. Copy the example environment file and configure your environment variables
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file to add your Google Maps API key and any other configuration

3. Build and start the containers
   ```
   docker-compose up -d --build
   ```

4. Access the application at `http://localhost`

5. Monitor the container logs
   ```
   docker-compose logs -f
   ```

6. To stop the containers
   ```
   docker-compose down
   ```

### Deployment on Cloud Platforms

#### Heroku Deployment

This project includes configuration for deploying to Heroku with Docker support.

##### Option 1: Using the Deployment Script (Recommended)

1. Make sure you have the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed
2. Run the deployment script:
   ```
   scripts\deploy-heroku.bat
   ```
3. Follow the prompts to enter your app name and Google Maps API key

##### Option 2: Manual Deployment with Docker

1. Login to Heroku and the Container Registry:
   ```
   heroku login
   heroku container:login
   ```

2. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```

3. Set your Google Maps API key:
   ```
   heroku config:set GOOGLE_MAPS_API_KEY=your_api_key_here --app your-app-name
   ```

4. Build and push the container:
   ```
   heroku container:push web --app your-app-name
   ```

5. Release the container:
   ```
   heroku container:release web --app your-app-name
   ```

6. Open your app:
   ```
   heroku open --app your-app-name
   ```

##### Option 3: Manual Deployment with Git

1. Login to Heroku:
   ```
   heroku login
   ```

2. Create a new Heroku app:
   ```
   heroku create your-app-name
   ```

3. Set your Google Maps API key:
   ```
   heroku config:set GOOGLE_MAPS_API_KEY=your_api_key_here --app your-app-name
   ```

4. Deploy your app:
   ```
   git add .
   git commit -m "Heroku deployment"
   git push heroku master
   ```

5. Open your app:
   ```
   heroku open --app your-app-name
   ```

#### AWS Deployment

1. Create an ECR repository for your Docker images
2. Build and push your images to ECR
3. Deploy using ECS or Kubernetes (EKS)
4. Configure an Application Load Balancer

#### Google Cloud Platform

1. Push images to Google Container Registry
2. Deploy using Google Kubernetes Engine (GKE)
3. Configure external load balancing

#### Microsoft Azure

1. Push images to Azure Container Registry
2. Deploy using Azure Kubernetes Service (AKS)
3. Configure Azure Load Balancer

#### Render Deployment

This project includes configuration for deploying to Render.com, a modern cloud platform.

##### Option 1: Using the Deployment Script (Recommended)

1. Make sure you have a [Render account](https://render.com) and Git installed
2. Run the deployment script:
   ```
   # On Windows
   scripts\deploy-render.bat
   
   # On Unix/Linux/Mac
   bash scripts/deploy-render.sh
   ```
3. Follow the prompts to initialize Git if needed and open the Render dashboard
4. Connect your repository and deploy the blueprint

##### Option 2: Manual Deployment

1. Push your code to GitHub or GitLab
2. Log in to [Render Dashboard](https://dashboard.render.com)
3. Click "New" and select "Blueprint"
4. Connect your repository with the code
5. Configure your services based on the `render.yaml` file
6. Set your Google Maps API key as an environment variable
7. Apply the changes and wait for deployment to complete

Your application will be available at:
- Frontend: https://tsp-frontend.onrender.com
- Backend API: https://tsp-backend.onrender.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Special thanks to all contributors
- Inspired by the traveling salesman problem in combinatorial optimization

# Docker Deployment Steps

1. **Build the Docker images**:
   ```
   # Build backend image
   docker build -t tsp-backend ./backend
   
   # Build frontend image
   docker build -t tsp-frontend ./frontend
   ```

2. **Create a Docker network**:
   ```
   docker network create tsp-network
   ```

3. **Run the containers**:
   ```
   # Run backend container
   docker run -d --name tsp-backend \
     --network tsp-network \
     -p 5000:5000 \
     -e GOOGLE_MAPS_API_KEY=your_api_key_here \
     tsp-backend
   
   # Run frontend container
   docker run -d --name tsp-frontend \
     --network tsp-network \
     -p 80:80 \
     tsp-frontend
   ```

4. **Alternatively, use Docker Compose (recommended)**:
   ```
   # Set your Google Maps API key in .env file first
   docker-compose up -d
   ```

5. **Access the application**:
   - Frontend: http://localhost
   - Backend API: http://localhost:5000/health

6. **Monitor the containers**:
   ```
   # View logs
   docker-compose logs -f
   
   # Check status
   docker-compose ps
   ```

7. **Stop the application**:
   ```
   docker-compose down
   ```
