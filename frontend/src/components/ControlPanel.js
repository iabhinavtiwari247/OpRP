import React, { useState } from 'react';
import './ControlPanel.css';

const ALGORITHMS = [
  { id: 'brute_force', name: 'Brute Force', description: 'Examines all possible permutations (best for < 10 locations)' },
  { id: 'dynamic_programming', name: 'Dynamic Programming', description: 'Uses Held-Karp algorithm (best for < 20 locations)' },
  { id: 'nearest_neighbor', name: 'Nearest Neighbor', description: 'Greedy approach that visits closest unvisited location next' },
  { id: 'mst_approximation', name: 'MST Approximation', description: '2-approximation algorithm using minimum spanning tree' },
  { id: 'christofides', name: 'Christofides', description: '3/2-approximation algorithm for metric TSP instances' },
  { id: 'simulated_annealing', name: 'Simulated Annealing', description: 'Metaheuristic inspired by annealing in metallurgy' },
  { id: 'genetic_algorithm', name: 'Genetic Algorithm', description: 'Evolutionary approach with crossover and mutation operators' }
];

const DISTANCE_METRICS = [
  { id: 'euclidean', name: 'Euclidean', description: 'Straight-line distance (as the crow flies)' },
  { id: 'manhattan', name: 'Manhattan', description: 'Sum of absolute differences (city block distance)' }
];

const ControlPanel = ({ 
  locations, 
  setLocations, 
  onSolve, 
  onCompare, 
  isLoading, 
  algorithm, 
  setAlgorithm, 
  distanceMetric, 
  setDistanceMetric,
  startLocationIndex,
  setStartLocationIndex,
  onAddLocation,
  useGoogleMaps,
  setUseGoogleMaps,
  showTraffic,
  setShowTraffic,
  showWeather,
  setShowWeather
}) => {
  const [locationName, setLocationName] = useState('');
  const [locationAddress, setLocationAddress] = useState('');
  const [selectedAlgorithms, setSelectedAlgorithms] = useState(['nearest_neighbor', 'genetic_algorithm']);
  const [showCompare, setShowCompare] = useState(false);

  const handleAddLocation = () => {
    if (locationName.trim() === '') {
      alert('Please enter a location name');
      return;
    }

    onAddLocation(locationName, locationAddress)
      .then(success => {
        if (success) {
          setLocationName('');
          setLocationAddress('');
        }
      });
  };

  const handleRemoveLocation = (index) => {
    const updatedLocations = [...locations];
    updatedLocations.splice(index, 1);
    setLocations(updatedLocations);
    
    // Update start location index if needed
    if (startLocationIndex >= index) {
      setStartLocationIndex(Math.max(0, startLocationIndex - 1));
    }
  };

  const handleSetStartLocation = (index) => {
    setStartLocationIndex(index);
  };

  const handleSolve = () => {
    if (locations.length < 3) {
      alert('Please add at least 3 locations to find an optimal route');
      return;
    }
    onSolve(algorithm, distanceMetric, startLocationIndex);
  };

  const handleCompare = () => {
    if (locations.length < 3) {
      alert('Please add at least 3 locations to compare algorithms');
      return;
    }
    if (selectedAlgorithms.length < 2) {
      alert('Please select at least 2 algorithms to compare');
      return;
    }
    onCompare(selectedAlgorithms, distanceMetric, startLocationIndex);
  };

  const toggleAlgorithmSelection = (alg) => {
    if (selectedAlgorithms.includes(alg)) {
      setSelectedAlgorithms(selectedAlgorithms.filter(a => a !== alg));
    } else {
      setSelectedAlgorithms([...selectedAlgorithms, alg]);
    }
  };

  const algorithmOptions = [
    { value: 'brute_force', label: 'Brute Force (exact)', maxLocations: 10 },
    { value: 'dynamic_programming', label: 'Dynamic Programming (exact)', maxLocations: 15 },
    { value: 'nearest_neighbor', label: 'Nearest Neighbor (approx)', maxLocations: 100 },
    { value: 'two_opt', label: '2-Opt (approx)', maxLocations: 100 },
    { value: 'genetic', label: 'Genetic Algorithm (approx)', maxLocations: 100 },
    { value: 'simulated_annealing', label: 'Simulated Annealing (approx)', maxLocations: 100 },
  ];

  return (
    <div className="control-panel">
      <h2>Route Planner</h2>
      
      <div className="section">
        <h3>Locations</h3>
        <div className="location-form">
          <div className="form-group">
            <label htmlFor="location-name">Name:</label>
            <input
              type="text"
              id="location-name"
              value={locationName}
              onChange={(e) => setLocationName(e.target.value)}
              placeholder="e.g., Home, Office, Store"
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="location-address">Address:</label>
            <input
              type="text"
              id="location-address"
              value={locationAddress}
              onChange={(e) => setLocationAddress(e.target.value)}
              placeholder="e.g., 123 Main St, City, State"
            />
          </div>
          
          <button 
            className="btn btn-primary" 
            onClick={handleAddLocation}
            disabled={isLoading}
          >
            Add Location
          </button>
        </div>
        
        <div className="locations-list">
          <h3>Added Locations ({locations.length})</h3>
          {locations.length === 0 ? (
            <div className="empty-list">No locations added yet</div>
          ) : (
            <ul>
              {locations.map((location, index) => (
                <li key={index} className={index === startLocationIndex ? 'start-location' : ''}>
                  <div className="location-item">
                    <div className="location-info">
                      <strong>{location.name}</strong>
                      <div className="location-address">{location.address}</div>
                      <div className="location-coords">
                        Lat: {location.lat.toFixed(4)}, Lng: {location.lng.toFixed(4)}
                      </div>
                    </div>
                    <div className="location-actions">
                      <button 
                        className="btn btn-small"
                        onClick={() => handleSetStartLocation(index)}
                        disabled={index === startLocationIndex}
                      >
                        Set as Start
                      </button>
                      <button 
                        className="btn btn-small btn-danger"
                        onClick={() => handleRemoveLocation(index)}
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      
      <div className="section">
        <h3>Algorithm</h3>
        
        <div className="form-group">
          <label htmlFor="algorithm">Algorithm:</label>
          <select
            id="algorithm"
            value={algorithm}
            onChange={(e) => setAlgorithm(e.target.value)}
            disabled={isLoading}
          >
            {algorithmOptions.map((option) => (
              <option 
                key={option.value} 
                value={option.value}
                disabled={locations.length > option.maxLocations}
              >
                {option.label} {locations.length > option.maxLocations ? '(too many locations)' : ''}
              </option>
            ))}
          </select>
        </div>
        
        <div className="form-group">
          <label htmlFor="distance-metric">Distance Metric:</label>
          <select
            id="distance-metric"
            value={distanceMetric}
            onChange={(e) => setDistanceMetric(e.target.value)}
            disabled={isLoading}
          >
            <option value="euclidean">Euclidean (straight line)</option>
            <option value="manhattan">Manhattan (city block)</option>
            <option value="haversine">Haversine (great circle)</option>
          </select>
        </div>
      </div>
      
      <div className="actions">
        <button 
          className="btn btn-primary btn-large" 
          onClick={handleSolve}
          disabled={isLoading || locations.length < 3}
        >
          {isLoading ? 'Calculating...' : 'Find Optimal Route'}
        </button>
        
        <button 
          className="btn btn-secondary" 
          onClick={() => setShowCompare(!showCompare)}
          disabled={isLoading || locations.length < 3}
        >
          {showCompare ? 'Hide Comparison' : 'Compare Algorithms'}
        </button>
      </div>
      
      {showCompare && (
        <div className="comparison-section">
          <h3>Compare Algorithms</h3>
          <div className="algorithms-selection">
            {algorithmOptions.map((option) => (
              <div key={option.value} className="form-group checkbox-group">
                <label>
                  <input
                    type="checkbox"
                    value={option.value}
                    checked={selectedAlgorithms.includes(option.value)}
                    onChange={() => toggleAlgorithmSelection(option.value)}
                    disabled={isLoading || locations.length > option.maxLocations}
                  />
                  {option.label}
                </label>
              </div>
            ))}
          </div>
          
          <button 
            className="btn btn-primary" 
            onClick={handleCompare}
            disabled={isLoading || locations.length < 3 || selectedAlgorithms.length < 2}
          >
            {isLoading ? 'Comparing...' : 'Compare Selected Algorithms'}
          </button>
        </div>
      )}

      <section className="visualization-section">
        <h2>Visualization Options</h2>
        <div className="form-group checkbox-group">
          <label>
            <input
              type="checkbox"
              checked={useGoogleMaps}
              onChange={(e) => setUseGoogleMaps(e.target.checked)}
              disabled={isLoading}
            />
            Use Google Maps
          </label>
        </div>
        
        {useGoogleMaps && (
          <>
            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={showTraffic}
                  onChange={(e) => setShowTraffic(e.target.checked)}
                  disabled={isLoading}
                />
                Show Traffic Information
              </label>
            </div>
            
            <div className="form-group checkbox-group">
              <label>
                <input
                  type="checkbox"
                  checked={showWeather}
                  onChange={(e) => setShowWeather(e.target.checked)}
                  disabled={isLoading}
                />
                Show Weather Information
              </label>
            </div>
          </>
        )}
      </section>
    </div>
  );
};

export default ControlPanel; 