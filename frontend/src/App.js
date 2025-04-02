import React, { useState, useEffect } from 'react';
import RouteMap from './components/RouteMap';
import GoogleMapView from './components/GoogleMapView';
import ControlPanel from './components/ControlPanel';
import './App.css';

function App() {
  const [locations, setLocations] = useState([]);
  const [routeResult, setRouteResult] = useState(null);
  const [comparisonResult, setComparisonResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [algorithm, setAlgorithm] = useState('nearest_neighbor');
  const [distanceMetric, setDistanceMetric] = useState('euclidean');
  const [startLocationIndex, setStartLocationIndex] = useState(0);
  const [showComparison, setShowComparison] = useState(false);
  const [useGoogleMaps, setUseGoogleMaps] = useState(true);
  const [showTraffic, setShowTraffic] = useState(true);
  const [showWeather, setShowWeather] = useState(true);
  
  // Google Maps API key from environment variable
  const googleMapsApiKey = process.env.REACT_APP_GOOGLE_MAPS_API_KEY;
  
  // Load Google Maps API script
  useEffect(() => {
    if (!window.google && googleMapsApiKey && useGoogleMaps) {
      const script = document.createElement('script');
      script.src = `https://maps.googleapis.com/maps/api/js?key=${googleMapsApiKey}&libraries=places`;
      script.async = true;
      script.defer = true;
      script.onload = () => console.log('Google Maps API loaded');
      script.onerror = () => {
        console.error('Google Maps API failed to load');
        setUseGoogleMaps(false);
      };
      document.head.appendChild(script);
      
      return () => {
        // Clean up script if component unmounts during loading
        if (document.head.contains(script)) {
          document.head.removeChild(script);
        }
      };
    }
  }, [googleMapsApiKey, useGoogleMaps]);

  // Reset the route when locations change
  useEffect(() => {
    setRouteResult(null);
    setComparisonResult(null);
    setShowComparison(false);
  }, [locations]);

  // Geocode an address using Google Maps API
  const geocodeAddress = async (address) => {
    if (!window.google || !address) return null;
    
    return new Promise((resolve, reject) => {
      const geocoder = new window.google.maps.Geocoder();
      geocoder.geocode({ address }, (results, status) => {
        if (status === 'OK' && results[0]) {
          const location = results[0].geometry.location;
          resolve({
            lat: location.lat(),
            lng: location.lng(),
            formatted_address: results[0].formatted_address
          });
        } else {
          reject(new Error(`Geocoding failed: ${status}`));
        }
      });
    });
  };

  const handleAddLocation = async (name, address) => {
    try {
      let locationData;
      
      if (window.google && address && useGoogleMaps) {
        // Use Google Maps Geocoding API if available
        locationData = await geocodeAddress(address);
      } else {
        // Fallback to random coordinates for demo purposes
        locationData = {
          lat: Math.random() * 10 + 30, // Random lat between 30-40
          lng: Math.random() * 10 - 120, // Random lng between -120 and -110
          formatted_address: address || 'No address provided'
        };
      }
      
      const newLocation = {
        name,
        lat: locationData.lat,
        lng: locationData.lng,
        address: locationData.formatted_address
      };
      
      setLocations([...locations, newLocation]);
      return true;
    } catch (err) {
      console.error('Error adding location:', err);
      setError(`Failed to add location: ${err.message}`);
      return false;
    }
  };

  const handleSolve = async (selectedAlgorithm, selectedMetric, startLocation) => {
    setIsLoading(true);
    setError(null);
    setShowComparison(false);
    
    try {
      // Prepare the request data
      const requestData = {
        algorithm: selectedAlgorithm,
        locations: locations,
        start_location_index: startLocation,
        distance_metric: selectedMetric
      };
      
      // In a real app, send to backend API
      // For demo purposes, we'll simulate the API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Simulate a route result
      const simulatedResult = {
        route: locations.map((_, i) => (i + startLocation) % locations.length),
        route_names: locations.map(loc => loc.name),
        distance: Math.random() * 1000 + 500,
        execution_time: Math.random() * 2,
        route_visualization: 'R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==' // Dummy base64 image
      };
      
      setRouteResult(simulatedResult);
    } catch (err) {
      setError('Failed to solve the route. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCompare = async (algorithms, selectedMetric, startLocation) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Prepare the request data
      const requestData = {
        algorithms: algorithms,
        locations: locations,
        start_location_index: startLocation,
        distance_metric: selectedMetric
      };
      
      // In a real app, send to backend API
      // For demo purposes, we'll simulate the API call
      await new Promise(resolve => setTimeout(resolve, 2500));
      
      // Simulate comparison results
      const simulatedResults = algorithms.map(alg => ({
        algorithm: alg,
        route: locations.map((_, i) => (i + startLocation) % locations.length),
        distance: Math.random() * 1000 + 500,
        execution_time: Math.random() * 2
      }));
      
      const simulatedComparison = {
        results: simulatedResults,
        route_comparison: 'R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==', // Dummy base64 image
        performance_comparison: 'R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==' // Dummy base64 image
      };
      
      setComparisonResult(simulatedComparison);
      setShowComparison(true);
    } catch (err) {
      setError('Failed to compare algorithms. Please try again.');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>Optimized Route Planner</h1>
        <p>Using the Traveling Salesman Problem (TSP) Algorithms with Real-time Traffic & Weather</p>
      </header>
      
      <main className="app-content">
        <div className="content-grid">
          <div className="control-section">
            <ControlPanel
              locations={locations}
              setLocations={setLocations}
              onSolve={handleSolve}
              onCompare={handleCompare}
              isLoading={isLoading}
              algorithm={algorithm}
              setAlgorithm={setAlgorithm}
              distanceMetric={distanceMetric}
              setDistanceMetric={setDistanceMetric}
              startLocationIndex={startLocationIndex}
              setStartLocationIndex={setStartLocationIndex}
              onAddLocation={handleAddLocation}
              useGoogleMaps={useGoogleMaps}
              setUseGoogleMaps={setUseGoogleMaps}
              showTraffic={showTraffic}
              setShowTraffic={setShowTraffic}
              showWeather={showWeather}
              setShowWeather={setShowWeather}
            />
          </div>
          
          <div className="visualization-section">
            {useGoogleMaps && googleMapsApiKey && routeResult ? (
              <GoogleMapView
                locations={locations}
                routeData={routeResult}
                apiKey={googleMapsApiKey}
                showTraffic={showTraffic}
                showWeather={showWeather}
              />
            ) : (
              <RouteMap
                routeImage={routeResult?.route_visualization}
                isLoading={isLoading}
                error={error}
              />
            )}
            
            {routeResult && !showComparison && (
              <div className="result-details">
                <h2>Route Details</h2>
                <div className="result-stats">
                  <div className="stat-item">
                    <strong>Total Distance:</strong> {routeResult.distance.toFixed(2)} units
                  </div>
                  <div className="stat-item">
                    <strong>Execution Time:</strong> {routeResult.execution_time.toFixed(4)} seconds
                  </div>
                </div>
                
                <h3>Optimal Route Order:</h3>
                <ol className="route-order">
                  {routeResult.route_names.map((name, i) => (
                    <li key={i}>{name}</li>
                  ))}
                  <li>{routeResult.route_names[0]} (Return to Start)</li>
                </ol>
              </div>
            )}
            
            {showComparison && comparisonResult && (
              <div className="comparison-results">
                <h2>Algorithm Comparison</h2>
                
                <div className="comparison-visualizations">
                  <div className="visualization-card">
                    <h3>Route Comparison</h3>
                    <img 
                      src={`data:image/png;base64,${comparisonResult.route_comparison}`} 
                      alt="Route Comparison"
                      className="comparison-image"
                    />
                  </div>
                  
                  <div className="visualization-card">
                    <h3>Performance Comparison</h3>
                    <img 
                      src={`data:image/png;base64,${comparisonResult.performance_comparison}`} 
                      alt="Performance Comparison"
                      className="comparison-image"
                    />
                  </div>
                </div>
                
                <div className="comparison-table">
                  <h3>Results Summary</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Algorithm</th>
                        <th>Distance</th>
                        <th>Execution Time</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparisonResult.results.map((result, i) => (
                        <tr key={i}>
                          <td>{result.algorithm.replace('_', ' ').toUpperCase()}</td>
                          <td>{result.distance.toFixed(2)} units</td>
                          <td>{result.execution_time.toFixed(4)} seconds</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
      
      <footer className="app-footer">
        <p>&copy; 2023 Optimized Route Planner. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App; 