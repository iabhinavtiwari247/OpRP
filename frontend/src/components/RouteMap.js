import React from 'react';
import './RouteMap.css';

const RouteMap = ({ routeImage, isLoading, error }) => {
  return (
    <div className="route-map-container">
      <h2>Route Visualization</h2>
      
      {isLoading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Calculating optimal route...</p>
        </div>
      )}
      
      {error && (
        <div className="error-container">
          <p className="error-message">{error}</p>
        </div>
      )}
      
      {routeImage && !isLoading && !error && (
        <div className="map-container">
          <img 
            src={`data:image/png;base64,${routeImage}`} 
            alt="Optimized Route Map" 
            className="route-image"
          />
        </div>
      )}
      
      {!routeImage && !isLoading && !error && (
        <div className="empty-state">
          <p>Add locations and solve to visualize the optimal route</p>
        </div>
      )}
    </div>
  );
};

export default RouteMap; 