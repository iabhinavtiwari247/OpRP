import React, { useRef, useEffect, useState } from 'react';
import './GoogleMapView.css';

const GoogleMapView = ({ locations, routeData, apiKey, showTraffic, showWeather }) => {
  const mapRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const directionsRendererRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const weatherApiKey = process.env.REACT_APP_OPENWEATHER_API_KEY || 'your_openweather_api_key';

  useEffect(() => {
    if (!apiKey) {
      setError('Google Maps API key is missing. Please add it to your environment variables.');
      setLoading(false);
      return;
    }

    if (!locations || locations.length < 2 || !routeData) {
      setError('Not enough locations to display a route.');
      setLoading(false);
      return;
    }

    // Initialize map if it doesn't exist yet
    if (!mapInstanceRef.current && mapRef.current) {
      // Calculate the center of all locations
      const bounds = new window.google.maps.LatLngBounds();
      locations.forEach(location => {
        bounds.extend(new window.google.maps.LatLng(location.lat, location.lng));
      });

      // Create the map instance
      mapInstanceRef.current = new window.google.maps.Map(mapRef.current, {
        zoom: 10,
        center: bounds.getCenter(),
        mapTypeId: window.google.maps.MapTypeId.ROADMAP,
        mapTypeControl: true,
        mapTypeControlOptions: {
          style: window.google.maps.MapTypeControlStyle.DROPDOWN_MENU,
          position: window.google.maps.ControlPosition.TOP_RIGHT
        },
        fullscreenControl: true,
        streetViewControl: true,
        zoomControl: true
      });

      // Create directions renderer
      directionsRendererRef.current = new window.google.maps.DirectionsRenderer({
        suppressMarkers: false,
        polylineOptions: {
          strokeColor: '#0088ff',
          strokeWeight: 6,
          strokeOpacity: 0.7
        }
      });
      directionsRendererRef.current.setMap(mapInstanceRef.current);
    }

    // Create DirectionsService
    const directionsService = new window.google.maps.DirectionsService();

    // Add traffic layer if enabled
    if (showTraffic && mapInstanceRef.current) {
      const trafficLayer = new window.google.maps.TrafficLayer();
      trafficLayer.setMap(mapInstanceRef.current);
    }

    // Prepare the waypoints from the route order
    const routeOrder = routeData.route || [];
    const waypoints = routeOrder
      .slice(1, routeOrder.length - 1) // Exclude first and last points (origin/destination)
      .map(index => ({
        location: new window.google.maps.LatLng(
          locations[index].lat,
          locations[index].lng
        ),
        stopover: true
      }));

    // Get origin and destination
    const origin = new window.google.maps.LatLng(
      locations[routeOrder[0]].lat,
      locations[routeOrder[0]].lng
    );
    
    const destination = origin; // Return to start for TSP

    // Request directions
    setLoading(true);
    directionsService.route(
      {
        origin: origin,
        destination: destination,
        waypoints: waypoints,
        optimizeWaypoints: false, // We already optimized the route
        travelMode: window.google.maps.TravelMode.DRIVING,
      },
      (result, status) => {
        if (status === 'OK') {
          directionsRendererRef.current.setDirections(result);
          // Fit the map to the bounds of the route
          const bounds = new window.google.maps.LatLngBounds();
          result.routes[0].legs.forEach(leg => {
            leg.steps.forEach(step => {
              step.path.forEach(point => {
                bounds.extend(point);
              });
            });
          });
          mapInstanceRef.current.fitBounds(bounds);
          
          // Fetch weather data for each location
          if (showWeather) {
            fetchWeatherForLocations(locations, routeOrder);
          }
          
          setLoading(false);
        } else {
          setError(`Failed to calculate directions: ${status}`);
          setLoading(false);
        }
      }
    );

    return () => {
      // Cleanup if needed
    };
  }, [locations, routeData, apiKey, showTraffic, showWeather]);

  const fetchWeatherForLocations = async (locations, routeOrder) => {
    if (!weatherApiKey || weatherApiKey === 'your_openweather_api_key') {
      console.warn('OpenWeather API key is missing. Weather data will not be displayed.');
      return;
    }

    try {
      // Create a weather marker for each location
      for (const index of routeOrder) {
        const location = locations[index];
        const weatherResponse = await fetch(
          `https://api.openweathermap.org/data/2.5/weather?lat=${location.lat}&lon=${location.lng}&units=metric&appid=${weatherApiKey}`
        );
        
        if (weatherResponse.ok) {
          const weatherData = await weatherResponse.json();
          
          // Create a marker with weather info
          const weatherIcon = weatherData.weather[0].icon;
          const iconUrl = `https://openweathermap.org/img/wn/${weatherIcon}@2x.png`;
          
          const marker = new window.google.maps.Marker({
            position: new window.google.maps.LatLng(location.lat, location.lng),
            map: mapInstanceRef.current,
            title: location.name,
            icon: {
              url: iconUrl,
              scaledSize: new window.google.maps.Size(40, 40)
            }
          });
          
          // Create an info window with weather details
          const infoWindow = new window.google.maps.InfoWindow({
            content: `
              <div class="weather-info">
                <h3>${location.name}</h3>
                <p>${location.address}</p>
                <div class="weather-data">
                  <img src="${iconUrl}" alt="${weatherData.weather[0].description}" />
                  <div>
                    <p class="weather-details"><strong>Temperature:</strong> ${weatherData.main.temp}°C</p>
                    <p class="weather-details"><strong>Condition:</strong> ${weatherData.weather[0].description}</p>
                    <p class="weather-details"><strong>Humidity:</strong> ${weatherData.main.humidity}%</p>
                    <p class="weather-details"><strong>Wind:</strong> ${weatherData.wind.speed} m/s</p>
                  </div>
                </div>
              </div>
            `
          });
          
          // Add click listener to open info window
          marker.addListener('click', () => {
            infoWindow.open(mapInstanceRef.current, marker);
          });
        }
      }
    } catch (err) {
      console.error('Error fetching weather data:', err);
    }
  };

  return (
    <div className="map-container">
      {!apiKey && (
        <div className="api-key-warning">
          Google Maps API key not found. Please add it to your environment variables as REACT_APP_GOOGLE_MAPS_API_KEY.
        </div>
      )}
      
      {loading && (
        <div className="map-loading">
          <div className="spinner"></div>
          <p>Loading route and weather data...</p>
        </div>
      )}
      
      {error && (
        <div className="map-error">{error}</div>
      )}
      
      <div ref={mapRef} className="google-map" style={{ height: '500px', width: '100%' }}></div>
      
      <div className="map-legend">
        <div className="legend-item">
          <strong>Map Legend:</strong>
        </div>
        
        {showTraffic && (
          <div className="legend-item">
            <strong>Traffic:</strong>
            <div className="traffic-indicators">
              <span style={{ backgroundColor: '#008000' }}></span> Good
              <span style={{ backgroundColor: '#FFFF00' }}></span> Moderate
              <span style={{ backgroundColor: '#FF0000' }}></span> Heavy
            </div>
          </div>
        )}
        
        {showWeather && (
          <div className="legend-item">
            <strong>Weather:</strong> Click on weather icons to see details
          </div>
        )}
      </div>
    </div>
  );
};

export default GoogleMapView; 