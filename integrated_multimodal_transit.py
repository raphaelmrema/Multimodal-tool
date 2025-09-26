# main.py
# Clean Integrated Multimodal Transit Tool for Railway Deployment

import os
import json
import logging
import requests
import datetime
import math
import random
import tempfile
import zipfile
import io
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Auto-install required packages
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
for pkg in ["polyline", "pandas", "geopandas", "shapely", "numpy"]:
    install_package(pkg)

import polyline
import pandas as pd
import numpy as np

# GeoPandas imports with fallback
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("GeoPandas not available - shapefile analysis disabled")

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
BIKE_SPEED_MPH = float(os.getenv("BIKE_SPEED_MPH", "11"))
DATA_ZIP_URL = os.getenv("DATA_ZIP_URL", "https://raw.githubusercontent.com/raphaelmrema/Multimodal-tool/main/data.zip")

# GTFS URLs
GTFS_URLS = [
    "https://ride.jtafla.com/gtfs-archive/gtfs.zip",
    "https://schedules.jtafla.com/schedulesgtfs/download",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multimodal-transit")

# =============================================================================
# GTFS Manager
# =============================================================================

class GTFSManager:
    def __init__(self):
        self.stops_df = None
        self.routes_df = None
        self.is_loaded = False
        
    def load_gtfs_data(self):
        logger.info("Loading GTFS data...")
        
        for i, url in enumerate(GTFS_URLS):
            try:
                response = requests.get(url, timeout=30, verify=False, stream=True)
                response.raise_for_status()
                
                if not response.content.startswith(b'PK'):
                    continue
                
                with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                    if 'stops.txt' in z.namelist():
                        with z.open('stops.txt') as f:
                            self.stops_df = pd.read_csv(f)
                        logger.info(f"Loaded {len(self.stops_df)} GTFS stops")
                        self.is_loaded = True
                        return True
                        
            except Exception as e:
                logger.error(f"GTFS source {i+1} failed: {e}")
                continue
        
        logger.warning("Could not load GTFS data")
        return False
    
    def find_nearby_stops(self, lat, lon, radius_km=0.5):
        if not self.is_loaded:
            return []
        
        try:
            stops = self.stops_df.copy()
            stops['distance_km'] = ((stops['stop_lat'] - lat)**2 + (stops['stop_lon'] - lon)**2)**0.5 * 111
            nearby = stops[stops['distance_km'] <= radius_km].sort_values('distance_km')
            return nearby[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'distance_km']].head(3).to_dict('records')
        except Exception as e:
            logger.error(f"Error finding stops: {e}")
            return []

# =============================================================================
# Shapefile Analyzer
# =============================================================================

class ShapefileAnalyzer:
    def __init__(self):
        self.roads_gdf = None
        self.loaded = False
        
    def load_data(self):
        if not GEOPANDAS_AVAILABLE or not DATA_ZIP_URL:
            return False
            
        try:
            logger.info(f"Downloading data from: {DATA_ZIP_URL}")
            response = requests.get(DATA_ZIP_URL, timeout=60)
            response.raise_for_status()
            
            temp_dir = tempfile.mkdtemp()
            zip_path = os.path.join(temp_dir, "data.zip")
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            extract_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_dir)
            
            # Find roads shapefile
            roads_file = self._find_shapefile(extract_dir, ["road", "bike", "street", "lane"])
            if roads_file:
                self.roads_gdf = gpd.read_file(roads_file)
                self.loaded = True
                logger.info(f"Loaded roads shapefile: {len(self.roads_gdf)} features")
                return True
                
        except Exception as e:
            logger.error(f"Shapefile loading failed: {e}")
        
        return False
    
    def _find_shapefile(self, search_dir, keywords):
        import os
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.shp'):
                    for keyword in keywords:
                        if keyword in file.lower():
                            return os.path.join(root, file)
        return None
    
    def analyze_route(self, coordinates):
        if not self.loaded:
            return 50, {"NO_DATA": {"percentage": 100}}
        
        try:
            route_line = LineString(coordinates)
            route_gdf = gpd.GeoDataFrame([1], geometry=[route_line], crs="EPSG:4326")
            
            # Simple buffer analysis
            buffer = route_gdf.buffer(0.0001)
            intersecting = self.roads_gdf[self.roads_gdf.intersects(buffer.iloc[0])]
            
            if intersecting.empty:
                return 50, {"NO_ROADS": {"percentage": 100}}
            
            # Basic scoring
            avg_score = 60  # Default
            facility_stats = {"MIXED": {"percentage": 100}}
            
            return avg_score, facility_stats
            
        except Exception as e:
            logger.error(f"Route analysis error: {e}")
            return 50, {"ERROR": {"percentage": 100}}

# =============================================================================
# Routing Functions
# =============================================================================

def format_time(minutes):
    if minutes < 60:
        return f"{int(minutes)} min"
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}h {m}m" if m > 0 else f"{h}h"

def get_bike_route(start_coords, end_coords, name="Bike Route"):
    try:
        coords_str = f"{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords_str}"
        
        response = requests.get(url, params={"overview": "full", "geometries": "polyline"}, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("code") != "Ok":
            return None
        
        route = data["routes"][0]
        distance_mi = route["distance"] * 0.000621371
        duration_min = route["duration"] / 60
        
        # Decode geometry
        coords = polyline.decode(route["geometry"])
        geometry_coords = [[lon, lat] for lat, lon in coords]
        
        # Analyze with shapefile
        score, facilities = shapefile_analyzer.analyze_route(geometry_coords)
        
        return {
            "name": name,
            "distance_miles": round(distance_mi, 2),
            "duration_minutes": round(duration_min, 1),
            "duration_text": format_time(duration_min),
            "geometry": {"type": "LineString", "coordinates": geometry_coords},
            "safety_score": score,
            "facilities": facilities
        }
        
    except Exception as e:
        logger.error(f"OSRM error: {e}")
        return None

def get_transit_route(start_coords, end_coords):
    if not GOOGLE_API_KEY:
        return None
    
    try:
        params = {
            "origin": f"{start_coords[1]},{start_coords[0]}",
            "destination": f"{end_coords[1]},{end_coords[0]}",
            "mode": "transit",
            "departure_time": int(datetime.datetime.now().timestamp()),
            "key": GOOGLE_API_KEY
        }
        
        response = requests.get("https://maps.googleapis.com/maps/api/directions/json", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "OK" or not data.get("routes"):
            return None
        
        route = data["routes"][0]
        leg = route["legs"][0]
        
        duration_min = leg["duration"]["value"] / 60
        distance_mi = leg["distance"]["value"] * 0.000621371
        
        # Get geometry
        geometry_coords = []
        if "overview_polyline" in route:
            coords = polyline.decode(route["overview_polyline"]["points"])
            geometry_coords = [[lon, lat] for lat, lon in coords]
        
        # Count transfers
        transit_steps = [s for s in leg["steps"] if s.get("travel_mode") == "TRANSIT"]
        transfers = max(0, len(transit_steps) - 1)
        
        return {
            "name": "Transit Route",
            "distance_miles": round(distance_mi, 2),
            "duration_minutes": round(duration_min, 1),
            "duration_text": format_time(duration_min),
            "transfers": transfers,
            "geometry": {"type": "LineString", "coordinates": geometry_coords}
        }
        
    except Exception as e:
        logger.error(f"Google transit error: {e}")
        return None

def find_transit_stops(coords, radius_m=800):
    stops = []
    
    # Try GTFS first
    if gtfs_manager.is_loaded:
        gtfs_stops = gtfs_manager.find_nearby_stops(coords[1], coords[0], radius_m/1000)
        for stop in gtfs_stops:
            stops.append({
                "id": stop["stop_id"],
                "name": stop["stop_name"],
                "lat": stop["stop_lat"],
                "lng": stop["stop_lon"],
                "distance_m": stop["distance_km"] * 1000
            })
    
    # Fallback to Google Places
    if not stops and GOOGLE_API_KEY:
        try:
            params = {
                "location": f"{coords[1]},{coords[0]}",
                "radius": radius_m,
                "type": "transit_station",
                "key": GOOGLE_API_KEY
            }
            response = requests.get("https://maps.googleapis.com/maps/api/place/nearbysearch/json", params=params, timeout=10)
            data = response.json()
            
            for place in data.get("results", [])[:3]:
                loc = place["geometry"]["location"]
                stops.append({
                    "id": place["place_id"],
                    "name": place["name"],
                    "lat": loc["lat"],
                    "lng": loc["lng"]
                })
        except Exception as e:
            logger.error(f"Places API error: {e}")
    
    return stops

def analyze_multimodal_routes(start_coords, end_coords):
    routes = []
    
    # Find transit stops
    start_stops = find_transit_stops(start_coords)
    end_stops = find_transit_stops(end_coords)
    
    # Create bike-bus-bike route if stops found
    if start_stops and end_stops:
        start_stop = start_stops[0]
        end_stop = end_stops[0]
        
        # Bike to transit
        bike1 = get_bike_route(start_coords, [start_stop["lng"], start_stop["lat"]], "Bike to Transit")
        # Transit
        transit = get_transit_route([start_stop["lng"], start_stop["lat"]], [end_stop["lng"], end_stop["lat"]])
        # Bike from transit
        bike2 = get_bike_route([end_stop["lng"], end_stop["lat"]], end_coords, "Bike from Transit")
        
        if bike1 and transit and bike2:
            total_time = bike1["duration_minutes"] + transit["duration_minutes"] + bike2["duration_minutes"] + 5
            total_distance = bike1["distance_miles"] + transit["distance_miles"] + bike2["distance_miles"]
            bike_distance = bike1["distance_miles"] + bike2["distance_miles"]
            avg_bike_score = (bike1["safety_score"] * bike1["distance_miles"] + 
                            bike2["safety_score"] * bike2["distance_miles"]) / bike_distance if bike_distance > 0 else 50
            
            routes.append({
                "id": 1,
                "name": "Bike-Bus-Bike",
                "type": "multimodal",
                "total_time_minutes": round(total_time, 1),
                "total_time_text": format_time(total_time),
                "total_distance_miles": round(total_distance, 2),
                "bike_distance_miles": round(bike_distance, 2),
                "bike_safety_score": round(avg_bike_score, 1),
                "transfers": transit["transfers"],
                "legs": [
                    {"type": "bike", "route": bike1, "color": "#27ae60"},
                    {"type": "transit", "route": transit, "color": "#3498db"},
                    {"type": "bike", "route": bike2, "color": "#27ae60"}
                ]
            })
    
    # Add direct bike route
    direct_bike = get_bike_route(start_coords, end_coords, "Direct Bike")
    if direct_bike:
        routes.append({
            "id": len(routes) + 1,
            "name": "Direct Bike",
            "type": "bike_only",
            "total_time_minutes": direct_bike["duration_minutes"],
            "total_time_text": direct_bike["duration_text"],
            "total_distance_miles": direct_bike["distance_miles"],
            "bike_distance_miles": direct_bike["distance_miles"],
            "bike_safety_score": direct_bike["safety_score"],
            "transfers": 0,
            "legs": [{"type": "bike", "route": direct_bike, "color": "#e74c3c"}]
        })
    
    # Sort by time
    routes.sort(key=lambda x: x["total_time_minutes"])
    
    return {
        "success": True,
        "routes": routes,
        "timestamp": datetime.datetime.now().isoformat()
    }

# =============================================================================
# Initialize Components
# =============================================================================

gtfs_manager = GTFSManager()
shapefile_analyzer = ShapefileAnalyzer()

def initialize_data():
    logger.info("Initializing data...")
    gtfs_manager.load_gtfs_data()
    shapefile_analyzer.load_data()

# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(title="Multimodal Transit Tool", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    initialize_data()

@app.get("/", response_class=HTMLResponse)
async def home():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Multimodal Transit Tool</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
        .header { background: #2c3e50; color: white; padding: 20px; text-align: center; }
        .container { display: flex; height: calc(100vh - 80px); }
        #map { flex: 2; }
        .sidebar { flex: 1; padding: 20px; max-width: 400px; background: #f8f9fa; overflow-y: auto; }
        .controls { margin-bottom: 20px; }
        select, button { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #3498db; color: white; border: none; cursor: pointer; }
        button:hover { background: #2980b9; }
        button:disabled { background: #bdc3c7; cursor: not-allowed; }
        .route-card { background: white; border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin: 10px 0; cursor: pointer; }
        .route-card:hover { background: #f8f9fa; }
        .route-header { font-weight: bold; margin-bottom: 10px; }
        .route-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 14px; }
        .coords { background: #e9ecef; padding: 10px; border-radius: 4px; margin: 10px 0; font-size: 12px; }
        .error { background: #f8d7da; color: #721c24; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #3498db; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: 20px auto; display: none; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Multimodal Transit Route Planner</h1>
        <p>Click two points on the map to plan your route</p>
    </div>
    
    <div class="container">
        <div id="map"></div>
        
        <div class="sidebar">
            <div class="controls">
                <select id="mode">
                    <option value="bike">Bike Routes Only</option>
                    <option value="transit">Transit Routes Only</option>
                    <option value="multimodal">Bike + Transit (Multimodal)</option>
                </select>
                <button id="findRoutes" disabled onclick="findRoutes()">Find Routes</button>
                <button onclick="clearMap()">Clear Map</button>
                
                <div class="coords">
                    <div><strong>Start:</strong> <span id="startCoords">Click map</span></div>
                    <div><strong>End:</strong> <span id="endCoords">Click map</span></div>
                </div>
            </div>
            
            <div class="spinner" id="spinner"></div>
            <div id="results"></div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        let map, startPoint = null, endPoint = null, startMarker = null, endMarker = null;
        let routeGroup = L.layerGroup(), currentRoutes = [];
        
        // Initialize map
        map = L.map('map').setView([30.3293, -81.6556], 12);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
        routeGroup.addTo(map);
        
        map.on('click', function(e) {
            if (!startPoint) {
                startPoint = e.latlng;
                if (startMarker) map.removeLayer(startMarker);
                startMarker = L.marker(startPoint).addTo(map).bindPopup("Start").openPopup();
                document.getElementById('startCoords').textContent = startPoint.lat.toFixed(4) + ', ' + startPoint.lng.toFixed(4);
            } else if (!endPoint) {
                endPoint = e.latlng;
                if (endMarker) map.removeLayer(endMarker);
                endMarker = L.marker(endPoint).addTo(map).bindPopup("End").openPopup();
                document.getElementById('endCoords').textContent = endPoint.lat.toFixed(4) + ', ' + endPoint.lng.toFixed(4);
                document.getElementById('findRoutes').disabled = false;
            } else {
                clearMap();
                map.fire('click', e);
            }
        });
        
        function clearMap() {
            startPoint = endPoint = null;
            if (startMarker) map.removeLayer(startMarker);
            if (endMarker) map.removeLayer(endMarker);
            startMarker = endMarker = null;
            routeGroup.clearLayers();
            document.getElementById('startCoords').textContent = 'Click map';
            document.getElementById('endCoords').textContent = 'Click map';
            document.getElementById('findRoutes').disabled = true;
            document.getElementById('results').innerHTML = '';
            currentRoutes = [];
        }
        
        async function findRoutes() {
            if (!startPoint || !endPoint) return;
            
            const mode = document.getElementById('mode').value;
            document.getElementById('spinner').style.display = 'block';
            document.getElementById('results').innerHTML = '';
            routeGroup.clearLayers();
            
            try {
                let endpoint = '';
                if (mode === 'bike') endpoint = '/api/bike-routes';
                else if (mode === 'transit') endpoint = '/api/transit-routes';
                else endpoint = '/api/multimodal-routes';
                
                const params = new URLSearchParams({
                    start_lng: startPoint.lng,
                    start_lat: startPoint.lat,
                    end_lng: endPoint.lng,
                    end_lat: endPoint.lat
                });
                
                const response = await fetch(endpoint + '?' + params);
                const data = await response.json();
                
                document.getElementById('spinner').style.display = 'none';
                
                if (data.error) {
                    document.getElementById('results').innerHTML = '<div class="error">' + data.error + '</div>';
                    return;
                }
                
                displayResults(data, mode);
                
            } catch (error) {
                document.getElementById('spinner').style.display = 'none';
                document.getElementById('results').innerHTML = '<div class="error">Error: ' + error.message + '</div>';
            }
        }
        
        function displayResults(data, mode) {
            let html = '';
            currentRoutes = [];
            
            if (mode === 'multimodal' && data.routes) {
                currentRoutes = data.routes;
                data.routes.forEach((route, i) => {
                    html += '<div class="route-card" onclick="showRoute(' + i + ')">';
                    html += '<div class="route-header">' + route.name + '</div>';
                    html += '<div class="route-stats">';
                    html += '<div>Time: ' + route.total_time_text + '</div>';
                    html += '<div>Distance: ' + route.total_distance_miles + ' mi</div>';
                    html += '<div>Bike Score: ' + route.bike_safety_score + '</div>';
                    html += '<div>Transfers: ' + route.transfers + '</div>';
                    html += '</div></div>';
                });
                if (data.routes.length > 0) showRoute(0);
            } else if (data.route) {
                currentRoutes = [data.route];
                html += '<div class="route-card">';
                html += '<div class="route-header">' + data.route.name + '</div>';
                html += '<div class="route-stats">';
                html += '<div>Time: ' + data.route.duration_text + '</div>';
                html += '<div>Distance: ' + data.route.distance_miles + ' mi</div>';
                if (data.route.safety_score) html += '<div>Safety: ' + data.route.safety_score + '</div>';
                if (data.route.transfers !== undefined) html += '<div>Transfers: ' + data.route.transfers + '</div>';
                html += '</div></div>';
                showRoute(0);
            } else {
                html = '<div class="error">No routes found</div>';
            }
            
            document.getElementById('results').innerHTML = html;
        }
        
        function showRoute(index) {
            routeGroup.clearLayers();
            const route = currentRoutes[index];
            if (!route) return;
            
            if (route.legs) {
                // Multimodal route
                route.legs.forEach(leg => {
                    if (leg.route && leg.route.geometry) {
                        const coords = leg.route.geometry.coordinates.map(c => [c[1], c[0]]);
                        L.polyline(coords, {color: leg.color, weight: 5}).addTo(routeGroup);
                    }
                });
            } else if (route.geometry) {
                // Single route
                const coords = route.geometry.coordinates.map(c => [c[1], c[0]]);
                L.polyline(coords, {color: '#3498db', weight: 5}).addTo(routeGroup);
            }
            
            if (routeGroup.getLayers().length > 0) {
                map.fitBounds(routeGroup.getBounds(), {padding: [20, 20]});
            }
        }
    </script>
</body>
</html>"""

@app.get("/api/bike-routes")
async def bike_routes(
    start_lng: float = Query(...), start_lat: float = Query(...),
    end_lng: float = Query(...), end_lat: float = Query(...)
):
    route = get_bike_route([start_lng, start_lat], [end_lng, end_lat])
    if route:
        return {"success": True, "route": route}
    else:
        raise HTTPException(status_code=404, detail="No bike route found")

@app.get("/api/transit-routes")
async def transit_routes(
    start_lng: float = Query(...), start_lat: float = Query(...),
    end_lng: float = Query(...), end_lat: float = Query(...)
):
    route = get_transit_route([start_lng, start_lat], [end_lng, end_lat])
    if route:
        return {"success": True, "route": route}
    else:
        raise HTTPException(status_code=404, detail="No transit route found")

@app.get("/api/multimodal-routes")
async def multimodal_routes(
    start_lng: float = Query(...), start_lat: float = Query(...),
    end_lng: float = Query(...), end_lat: float = Query(...)
):
    return analyze_multimodal_routes([start_lng, start_lat], [end_lng, end_lat])

@app.get("/api/status")
async def status():
    return {
        "status": "operational",
        "components": {
            "osrm": OSRM_SERVER,
            "google_api": bool(GOOGLE_API_KEY),
            "gtfs_loaded": gtfs_manager.is_loaded,
            "shapefiles_loaded": shapefile_analyzer.loaded
        },
        "config": {
            "bike_speed_mph": BIKE_SPEED_MPH,
            "data_url": DATA_ZIP_URL
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting Multimodal Transit Tool on port {port}")
    print(f"Google API: {'Configured' if GOOGLE_API_KEY else 'Missing'}")
    print(f"Data URL: {DATA_ZIP_URL}")
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
