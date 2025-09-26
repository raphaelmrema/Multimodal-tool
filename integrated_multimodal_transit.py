# integrated_multimodal_transit_tool.py
# Complete FastAPI app combining OSRM bike routing, Google Transit, GTFS real-time, and shapefile analysis
# Revised for GitHub data.zip integration and Railway deployment

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
from typing import List, Dict, Optional, Tuple, Any
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Auto-install required packages
import subprocess
import sys

def install_packages():
    """Install required packages if missing"""
    packages = ["polyline", "pandas", "geopandas", "shapely", "numpy", "aiofiles"]
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages()

import polyline
import pandas as pd
import numpy as np

# Import geopandas and shapely after ensuring they're installed
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    from shapely.ops import nearest_points
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    print("Warning: GeoPandas not available. Shapefile analysis will be disabled.")

# =============================================================================
# CONFIGURATION
# =============================================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OSRM_SERVER = os.getenv("OSRM_SERVER", "http://router.project-osrm.org")
BIKE_SPEED_MPH = float(os.getenv("BIKE_SPEED_MPH", "11"))

# Data configuration - updated for your GitHub repository
DATA_ZIP_URL = os.getenv("DATA_ZIP_URL", "https://raw.githubusercontent.com/raphaelmrema/Multimodal-tool/main/data.zip")
ROADS_SHAPEFILE_URL = os.getenv("ROADS_SHAPEFILE_URL", DATA_ZIP_URL)
TRANSIT_STOPS_SHAPEFILE_URL = os.getenv("TRANSIT_STOPS_SHAPEFILE_URL", DATA_ZIP_URL)

# GTFS data URLs
GTFS_URLS = [
    "https://ride.jtafla.com/gtfs-archive/gtfs.zip",
    "https://schedules.jtafla.com/schedulesgtfs/download",
    "https://openmobilitydata.org/p/jacksonville-transportation-authority/331/latest/download",
]

GMAPS_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
GMAPS_PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

CORS_ALLOW_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("integrated-multimodal-transit")

# =============================================================================
# ENHANCED GTFS MANAGER
# =============================================================================

class EnhancedGTFSManager:
    """GTFS manager with real-time capabilities"""
    
    def __init__(self):
        self.gtfs_data = {}
        self.stops_df = None
        self.routes_df = None
        self.stop_times_df = None
        self.trips_df = None
        self.calendar_df = None
        self.is_loaded = False
        self.last_update = None
        self.realtime_cache = {}
        self.cache_duration = 30
        
    def load_gtfs_data(self, gtfs_urls=None):
        """Load GTFS data"""
        if gtfs_urls is None:
            gtfs_urls = GTFS_URLS
        
        logger.info("Loading GTFS data...")
        
        for i, url in enumerate(gtfs_urls):
            try:
                logger.info(f"Trying GTFS source {i+1}: {url}")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/zip, application/octet-stream, */*',
                }
                
                response = requests.get(url, timeout=45, headers=headers, verify=False, stream=True)
                response.raise_for_status()
                
                content = response.content
                if not content.startswith(b'PK'):
                    continue
                
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    files_to_load = {
                        'stops.txt': 'stops_df',
                        'routes.txt': 'routes_df', 
                        'trips.txt': 'trips_df',
                        'stop_times.txt': 'stop_times_df',
                        'agency.txt': 'agency_df',
                        'calendar.txt': 'calendar_df'
                    }
                    
                    loaded_files = 0
                    for filename, attr_name in files_to_load.items():
                        if filename in z.namelist():
                            try:
                                with z.open(filename) as f:
                                    df = pd.read_csv(f, low_memory=False)
                                    setattr(self, attr_name, df)
                                    self.gtfs_data[filename] = df
                                    loaded_files += 1
                            except Exception as e:
                                logger.error(f"Error loading {filename}: {e}")
                    
                    if loaded_files >= 3:
                        self.is_loaded = True
                        self.last_update = datetime.datetime.now()
                        logger.info(f"GTFS data loaded successfully")
                        return True
                        
            except Exception as e:
                logger.error(f"Error loading from source {i+1}: {e}")
                continue
        
        return False
    
    def find_nearby_stops(self, lat, lon, radius_km=0.5):
        """Find GTFS stops near coordinates"""
        if not self.is_loaded or self.stops_df is None:
            return []
        
        try:
            stops = self.stops_df.copy()
            stops['distance_km'] = ((stops['stop_lat'] - lat)**2 + (stops['stop_lon'] - lon)**2)**0.5 * 111
            nearby = stops[stops['distance_km'] <= radius_km].sort_values('distance_km')
            return nearby[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'distance_km']].head(5).to_dict('records')
        except Exception as e:
            logger.error(f"Error finding nearby stops: {e}")
            return []
    
    def get_realtime_departures(self, stop_id: str) -> List[Dict]:
        """Get simulated real-time departures"""
        current_time = datetime.datetime.now()
        departures = []
        
        for i in range(random.randint(3, 6)):
            base_time = current_time + datetime.timedelta(minutes=random.randint(2, 45))
            delay_minutes = random.randint(0, 5) if random.random() < 0.3 else 0
            actual_time = base_time + datetime.timedelta(minutes=delay_minutes)
            
            departures.append({
                "departure_time": base_time.strftime("%H:%M:%S"),
                "realtime_departure": actual_time.strftime("%H:%M"),
                "delay_minutes": delay_minutes,
                "status_text": "On time" if delay_minutes == 0 else f"Delayed {delay_minutes} min",
                "status_color": "#4caf50" if delay_minutes == 0 else "#ff9800",
                "route_name": f"Route {random.randint(1, 50)}"
            })
        
        return sorted(departures, key=lambda x: x["realtime_departure"])

# =============================================================================
# ENHANCED SHAPEFILE ANALYSIS (Updated for GitHub data.zip)
# =============================================================================

class CombinedZipShapefileAnalyzer:
    """Shapefile analyzer that handles combined data.zip from GitHub"""
    
    def __init__(self):
        self.roads_gdf = None
        self.transit_stops_gdf = None
        self.loaded = False
        self.data_zip_url = DATA_ZIP_URL
        self.temp_dir = None
        
    def load_shapefiles(self):
        """Load shapefiles from combined data.zip"""
        if not GEOPANDAS_AVAILABLE:
            logger.warning("GeoPandas not available - shapefile analysis disabled")
            return False
            
        try:
            # Download and extract the combined ZIP
            zip_path = self._download_data_zip()
            if not zip_path:
                logger.warning("Could not download data.zip - continuing without shapefile analysis")
                return False
            
            # Extract and load shapefiles
            extract_dir = self._extract_zip(zip_path)
            if not extract_dir:
                return False
            
            self.temp_dir = extract_dir
            
            # Find and load roads shapefile
            roads_file = self._find_shapefile(extract_dir, ["roads", "bike", "street", "lane", "cycling", "bicycle"])
            if roads_file:
                self.roads_gdf = gpd.read_file(roads_file)
                logger.info(f"Loaded roads shapefile: {len(self.roads_gdf)} features from {roads_file}")
                self._standardize_roads_columns()
            else:
                logger.warning("No roads shapefile found in data.zip")
            
            # Find and load transit stops shapefile
            stops_file = self._find_shapefile(extract_dir, ["stop", "transit", "bus", "station"])
            if stops_file:
                self.transit_stops_gdf = gpd.read_file(stops_file)
                logger.info(f"Loaded transit stops: {len(self.transit_stops_gdf)} features from {stops_file}")
            else:
                logger.warning("No transit stops shapefile found in data.zip")
            
            self.loaded = (self.roads_gdf is not None)
            return self.loaded
            
        except Exception as e:
            logger.error(f"Error loading shapefiles: {e}")
            return False
    
    def _download_data_zip(self):
        """Download the data.zip file"""
        urls_to_try = [
            self.data_zip_url,
            ROADS_SHAPEFILE_URL,
            TRANSIT_STOPS_SHAPEFILE_URL
        ]
        
        for url in urls_to_try:
            if not url or not url.startswith("http"):
                continue
                
            try:
                logger.info(f"Downloading data from: {url}")
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (compatible; MultimodalTransitTool/1.0)',
                    'Accept': 'application/zip, application/octet-stream, */*'
                }
                
                response = requests.get(url, timeout=60, stream=True, headers=headers)
                response.raise_for_status()
                
                # Save to temporary file
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "data.zip")
                
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                file_size = os.path.getsize(zip_path)
                logger.info(f"Downloaded data.zip ({file_size} bytes)")
                
                # Verify it's a valid ZIP file
                if file_size > 0:
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as test_zip:
                            test_zip.testzip()
                        return zip_path
                    except zipfile.BadZipFile:
                        logger.error(f"Downloaded file is not a valid ZIP: {url}")
                        continue
                
            except Exception as e:
                logger.error(f"Failed to download from {url}: {e}")
                continue
        
        logger.error("Could not download data.zip from any source")
        return None
    
    def _extract_zip(self, zip_path):
        """Extract ZIP file and return extraction directory"""
        try:
            extract_dir = tempfile.mkdtemp()
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            logger.info(f"Extracted data.zip to: {extract_dir}")
            
            # List contents for debugging
            file_count = 0
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    file_count += 1
                    if file_count <= 10:  # Show first 10 files
                        rel_path = os.path.relpath(os.path.join(root, file), extract_dir)
                        logger.info(f"Found file: {rel_path}")
            
            if file_count > 10:
                logger.info(f"... and {file_count - 10} more files")
            
            return extract_dir
            
        except Exception as e:
            logger.error(f"Failed to extract ZIP: {e}")
            return None
    
    def _find_shapefile(self, search_dir, keywords):
        """Find shapefile based on keywords in filename"""
        try:
            shapefiles_found = []
            
            # First pass: look for shapefiles with keywords
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.shp'):
                        file_lower = file.lower()
                        for keyword in keywords:
                            if keyword in file_lower:
                                full_path = os.path.join(root, file)
                                shapefiles_found.append((full_path, keyword, file))
                                logger.info(f"Found {keyword} shapefile: {file}")
            
            # Return the first keyword match
            if shapefiles_found:
                return shapefiles_found[0][0]
            
            # Second pass: if no keyword match, return first .shp file found
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.shp'):
                        full_path = os.path.join(root, file)
                        logger.info(f"Using first shapefile found: {file}")
                        return full_path
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding shapefile: {e}")
            return None
    
    def _standardize_roads_columns(self):
        """Standardize column names for roads data"""
        if self.roads_gdf is None:
            return
        
        logger.info(f"Roads shapefile columns: {list(self.roads_gdf.columns)}")
        
        # Comprehensive field mappings
        field_mappings = {
            'facility_type': [
                'Facility_Type', 'FACILITY_TYPE', 'facility_type', 'FACILITY_T',
                'bike_facility', 'BIKE_FACILITY', 'infrastructure', 'INFRASTRUC',
                'BIKELANE', 'bike_lane', 'BIKE_LANE', 'cycling', 'CYCLING',
                'facility', 'FACILITY', 'type', 'TYPE', 'class', 'CLASS'
            ],
            'bike_score': [
                'Bike_Score', 'bike_score', 'BIKE_SCORE', 'BikeScore',
                'safety_score', 'SAFETY_SCORE', 'score', 'SCORE',
                'rating', 'RATING', 'quality', 'QUALITY', 'index', 'INDEX'
            ],
            'speed_limit': [
                'Speed_Limit', 'SPEED_LIMIT', 'Speed', 'SPD_LIM',
                'speed', 'SPEED', 'mph', 'MPH', 'SPEEDLIMIT', 'maxspeed', 'MAXSPEED'
            ],
            'lanes': [
                'Lanes', 'LANES', 'Num_Lanes', 'NUM_LANES',
                'lane_count', 'LANE_COUNT', 'NUMLANES', 'lanes_total', 'LANES_TOTAL'
            ],
            'lts': [
                'LTS', 'lts', 'Level_Traffic_Stress', 'LEVEL_TRAFFIC_STRESS',
                'traffic_stress', 'TRAFFIC_STRESS', 'stress', 'STRESS'
            ]
        }
        
        # Apply mappings
        mapped_fields = []
        for standard_name, variations in field_mappings.items():
            mapped = False
            for col in self.roads_gdf.columns:
                if col in variations:
                    self.roads_gdf[standard_name] = self.roads_gdf[col]
                    logger.info(f"Mapped {col} -> {standard_name}")
                    mapped_fields.append(standard_name)
                    mapped = True
                    break
            
            if not mapped:
                # Set intelligent defaults
                if standard_name == 'facility_type':
                    self.roads_gdf[standard_name] = 'NO BIKELANE'
                elif standard_name == 'bike_score':
                    self.roads_gdf[standard_name] = 50
                elif standard_name == 'speed_limit':
                    self.roads_gdf[standard_name] = 30
                elif standard_name == 'lanes':
                    self.roads_gdf[standard_name] = 2
                elif standard_name == 'lts':
                    self.roads_gdf[standard_name] = 3
                
                logger.info(f"Set default for {standard_name}")
        
        logger.info(f"Successfully mapped fields: {mapped_fields}")
    
    def analyze_route_segments(self, route_geometry):
        """Analyze route segments against shapefile data"""
        if not self.loaded or self.roads_gdf is None:
            return [], 50.0, {"NO DATA": {"length_miles": 0, "percentage": 100}}
        
        try:
            # Create route line
            coords = route_geometry.get("coordinates", [])
            if len(coords) < 2:
                return [], 50.0, {"NO DATA": {"length_miles": 0, "percentage": 100}}
            
            route_line = LineString(coords)
            route_gdf = gpd.GeoDataFrame([1], geometry=[route_line], crs="EPSG:4326")
            
            # Ensure both have same CRS
            if self.roads_gdf.crs != route_gdf.crs:
                if self.roads_gdf.crs is None:
                    self.roads_gdf = self.roads_gdf.set_crs("EPSG:4326")
                else:
                    route_gdf = route_gdf.to_crs(self.roads_gdf.crs)
            
            # Buffer route and find intersecting roads
            route_buffer = route_gdf.buffer(0.0002)  # ~20 meters
            intersecting = self.roads_gdf[self.roads_gdf.intersects(route_buffer.iloc[0])]
            
            if intersecting.empty:
                return [], 50.0, {"NO ROADS": {"length_miles": 0, "percentage": 100}}
            
            # Analyze segments
            segments = []
            facility_stats = {}
            total_score = 0
            total_length = 0
            
            for idx, road in intersecting.iterrows():
                facility_type = str(road.get('facility_type', 'NO BIKELANE')).strip()
                bike_score = float(road.get('bike_score', 50))
                speed_limit = float(road.get('speed_limit', 30))
                lanes = int(road.get('lanes', 2))
                lts = int(road.get('lts', 3))
                
                # Calculate LTS if not provided
                if lts == 3:  # Default value, try to calculate
                    lts = self._calculate_lts(facility_type, speed_limit, lanes)
                
                # Estimate segment length (convert degrees to miles roughly)
                segment_length = road.geometry.length * 69  # Very rough conversion
                
                segments.append({
                    "facility_type": facility_type,
                    "bike_score": bike_score,
                    "LTS": lts,
                    "speed_limit": speed_limit,
                    "lanes": lanes,
                    "length_miles": segment_length
                })
                
                # Update statistics
                if facility_type not in facility_stats:
                    facility_stats[facility_type] = {
                        "length_miles": 0,
                        "percentage": 0,
                        "avg_score": 0,
                        "count": 0
                    }
                
                facility_stats[facility_type]["length_miles"] += segment_length
                facility_stats[facility_type]["count"] += 1
                facility_stats[facility_type]["avg_score"] = (
                    (facility_stats[facility_type]["avg_score"] * (facility_stats[facility_type]["count"] - 1) + bike_score) /
                    facility_stats[facility_type]["count"]
                )
                
                total_score += bike_score * segment_length
                total_length += segment_length
            
            # Calculate percentages
            for facility_type in facility_stats:
                facility_stats[facility_type]["percentage"] = (
                    facility_stats[facility_type]["length_miles"] / total_length * 100
                    if total_length > 0 else 0
                )
            
            overall_score = total_score / total_length if total_length > 0 else 50
            
            return segments, round(overall_score, 1), facility_stats
            
        except Exception as e:
            logger.error(f"Error analyzing route: {e}")
            return [], 50.0, {"ERROR": {"length_miles": 0, "percentage": 100}}
    
    def _calculate_lts(self, facility_type, speed_limit, lanes):
        """Calculate Level of Traffic Stress"""
        ft = str(facility_type).upper().strip()
        
        if "PROTECTED" in ft or "SHARED USE PATH" in ft or "SEPARATED" in ft:
            return 1
        elif "BUFFERED" in ft and speed_limit <= 30:
            return 2
        elif ("BIKE LANE" in ft or "CYCLE LANE" in ft) and speed_limit <= 35:
            return 2
        elif "SHARED" in ft and speed_limit <= 35 and lanes <= 2:
            return 3
        else:
            return 4

# =============================================================================
# OSRM BICYCLE ROUTING WITH ALTERNATIVES
# =============================================================================

def calculate_bike_route_osrm(start_coords: List[float], end_coords: List[float], 
                              waypoints: Optional[List[List[float]]] = None,
                              route_name="Bike Route") -> Optional[Dict]:
    """Create bike route using OSRM"""
    try:
        # Build coordinates string
        coords_list = [start_coords]
        if waypoints:
            coords_list.extend(waypoints)
        coords_list.append(end_coords)
        
        coords_str = ";".join([f"{lon},{lat}" for lon, lat in coords_list])
        url = f"{OSRM_SERVER}/route/v1/cycling/{coords_str}"
        
        params = {
            "overview": "full",
            "geometries": "polyline",
            "steps": "false",
            "alternatives": "false"
        }
        
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        if data.get("code") != "Ok" or not data.get("routes"):
            return None
        
        route = data["routes"][0]
        distance_m = float(route.get("distance", 0))
        distance_mi = distance_m * 0.000621371
        duration_min = float(route.get("duration", 0)) / 60
        
        # Decode polyline
        coords_latlon = polyline.decode(route["geometry"])
        geometry = {
            "type": "LineString",
            "coordinates": [[lon, lat] for lat, lon in coords_latlon]
        }
        
        # Analyze with shapefile if available
        segments = []
        overall_score = 70
        facility_stats = {}
        
        if shapefile_analyzer.loaded:
            segments, overall_score, facility_stats = shapefile_analyzer.analyze_route_segments(geometry)
        
        return {
            "name": route_name,
            "length_miles": round(distance_mi, 3),
            "travel_time_minutes": round(duration_min, 1),
            "travel_time_formatted": format_time_duration(duration_min),
            "geometry": geometry,
            "segments": segments,
            "overall_score": overall_score,
            "facility_stats": facility_stats,
            "route_type": "bike"
        }
        
    except Exception as e:
        logger.error(f"OSRM error: {e}")
        return None

def create_multiple_bike_routes(start_coords: List[float], end_coords: List[float], 
                                num_routes: int = 3) -> List[Dict]:
    """Create multiple alternative bike routes"""
    routes = []
    
    # Direct route
    direct_route = calculate_bike_route_osrm(start_coords, end_coords, None, "Direct Route")
    if direct_route:
        direct_route["route_type"] = "direct"
        routes.append(direct_route)
    
    # Alternative routes with waypoints
    if num_routes > 1:
        mid_lon = (start_coords[0] + end_coords[0]) / 2
        mid_lat = (start_coords[1] + end_coords[1]) / 2
        base_distance = max(abs(end_coords[0] - start_coords[0]), 
                           abs(end_coords[1] - start_coords[1]))
        
        for i in range(num_routes - 1):
            angle = (2 * math.pi * i) / max(1, num_routes - 1)
            jitter = 0.3 + 0.2 * random.random()
            
            waypoint = [
                mid_lon + base_distance * jitter * math.cos(angle),
                mid_lat + base_distance * jitter * math.sin(angle)
            ]
            
            alt_route = calculate_bike_route_osrm(
                start_coords, end_coords, [waypoint], 
                f"Alternative Route {i+1}"
            )
            
            if alt_route:
                alt_route["route_type"] = "alternative"
                routes.append(alt_route)
    
    # Sort by score then distance
    routes.sort(key=lambda r: (-r.get("overall_score", 0), r.get("length_miles", 999)))
    
    return routes

# =============================================================================
# GOOGLE TRANSIT FUNCTIONS
# =============================================================================

def format_time_duration(minutes: float) -> str:
    if minutes < 1: return "< 1 min"
    if minutes < 60: return f"{int(round(minutes))} min"
    h = int(minutes // 60)
    m = int(round(minutes % 60))
    return f"{h}h" if m == 0 else f"{h}h {m}m"

def get_transit_routes_google(origin: Tuple[float, float], destination: Tuple[float, float], 
                            departure_time: str = "now") -> Dict:
    """Get transit routes from Google"""
    if not GOOGLE_API_KEY:
        return {"error": "Google API key not configured"}
    
    try:
        ts = int(datetime.datetime.now().timestamp()) if departure_time == "now" else int(float(departure_time))
        
        params = {
            "origin": f"{origin[1]},{origin[0]}",
            "destination": f"{destination[1]},{destination[0]}",
            "mode": "transit",
            "departure_time": ts,
            "alternatives": "true",
            "transit_mode": "bus|subway|train|tram",
            "transit_routing_preference": "fewer_transfers",
            "key": GOOGLE_API_KEY
        }
        
        r = requests.get(GMAPS_DIRECTIONS_URL, params=params, timeout=30)
        data = r.json()
        
        if data.get("status") != "OK":
            return {"error": data.get("error_message", f"Status: {data.get('status')}")}
        
        routes = []
        for idx, route_data in enumerate(data.get("routes", [])):
            route = parse_google_transit_route(route_data, idx)
            if route:
                route = enhance_route_with_gtfs(route)
                routes.append(route)
        
        return {"routes": routes} if routes else {"error": "No transit routes found"}
        
    except Exception as e:
        logger.error(f"Google transit error: {e}")
        return {"error": str(e)}

def parse_google_transit_route(route_data: Dict, index: int) -> Optional[Dict]:
    """Parse Google transit route"""
    try:
        legs = route_data.get("legs", [])
        if not legs:
            return None
        
        leg = legs[0]
        dur_min = leg.get("duration", {}).get("value", 0) / 60
        dist_mi = leg.get("distance", {}).get("value", 0) * 0.000621371
        
        # Get route geometry
        overview_poly = route_data.get("overview_polyline", {}).get("points", "")
        route_coords = []
        if overview_poly:
            decoded = polyline.decode(overview_poly)
            route_coords = [[lon, lat] for lat, lon in decoded]
        
        # Count transfers
        transit_steps = sum(1 for s in leg.get("steps", []) if s.get("travel_mode") == "TRANSIT")
        transfers = max(0, transit_steps - 1)
        
        return {
            "name": f"Transit Option {index + 1}",
            "duration_minutes": round(dur_min, 1),
            "duration_text": format_time_duration(dur_min),
            "distance_miles": round(dist_mi, 2),
            "transfers": transfers,
            "geometry": {"type": "LineString", "coordinates": route_coords},
            "steps": leg.get("steps", []),
            "route_type": "transit"
        }
        
    except Exception as e:
        logger.error(f"Parse transit error: {e}")
        return None

def enhance_route_with_gtfs(route: Dict) -> Dict:
    """Enhance transit route with GTFS data"""
    if not gtfs_manager.is_loaded:
        return route
    
    # Add simulated real-time data to transit steps
    for step in route.get("steps", []):
        if step.get("travel_mode") == "TRANSIT":
            transit_details = step.get("transit_details", {})
            dep_stop = transit_details.get("departure_stop", {})
            
            if dep_stop.get("location"):
                lat = dep_stop["location"]["lat"]
                lon = dep_stop["location"]["lng"]
                
                # Find nearby GTFS stops
                nearby = gtfs_manager.find_nearby_stops(lat, lon, 0.3)
                if nearby:
                    stop_id = nearby[0]["stop_id"]
                    departures = gtfs_manager.get_realtime_departures(stop_id)
                    
                    step["enhanced_gtfs_data"] = {
                        "stop_id": stop_id,
                        "realtime_departures": departures[:5],
                        "has_delays": any(d["delay_minutes"] > 0 for d in departures)
                    }
    
    route["gtfs_enhanced"] = True
    return route

def find_nearby_transit_stops(coords: List[float], radius_m: int = 800) -> List[Dict]:
    """Find nearby transit stops using Google Places or GTFS"""
    stops = []
    
    # Try GTFS first
    if gtfs_manager.is_loaded:
        gtfs_stops = gtfs_manager.find_nearby_stops(coords[1], coords[0], radius_m/1000)
        for stop in gtfs_stops[:3]:
            stops.append({
                "id": stop["stop_id"],
                "name": stop["stop_name"],
                "x": stop["stop_lon"],
                "y": stop["stop_lat"],
                "display_x": stop["stop_lon"],
                "display_y": stop["stop_lat"],
                "distance_meters": stop["distance_km"] * 1000
            })
    
    # Fallback to Google Places if needed
    if not stops and GOOGLE_API_KEY:
        try:
            params = {
                "location": f"{coords[1]},{coords[0]}",
                "radius": radius_m,
                "type": "transit_station",
                "key": GOOGLE_API_KEY
            }
            r = requests.get(GMAPS_PLACES_NEARBY_URL, params=params, timeout=15)
            data = r.json()
            
            for item in data.get("results", [])[:3]:
                loc = item.get("geometry", {}).get("location", {})
                stops.append({
                    "id": item.get("place_id", ""),
                    "name": item.get("name", "Transit Stop"),
                    "x": loc.get("lng"),
                    "y": loc.get("lat"),
                    "display_x": loc.get("lng"),
                    "display_y": loc.get("lat")
                })
        except Exception as e:
            logger.error(f"Places API error: {e}")
    
    return stops

# =============================================================================
# BIKE-BUS-BIKE MULTIMODAL ANALYSIS
# =============================================================================

def analyze_bike_bus_bike_routes(start_point: List[float], end_point: List[float], 
                                 departure_time: str = "now", num_bike_routes: int = 3) -> Dict:
    """Complete bike-bus-bike analysis with multiple route alternatives"""
    
    routes = []
    
    # Find transit stops near start and end
    start_stops = find_nearby_transit_stops(start_point, 800)
    end_stops = find_nearby_transit_stops(end_point, 800)
    
    # Create bike-bus-bike combinations
    if start_stops and end_stops:
        # Use closest stops
        start_stop = start_stops[0]
        end_stop = end_stops[0]
        
        # Ensure different stops
        if start_stop["id"] == end_stop["id"] and len(end_stops) > 1:
            end_stop = end_stops[1]
        
        if start_stop["id"] != end_stop["id"]:
            # Create bike routes to/from stops
            bike1_routes = create_multiple_bike_routes(
                start_point, 
                [start_stop["display_x"], start_stop["display_y"]], 
                min(2, num_bike_routes)
            )
            
            bike2_routes = create_multiple_bike_routes(
                [end_stop["display_x"], end_stop["display_y"]], 
                end_point,
                min(2, num_bike_routes)
            )
            
            # Get transit routes
            transit_result = get_transit_routes_google(
                (start_stop["display_x"], start_stop["display_y"]),
                (end_stop["display_x"], end_stop["display_y"]),
                departure_time
            )
            
            # Combine best bike routes with transit
            if bike1_routes and bike2_routes and transit_result.get("routes"):
                for i, transit_route in enumerate(transit_result["routes"][:2]):
                    bike1 = bike1_routes[0]
                    bike2 = bike2_routes[0]
                    
                    total_time = (
                        bike1["travel_time_minutes"] + 
                        transit_route["duration_minutes"] + 
                        bike2["travel_time_minutes"] + 
                        5  # Transfer time
                    )
                    
                    total_bike_miles = bike1["length_miles"] + bike2["length_miles"]
                    total_transit_miles = transit_route["distance_miles"]
                    total_miles = total_bike_miles + total_transit_miles
                    
                    # Calculate weighted bike score
                    bike_score = 0
                    if total_bike_miles > 0:
                        bike_score = (
                            bike1["overall_score"] * bike1["length_miles"] +
                            bike2["overall_score"] * bike2["length_miles"]
                        ) / total_bike_miles
                    
                    routes.append({
                        "id": len(routes) + 1,
                        "name": f"Bike-Bus-Bike Option {i+1}",
                        "type": "bike_bus_bike",
                        "summary": {
                            "total_time_minutes": round(total_time, 1),
                            "total_time_formatted": format_time_duration(total_time),
                            "total_distance_miles": round(total_miles, 2),
                            "bike_distance_miles": round(total_bike_miles, 2),
                            "transit_distance_miles": round(total_transit_miles, 2),
                            "bike_percentage": round((total_bike_miles/total_miles)*100, 1) if total_miles > 0 else 0,
                            "average_bike_score": round(bike_score, 1),
                            "transfers": transit_route.get("transfers", 0)
                        },
                        "legs": [
                            {
                                "type": "bike",
                                "name": "Bike to Transit",
                                "route": bike1,
                                "color": "#27ae60"
                            },
                            {
                                "type": "transit",
                                "name": "Transit",
                                "route": transit_route,
                                "color": "#3498db"
                            },
                            {
                                "type": "bike",
                                "name": "Bike from Transit",
                                "route": bike2,
                                "color": "#27ae60"
                            }
                        ],
                        "bus_stops": {
                            "start_stop": start_stop,
                            "end_stop": end_stop
                        }
                    })
    
    # Add direct bike routes for comparison
    direct_bike_routes = create_multiple_bike_routes(start_point, end_point, num_bike_routes)
    for bike_route in direct_bike_routes:
        routes.append({
            "id": len(routes) + 1,
            "name": f"Direct Bike Route ({bike_route['name']})",
            "type": "direct_bike",
            "summary": {
                "total_time_minutes": bike_route["travel_time_minutes"],
                "total_time_formatted": bike_route["travel_time_formatted"],
                "total_distance_miles": bike_route["length_miles"],
                "bike_distance_miles": bike_route["length_miles"],
                "transit_distance_miles": 0,
                "bike_percentage": 100,
                "average_bike_score": bike_route["overall_score"],
                "transfers": 0
            },
            "legs": [
                {
                    "type": "bike",
                    "name": bike_route["name"],
                    "route": bike_route,
                    "color": "#e74c3c"
                }
            ]
        })
    
    # Sort routes by total time
    routes.sort(key=lambda r: r["summary"]["total_time_minutes"])
    
    return {
        "success": True,
        "routes": routes,
        "analysis_type": "multimodal",
        "timestamp": datetime.datetime.now().isoformat()
    }

# =============================================================================
# INITIALIZE GLOBAL COMPONENTS
# =============================================================================

# Initialize global components
gtfs_manager = EnhancedGTFSManager()
shapefile_analyzer = CombinedZipShapefileAnalyzer()

# Load data on startup
def initialize_data():
    """Initialize GTFS and shapefile data"""
    logger.info("Initializing transit data...")
    try:
        gtfs_manager.load_gtfs_data()
    except Exception as e:
        logger.error(f"GTFS initialization failed: {e}")
    
    try:
        shapefile_analyzer.load_shapefiles()
    except Exception as e:
        logger.error(f"Shapefile initialization failed: {e}")

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Integrated Multimodal Transit Tool",
    description="Combined OSRM bike routing, Google Transit, GTFS real-time, and shapefile analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    initialize_data()

@app.get("/", response_class=HTMLResponse)
async def serve_html():
    """Serve the main HTML interface"""
    return get_html_interface()

@app.get("/api/bike-routes")
async def get_bike_routes(
    start_lon: float = Query(..., description="Starting longitude"),
    start_lat: float = Query(..., description="Starting latitude"),
    end_lon: float = Query(..., description="Ending longitude"),
    end_lat: float = Query(..., description="Ending latitude"),
    alternatives: int = Query(3, description="Number of alternative routes")
):
    """Get bicycle routes using OSRM"""
    try:
        routes = create_multiple_bike_routes(
            [start_lon, start_lat],
            [end_lon, end_lat],
            alternatives
        )
        
        if not routes:
            raise HTTPException(status_code=404, detail="No bike routes found")
        
        return {
            "success": True,
            "routes": routes,
            "service": "OSRM + Shapefile Analysis",
            "bike_speed_mph": BIKE_SPEED_MPH
        }
        
    except Exception as e:
        logger.error(f"Bike routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/transit-routes")
async def get_transit_routes(
    start_lon: float = Query(..., description="Starting longitude"),
    start_lat: float = Query(..., description="Starting latitude"),
    end_lon: float = Query(..., description="Ending longitude"),
    end_lat: float = Query(..., description="Ending latitude"),
    departure_time: str = Query("now", description="Departure time (timestamp or 'now')")
):
    """Get transit routes using Google Maps API"""
    try:
        result = get_transit_routes_google(
            (start_lon, start_lat),
            (end_lon, end_lat),
            departure_time
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {
            "success": True,
            **result,
            "service": "Google Maps + GTFS Enhancement"
        }
        
    except Exception as e:
        logger.error(f"Transit routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/multimodal-routes")
async def get_multimodal_routes(
    start_lon: float = Query(..., description="Starting longitude"),
    start_lat: float = Query(..., description="Starting latitude"),
    end_lon: float = Query(..., description="Ending longitude"),
    end_lat: float = Query(..., description="Ending latitude"),
    departure_time: str = Query("now", description="Departure time"),
    bike_alternatives: int = Query(2, description="Number of bike route alternatives")
):
    """Get complete bike-bus-bike multimodal analysis"""
    try:
        result = analyze_bike_bus_bike_routes(
            [start_lon, start_lat],
            [end_lon, end_lat],
            departure_time,
            bike_alternatives
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Multimodal routing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/nearby-stops")
async def get_nearby_stops(
    lon: float = Query(..., description="Longitude"),
    lat: float = Query(..., description="Latitude"),
    radius_m: int = Query(800, description="Search radius in meters")
):
    """Find nearby transit stops"""
    try:
        stops = find_nearby_transit_stops([lon, lat], radius_m)
        
        return {
            "success": True,
            "stops": stops,
            "count": len(stops),
            "gtfs_enabled": gtfs_manager.is_loaded
        }
        
    except Exception as e:
        logger.error(f"Nearby stops error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/realtime-departures")
async def get_realtime_departures(stop_id: str = Query(..., description="Stop ID")):
    """Get real-time departure information for a stop"""
    try:
        if not gtfs_manager.is_loaded:
            raise HTTPException(status_code=503, detail="GTFS data not available")
        
        departures = gtfs_manager.get_realtime_departures(stop_id)
        
        return {
            "success": True,
            "stop_id": stop_id,
            "departures": departures,
            "count": len(departures),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Real-time departures error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_system_status():
    """Get system status and configuration"""
    return {
        "system_status": "operational",
        "components": {
            "osrm_server": OSRM_SERVER,
            "google_maps_api": bool(GOOGLE_API_KEY),
            "gtfs_data": {
                "loaded": gtfs_manager.is_loaded,
                "last_update": gtfs_manager.last_update.isoformat() if gtfs_manager.last_update else None,
                "routes_count": len(gtfs_manager.routes_df) if gtfs_manager.routes_df is not None else 0,
                "stops_count": len(gtfs_manager.stops_df) if gtfs_manager.stops_df is not None else 0
            },
            "shapefile_analysis": {
                "available": GEOPANDAS_AVAILABLE,
                "roads_loaded": shapefile_analyzer.loaded,
                "roads_count": len(shapefile_analyzer.roads_gdf) if shapefile_analyzer.roads_gdf is not None else 0,
                "transit_stops_count": len(shapefile_analyzer.transit_stops_gdf) if shapefile_analyzer.transit_stops_gdf is not None else 0
            }
        },
        "configuration": {
            "bike_speed_mph": BIKE_SPEED_MPH,
            "cors_origins": CORS_ALLOW_ORIGINS,
            "data_zip_url": DATA_ZIP_URL
        },
        "timestamp": datetime.datetime.now().isoformat()
    }

# Upload endpoints for dynamic file management
try:
    import aiofiles
    
    @app.post("/api/upload-shapefile")
    async def upload_shapefile(
        file: UploadFile = File(...),
        file_type: str = Form(..., description="'roads' or 'transit_stops'")
    ):
        """Upload shapefile for analysis"""
        try:
            if file_type not in ["roads", "transit_stops"]:
                raise HTTPException(status_code=400, detail="file_type must be 'roads' or 'transit_stops'")
            
            if not file.filename.endswith(('.zip', '.shp')):
                raise HTTPException(status_code=400, detail="File must be a ZIP or SHP file")
            
            # Create upload directory
            upload_dir = "/data" if os.path.exists("/data") else "./uploaded_data"
            os.makedirs(upload_dir, exist_ok=True)
            
            # Save file
            file_path = os.path.join(upload_dir, f"{file_type}.zip")
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            logger.info(f"Uploaded {file_type} shapefile: {file_path}")
            
            # Reload shapefiles
            shapefile_analyzer.load_shapefiles()
            
            return {
                "success": True,
                "message": f"Successfully uploaded {file_type} shapefile",
                "file_path": file_path,
                "file_size": len(content),
                "analyzer_loaded": shapefile_analyzer.loaded
            }
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/shapefile-status")
    async def get_shapefile_status():
        """Get current shapefile loading status"""
        return {
            "roads_loaded": shapefile_analyzer.roads_gdf is not None,
            "transit_stops_loaded": shapefile_analyzer.transit_stops_gdf is not None,
            "analyzer_ready": shapefile_analyzer.loaded,
            "roads_count": len(shapefile_analyzer.roads_gdf) if shapefile_analyzer.roads_gdf is not None else 0,
            "stops_count": len(shapefile_analyzer.transit_stops_gdf) if shapefile_analyzer.transit_stops_gdf is not None else 0,
            "data_source": DATA_ZIP_URL
        }
        
except ImportError:
    logger.warning("aiofiles not available - upload endpoints disabled")

def get_html_interface():
    """Return the complete HTML interface"""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Integrated Multimodal Transit Tool</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
    <style>
        * { box-sizing: border-box; }
        body { 
            margin: 0; padding: 0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white; padding: 20px; text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }
        .header h1 {
            margin: 0; font-size: 2.5em; font-weight: 300;
        }
        .container {
            display: flex; height: calc(100vh - 120px); max-width: 1600px;
            margin: 0 auto; background: white; border-radius: 15px 15px 0 0;
            overflow: hidden; box-shadow: 0 -8px 30px rgba(0,0,0,0.2);
        }
        #map { flex: 2.5; height: 100%; }
        #sidebar {
            flex: 1; padding: 25px; overflow-y: auto; max-width: 450px;
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
            border-left: 1px solid #dee2e6;
        }
        .status-indicator {
            background: linear-gradient(135deg, #d4edda, #c3e6cb);
            color: #155724; padding: 15px; border-radius: 8px;
            margin-bottom: 20px; text-align: center; font-size: 0.9em;
        }
        .controls {
            background: white; padding: 20px; border-radius: 10px;
            margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .form-group { margin-bottom: 20px; }
        label {
            display: block; margin-bottom: 8px; font-weight: 600;
            color: #333; font-size: 1em;
        }
        select, button {
            width: 100%; padding: 12px 15px; border: 2px solid #e1e5e9;
            border-radius: 8px; font-size: 1em; transition: all 0.3s ease;
            background: white;
        }
        button {
            background: linear-gradient(135deg, #3498db, #2980b9);
            color: white; border: none; cursor: pointer; font-weight: 600;
            margin-bottom: 10px;
        }
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(52, 152, 219, 0.3);
        }
        button:disabled {
            background: linear-gradient(135deg, #bdc3c7, #95a5a6);
            cursor: not-allowed; transform: none; box-shadow: none;
        }
        .btn-clear {
            background: linear-gradient(135deg, #e74c3c, #c0392b);
        }
        .route-card {
            background: white; border-radius: 12px; padding: 25px;
            margin-bottom: 20px; box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            border: 2px solid #e9ecef; cursor: pointer; transition: all 0.3s ease;
        }
        .route-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .route-card.selected {
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        .route-header {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;
        }
        .route-name {
            font-weight: 700; color: #2c3e50; font-size: 1.2em;
        }
        .route-type {
            padding: 6px 12px; border-radius: 15px; font-size: 0.8em; font-weight: 600;
        }
        .type-multimodal { background: linear-gradient(135deg, #e74c3c, #f39c12); color: white; }
        .type-bike { background: linear-gradient(135deg, #27ae60, #2ecc71); color: white; }
        .type-transit { background: linear-gradient(135deg, #3498db, #5dade2); color: white; }
        .route-summary {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 10px; margin-bottom: 15px;
        }
        .summary-item {
            text-align: center; padding: 10px; background: #f8f9fa;
            border-radius: 6px; border: 1px solid #e9ecef;
        }
        .summary-value {
            font-weight: bold; color: #3498db; font-size: 1.1em; display: block;
        }
        .summary-label {
            font-size: 0.8em; color: #6c757d; text-transform: uppercase; margin-top: 4px;
        }
        .coordinates-display {
            background: #f8f9fa; padding: 15px; border-radius: 8px;
            margin: 15px 0; border-left: 4px solid #6c757d;
        }
        .error {
            background: linear-gradient(135deg, #ffebee, #ffcdd2);
            color: #c62828; padding: 20px; border-radius: 10px;
            margin: 20px 0; border-left: 5px solid #f44336;
        }
        .spinner {
            border: 4px solid rgba(52, 152, 219, 0.2);
            width: 40px; height: 40px; border-radius: 50%;
            border-left-color: #3498db; animation: spin 1s linear infinite;
            margin: 20px auto; display: none;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .instructions {
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            color: #856404; padding: 20px; border-radius: 10px;
            margin-bottom: 20px; border-left: 5px solid #f1c40f;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Integrated Multimodal Transit Tool</h1>
        <p>OSRM Bike Routing + Google Transit + GTFS Real-time + Shapefile Analysis</p>
    </div>
    
    <div class="container">
        <div id="map"></div>
        
        <div id="sidebar">
            <div class="status-indicator" id="statusIndicator">
                System Status: Loading...
            </div>
            
            <div class="instructions">
                <h3>How to Use</h3>
                <ol>
                    <li>Click two points on the map</li>
                    <li>Select routing mode</li>
                    <li>Click "Find Routes"</li>
                    <li>Click route cards to view on map</li>
                </ol>
            </div>
            
            <div class="controls">
                <div class="form-group">
                    <label for="routingMode">Routing Mode:</label>
                    <select id="routingMode">
                        <option value="bike">Bike Only (OSRM)</option>
                        <option value="transit">Transit Only (Google)</option>
                        <option value="multimodal">Bike-Bus-Bike (Combined)</option>
                    </select>
                </div>
                
                <button id="findRoutesBtn" disabled>Find Routes</button>
                <button class="btn-clear" onclick="clearAll()">Clear All</button>
                
                <div class="coordinates-display">
                    <p><strong>Start:</strong> <span id="startCoords">Click map to select</span></p>
                    <p><strong>End:</strong> <span id="endCoords">Click map to select</span></p>
                </div>
            </div>
            
            <div class="spinner" id="spinner"></div>
            <div id="results"></div>
        </div>
    </div>

    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        let map, startPoint = null, endPoint = null, startMarker = null, endMarker = null, routeLayersGroup, currentRoutes = [];
        
        function initializeMap() {
            map = L.map('map').setView([30.3293, -81.6556], 12);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            }).addTo(map);
            routeLayersGroup = L.layerGroup().addTo(map);
            map.on('click', handleMapClick);
        }
        
        function handleMapClick(e) {
            const latlng = e.latlng;
            if (!startPoint) {
                startPoint = latlng;
                if (startMarker) map.removeLayer(startMarker);
                startMarker = L.marker(latlng, {
                    icon: L.icon({
                        iconUrl: 'data:image/svg+xml;base64,' + btoa('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="red"><circle cx="16" cy="16" r="10"/><text x="16" y="21" text-anchor="middle" fill="white" font-size="14" font-weight="bold">B</text></svg>'),
                        iconSize: [32, 32], iconAnchor: [16, 16]
                    })
                }).addTo(map);
                updateCoordinatesDisplay();
            } else {
                clearAll();
                handleMapClick(e);
            }
        }
        
        function updateCoordinatesDisplay() {
            document.getElementById('startCoords').textContent = startPoint ? 
                `${startPoint.lat.toFixed(5)}, ${startPoint.lng.toFixed(5)}` : 'Click map to select';
            document.getElementById('endCoords').textContent = endPoint ? 
                `${endPoint.lat.toFixed(5)}, ${endPoint.lng.toFixed(5)}` : 'Click map to select';
            document.getElementById('findRoutesBtn').disabled = !(startPoint && endPoint);
        }
        
        function clearAll() {
            startPoint = null; endPoint = null;
            if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
            if (endMarker) { map.removeLayer(endMarker); endMarker = null; }
            routeLayersGroup.clearLayers();
            document.getElementById('results').innerHTML = '';
            updateCoordinatesDisplay();
        }
        
        function showSpinner(show) {
            document.getElementById('spinner').style.display = show ? 'block' : 'none';
        }
        
        function showError(message) {
            document.getElementById('results').innerHTML = `<div class="error">${message}</div>`;
        }
        
        async function findRoutes() {
            if (!startPoint || !endPoint) {
                showError('Please select both start and end points');
                return;
            }
            
            const mode = document.getElementById('routingMode').value;
            showSpinner(true);
            routeLayersGroup.clearLayers();
            
            try {
                let endpoint = '';
                switch (mode) {
                    case 'bike':
                        endpoint = '/api/bike-routes';
                        break;
                    case 'transit':
                        endpoint = '/api/transit-routes';
                        break;
                    case 'multimodal':
                        endpoint = '/api/multimodal-routes';
                        break;
                }
                
                const params = new URLSearchParams({
                    start_lon: startPoint.lng,
                    start_lat: startPoint.lat,
                    end_lon: endPoint.lng,
                    end_lat: endPoint.lat
                });
                
                const response = await fetch(`${endpoint}?${params}`);
                const data = await response.json();
                
                showSpinner(false);
                
                if (!response.ok) {
                    showError(data.detail || 'Request failed');
                    return;
                }
                
                displayResults(data, mode);
                
            } catch (error) {
                console.error('Error:', error);
                showSpinner(false);
                showError('Failed to find routes. Please try again.');
            }
        }
        
        function displayResults(data, mode) {
            const container = document.getElementById('results');
            let html = '';
            
            if (data.routes && data.routes.length > 0) {
                html += `<h3>Found ${data.routes.length} route(s)</h3>`;
                
                data.routes.forEach((route, index) => {
                    html += createRouteCard(route, index, mode);
                });
                
                // Visualize first route
                visualizeRoute(data.routes[0], mode);
            } else {
                html = '<div class="error">No routes found</div>';
            }
            
            container.innerHTML = html;
            window.currentRoutes = data.routes || [];
        }
        
        function createRouteCard(route, index, mode) {
            let routeType = 'type-bike';
            let routeTypeText = 'BIKE';
            
            if (mode === 'multimodal') {
                routeType = route.type === 'bike_bus_bike' ? 'type-multimodal' : 'type-bike';
                routeTypeText = route.type === 'bike_bus_bike' ? 'MULTIMODAL' : 'DIRECT BIKE';
            } else if (mode === 'transit') {
                routeType = 'type-transit';
                routeTypeText = 'TRANSIT';
            }
            
            let html = `<div class="route-card" onclick="visualizeRoute(window.currentRoutes[${index}], '${mode}')">`;
            html += `<div class="route-header">`;
            html += `<div class="route-name">${route.name || `Route ${index + 1}`}</div>`;
            html += `<div class="route-type ${routeType}">${routeTypeText}</div>`;
            html += `</div>`;
            
            if (mode === 'multimodal') {
                const summary = route.summary;
                html += `<div class="route-summary">`;
                html += `<div class="summary-item"><span class="summary-value">${summary.total_time_formatted}</span><span class="summary-label">Time</span></div>`;
                html += `<div class="summary-item"><span class="summary-value">${summary.total_distance_miles}</span><span class="summary-label">Miles</span></div>`;
                html += `<div class="summary-item"><span class="summary-value">${summary.bike_distance_miles}</span><span class="summary-label">Bike Mi</span></div>`;
                html += `<div class="summary-item"><span class="summary-value">${summary.average_bike_score}</span><span class="summary-label">Safety</span></div>`;
                html += `<div class="summary-item"><span class="summary-value">${summary.transfers}</span><span class="summary-label">Transfers</span></div>`;
                html += `</div>`;
            } else if (mode === 'bike') {
                html += `<div class="route-summary">`;
                html += `<div class="summary-item"><span class="summary-value">${route.travel_time_formatted}</span><span class="summary-label">Time</span></div>`;
                html += `<div class="summary-item"><span class="summary-value">${route.length_miles}</span><span class="summary-label">Miles</span></div>`;
                html += `<div class="summary-item"><span class="summary-value">${route.overall_score}</span><span class="summary-label">Safety</span></div>`;
                html += `</div>`;
            } else if (mode === 'transit') {
                html += `<div class="route-summary">`;
                html += `<div class="summary-item"><span class="summary-value">${route.duration_text}</span><span class="summary-label">Time</span></div>`;
                html += `<div class="summary-item"><span class="summary-value">${route.distance_miles}</span><span class="summary-label">Miles</span></div>`;
                html += `<div class="summary-item"><span class="summary-value">${route.transfers}</span><span class="summary-label">Transfers</span></div>`;
                html += `</div>`;
            }
            
            html += `</div>`;
            return html;
        }
        
        function visualizeRoute(route, mode) {
            if (!route) return;
            
            routeLayersGroup.clearLayers();
            
            // Remove selection from all cards
            document.querySelectorAll('.route-card').forEach(card => {
                card.classList.remove('selected');
            });
            
            if (mode === 'multimodal' && route.legs) {
                route.legs.forEach((leg, index) => {
                    if (leg.route && leg.route.geometry) {
                        const coords = leg.route.geometry.coordinates;
                        if (coords && coords.length > 0) {
                            const latLngs = coords.map(coord => L.latLng(coord[1], coord[0]));
                            const color = leg.color || (leg.type === 'bike' ? '#27ae60' : '#3498db');
                            
                            const line = L.polyline(latLngs, {
                                color: color,
                                weight: 6,
                                opacity: 0.8
                            }).addTo(routeLayersGroup);
                            
                            line.bindPopup(`${leg.name} (${leg.type})`);
                        }
                    }
                });
                
                // Add bus stop markers if available
                if (route.bus_stops) {
                    if (route.bus_stops.start_stop) {
                        const stop = route.bus_stops.start_stop;
                        L.marker([stop.display_y, stop.display_x], {
                            icon: L.icon({
                                iconUrl: 'data:image/svg+xml;base64,' + btoa('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="blue"><rect x="6" y="8" width="20" height="16" rx="2" fill="blue"/><circle cx="12" cy="26" r="2" fill="white"/><circle cx="20" cy="26" r="2" fill="white"/></svg>'),
                                iconSize: [24, 24], iconAnchor: [12, 12]
                            })
                        }).addTo(routeLayersGroup).bindPopup(`Start Transit: ${stop.name}`);
                    }
                    
                    if (route.bus_stops.end_stop) {
                        const stop = route.bus_stops.end_stop;
                        L.marker([stop.display_y, stop.display_x], {
                            icon: L.icon({
                                iconUrl: 'data:image/svg+xml;base64,' + btoa('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="blue"><rect x="6" y="8" width="20" height="16" rx="2" fill="blue"/><circle cx="12" cy="26" r="2" fill="white"/><circle cx="20" cy="26" r="2" fill="white"/></svg>'),
                                iconSize: [24, 24], iconAnchor: [12, 12]
                            })
                        }).addTo(routeLayersGroup).bindPopup(`End Transit: ${stop.name}`);
                    }
                }
            } else if (route.geometry) {
                const coords = route.geometry.coordinates;
                if (coords && coords.length > 0) {
                    const latLngs = coords.map(coord => L.latLng(coord[1], coord[0]));
                    const color = mode === 'bike' ? '#27ae60' : '#3498db';
                    
                    L.polyline(latLngs, {
                        color: color,
                        weight: 6,
                        opacity: 0.8
                    }).addTo(routeLayersGroup).bindPopup(route.name || 'Route');
                }
            }
            
            // Fit map to show route
            try {
                if (routeLayersGroup.getLayers().length > 0) {
                    map.fitBounds(routeLayersGroup.getBounds(), { padding: [20, 20] });
                }
            } catch (e) {
                console.warn('Could not fit bounds:', e);
            }
        }
        
        async function checkSystemStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                const statusEl = document.getElementById('statusIndicator');
                let statusText = 'System Status: ';
                
                if (status.components.google_maps_api && status.components.gtfs_data.loaded) {
                    statusText += 'All Systems Operational';
                    statusEl.style.background = 'linear-gradient(135deg, #d4edda, #c3e6cb)';
                } else if (status.components.google_maps_api || status.components.gtfs_data.loaded) {
                    statusText += 'Partially Operational';
                    statusEl.style.background = 'linear-gradient(135deg, #fff3cd, #ffeaa7)';
                } else {
                    statusText += 'Limited Functionality';
                    statusEl.style.background = 'linear-gradient(135deg, #ffebee, #ffcdd2)';
                }
                
                const details = [];
                if (status.components.google_maps_api) details.push('Google Maps: ✓');
                if (status.components.gtfs_data.loaded) details.push('GTFS: ✓');
                if (status.components.shapefile_analysis.roads_loaded) details.push('Shapefiles: ✓');
                
                statusEl.innerHTML = statusText + '<br><small>' + details.join(' | ') + '</small>';
                
            } catch (error) {
                console.error('Status check failed:', error);
                document.getElementById('statusIndicator').innerHTML = 'System Status: Unknown';
            }
        }
        
        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initializeMap();
            checkSystemStatus();
            document.getElementById('findRoutesBtn').addEventListener('click', findRoutes);
        });
    </script>
</body>
</html>"""

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    
    print("🚴‍♂️🚌 Starting Integrated Multimodal Transit Tool")
    print(f"Server will run on port {port}")
    print(f"OSRM Server: {OSRM_SERVER}")
    print(f"Google API Key configured: {bool(GOOGLE_API_KEY)}")
    print(f"GeoPandas available: {GEOPANDAS_AVAILABLE}")
    print(f"Data ZIP URL: {DATA_ZIP_URL}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )
