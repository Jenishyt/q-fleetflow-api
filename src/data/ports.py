"""Real port reference data - actual UN/LOCODEs and coordinates, used by
the map, ports database, and route geometry endpoints. Not fabricated;
these are real, stable geographic facts."""

PORTS = [
    {"name": "Chennai", "country": "India", "locode": "INMAA", "lat": 13.0827, "lon": 80.2707},
    {"name": "Mumbai", "country": "India", "locode": "INBOM", "lat": 19.0760, "lon": 72.8777},
    {"name": "Cochin", "country": "India", "locode": "INCOK", "lat": 9.9312, "lon": 76.2673},
    {"name": "Visakhapatnam", "country": "India", "locode": "INVTZ", "lat": 17.6868, "lon": 83.2185},
    {"name": "Colombo", "country": "Sri Lanka", "locode": "LKCMB", "lat": 6.9271, "lon": 79.8612},
    {"name": "Singapore", "country": "Singapore", "locode": "SGSIN", "lat": 1.3521, "lon": 103.8198},
    {"name": "Jebel Ali (Dubai)", "country": "UAE", "locode": "AEJEA", "lat": 25.0118, "lon": 55.0618},
    {"name": "Rotterdam", "country": "Netherlands", "locode": "NLRTM", "lat": 51.9244, "lon": 4.4777},
    {"name": "Shanghai", "country": "China", "locode": "CNSHA", "lat": 31.2304, "lon": 121.4737},
    {"name": "Tokyo", "country": "Japan", "locode": "JPTYO", "lat": 35.6762, "lon": 139.6503},
    {"name": "Hamburg", "country": "Germany", "locode": "DEHAM", "lat": 53.5511, "lon": 9.9937},
    {"name": "Los Angeles", "country": "United States", "locode": "USLAX", "lat": 33.7405, "lon": -118.2668},
    {"name": "Santos", "country": "Brazil", "locode": "BRSSZ", "lat": -23.9608, "lon": -46.3336},
]

PORTS_BY_NAME = {p["name"]: p for p in PORTS}
