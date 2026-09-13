"""
backend/geo.py
Geospatial and KML processing module for CanopyLens.
Handles KML boundary parsing, geodesic/projected area calculations,
and raster georeferencing / GSD extraction.
"""

import os
import re
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import transform
import pyproj

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def parse_kml_coordinates(coord_text: str) -> List[Tuple[float, float]]:
    """
    Parses coordinate string from KML into a list of (lon, lat) tuples.
    Format: 'lon,lat,alt lon,lat,alt ...' or separated by whitespace/newlines.
    """
    coords = []
    tokens = re.split(r"[\s\n\r]+", coord_text.strip())
    for token in tokens:
        if not token:
            continue
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                coords.append((lon, lat))
            except ValueError:
                continue
    return coords


def parse_kml(kml_input: Any) -> Dict[str, Any]:
    """
    Parses KML data (file path, file object, bytes, or string).
    Extracts polygons, validates geometry, and computes projected area in m² and hectares.

    Returns:
        dict with keys:
            - success (bool)
            - polygon (Shapely Polygon or None)
            - aoi_area_m2 (float or None)
            - aoi_area_ha (float or None)
            - centroid (tuple of (lon, lat) or None)
            - message (str)
            - placemark_names (list of str)
    """
    result = {
        "success": False,
        "polygon": None,
        "aoi_area_m2": None,
        "aoi_area_ha": None,
        "centroid": None,
        "message": "",
        "placemark_names": []
    }

    try:
        if isinstance(kml_input, (str, Path)) and os.path.exists(str(kml_input)):
            tree = ET.parse(str(kml_input))
            root = tree.getroot()
        elif isinstance(kml_input, bytes):
            root = ET.fromstring(kml_input.decode("utf-8", errors="ignore"))
        elif isinstance(kml_input, str):
            root = ET.fromstring(kml_input)
        elif hasattr(kml_input, "read"):
            content = kml_input.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            root = ET.fromstring(content)
        else:
            result["message"] = "Invalid KML input format."
            return result
    except Exception as e:
        result["message"] = f"Failed to parse KML XML structure: {str(e)}"
        return result

    # Strip XML namespaces for uniform querying
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    polygons = []
    names = []

    # Find all Placemarks
    for placemark in root.iter("Placemark"):
        name_elem = placemark.find("name")
        pm_name = name_elem.text.strip() if name_elem is not None and name_elem.text else "Unnamed AOI"
        
        # Check for Polygon outerBoundaryIs
        for poly_elem in placemark.iter("Polygon"):
            coords_elem = poly_elem.find(".//outerBoundaryIs//coordinates")
            if coords_elem is None or not coords_elem.text:
                coords_elem = poly_elem.find(".//coordinates")
            
            if coords_elem is not None and coords_elem.text:
                coords = parse_kml_coordinates(coords_elem.text)
                if len(coords) >= 3:
                    # Close polygon if not already closed
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    try:
                        poly = Polygon(coords)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if not poly.is_empty:
                            polygons.append(poly)
                            names.append(pm_name)
                    except Exception as e:
                        continue

    # Fallback: search for Polygon outside Placemark
    if not polygons:
        for poly_elem in root.iter("Polygon"):
            coords_elem = poly_elem.find(".//outerBoundaryIs//coordinates")
            if coords_elem is None or not coords_elem.text:
                coords_elem = poly_elem.find(".//coordinates")
            if coords_elem is not None and coords_elem.text:
                coords = parse_kml_coordinates(coords_elem.text)
                if len(coords) >= 3:
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    try:
                        poly = Polygon(coords)
                        if not poly.is_valid:
                            poly = poly.buffer(0)
                        if not poly.is_empty:
                            polygons.append(poly)
                    except Exception:
                        pass

    if not polygons:
        result["message"] = "No valid Polygon or MultiPolygon geometry found in KML."
        return result

    # Combine multiple polygons if present
    if len(polygons) == 1:
        merged_poly = polygons[0]
    else:
        from shapely.ops import unary_union
        merged_poly = unary_union(polygons)

    result["polygon"] = merged_poly
    result["placemark_names"] = names
    centroid = merged_poly.centroid
    result["centroid"] = (centroid.x, centroid.y)

    # Calculate real-world area in square meters using an appropriate projection
    try:
        lon, lat = centroid.x, centroid.y
        # Compute UTM zone based on centroid longitude
        utm_zone = int(math.floor((lon + 180) / 6) + 1)
        epsg_code = 32600 + utm_zone if lat >= 0 else 32700 + utm_zone
        
        project = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True).transform
        projected_poly = transform(project, merged_poly)
        area_m2 = float(projected_poly.area)
        area_ha = area_m2 / 10000.0

        result["aoi_area_m2"] = area_m2
        result["aoi_area_ha"] = area_ha
        result["success"] = True
        result["message"] = f"Parsed {len(polygons)} boundary polygon(s) successfully."
    except Exception as e:
        # Fallback to World Equal Area EPSG:6933 if local UTM projection errors
        try:
            project = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True).transform
            projected_poly = transform(project, merged_poly)
            area_m2 = float(projected_poly.area)
            result["aoi_area_m2"] = area_m2
            result["aoi_area_ha"] = area_m2 / 10000.0
            result["success"] = True
            result["message"] = "Parsed KML boundary using global equal-area projection."
        except Exception as e2:
            result["success"] = True
            result["message"] = f"Parsed KML geometry, but projected area calculation failed: {e2}"

    return result


def extract_geotiff_metadata(image_path: str) -> Dict[str, Any]:
    """
    Extracts spatial metadata and Ground Sampling Distance (GSD) from a GeoTIFF if available.

    Returns:
        dict with:
            - is_georeferenced (bool)
            - crs (str or None)
            - gsd_m (float or None): Ground Sampling Distance in meters/pixel
            - bounds (tuple or None): (minx, miny, maxx, maxy) in native CRS
            - bounds_wgs84 (tuple or None): (min_lon, min_lat, max_lon, max_lat)
            - width (int)
            - height (int)
            - message (str)
    """
    meta = {
        "is_georeferenced": False,
        "crs": None,
        "gsd_m": None,
        "bounds": None,
        "bounds_wgs84": None,
        "width": 0,
        "height": 0,
        "message": "Standard raster format (no geospatial headers detected)."
    }

    if not RASTERIO_AVAILABLE:
        meta["message"] = "rasterio library not installed; cannot inspect GeoTIFF tags."
        return meta

    ext = Path(image_path).suffix.lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        meta["message"] = f"Standard {ext.upper().replace('.', '')} raster format (no geospatial headers available)."
        return meta

    try:
        with rasterio.open(image_path) as ds:
            meta["width"] = ds.width
            meta["height"] = ds.height
            if ds.crs is not None and not ds.transform.is_identity:
                meta["crs"] = str(ds.crs)
                meta["bounds"] = (ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top)

                # Determine pixel size in meters
                res_x, res_y = abs(ds.transform[0]), abs(ds.transform[4])

                if ds.crs.is_projected:
                    # Projected CRS (meters)
                    computed_gsd = float((res_x + res_y) / 2.0)
                    if computed_gsd > 0 and not math.isnan(computed_gsd):
                        meta["gsd_m"] = computed_gsd
                        meta["is_georeferenced"] = True
                        meta["message"] = f"Georeferenced projected raster detected (GSD: {meta['gsd_m']:.3f} m/px, CRS: {ds.crs})."
                    else:
                        meta["message"] = "Projected CRS detected but pixel resolution is invalid."
                else:
                    # Geographic CRS (degrees) - approximate conversion at image center latitude
                    center_lat = (ds.bounds.bottom + ds.bounds.top) / 2.0
                    meters_per_deg_lat = 111320.0
                    meters_per_deg_lon = 111320.0 * math.cos(math.radians(center_lat))
                    gsd_x_m = res_x * meters_per_deg_lon
                    gsd_y_m = res_y * meters_per_deg_lat
                    computed_gsd = float((gsd_x_m + gsd_y_m) / 2.0)
                    if computed_gsd > 0 and not math.isnan(computed_gsd):
                        meta["gsd_m"] = computed_gsd
                        meta["is_georeferenced"] = True
                        meta["message"] = f"Geographic raster converted to meters (Approx GSD: {meta['gsd_m']:.3f} m/px, CRS: {ds.crs})."
                    else:
                        meta["message"] = "Geographic CRS detected but ground resolution could not be computed."

                # Transform bounds to WGS84 if needed
                try:
                    from rasterio.warp import transform_bounds
                    if ds.crs != "EPSG:4326":
                        b_wgs84 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
                        meta["bounds_wgs84"] = b_wgs84
                    else:
                        meta["bounds_wgs84"] = meta["bounds"]
                except Exception:
                    pass
            else:
                meta["message"] = "GeoTIFF does not contain valid embedded CRS or spatial transform tags."
    except Exception as e:
        meta["message"] = f"Standard non-georeferenced image ({type(e).__name__})."

    return meta


def validate_image_kml_alignment(
    image_meta: Dict[str, Any],
    kml_data: Optional[Dict[str, Any]]
) -> Tuple[bool, str]:
    """
    Evaluates spatial relationship between uploaded image and KML boundary.

    Returns:
        (aligned: bool, explanation: str)
    """
    if not kml_data or not kml_data.get("success") or kml_data.get("polygon") is None:
        return False, "Area of Interest boundary not provided."

    if not image_meta.get("is_georeferenced"):
        return False, (
            "KML boundary was successfully parsed, but the uploaded image does not contain "
            "sufficient geospatial metadata (CRS & spatial extent) to guarantee pixel-to-boundary alignment."
        )

    image_bounds_wgs84 = image_meta.get("bounds_wgs84")
    if not image_bounds_wgs84:
        return False, "Image geospatial bounds could not be transformed to WGS84 for spatial verification."

    from shapely.geometry import box
    img_box = box(*image_bounds_wgs84)
    kml_poly = kml_data["polygon"]

    if img_box.intersects(kml_poly):
        return True, "Image spatial extent overlaps with the provided KML Area of Interest."
    else:
        return False, "The georeferenced image and KML boundary do not spatially overlap."
