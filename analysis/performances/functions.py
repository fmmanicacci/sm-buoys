"""This package contains the functions used to analysis the performance of our system."""

# +
# | DEPENDENCIES
# +

from collections import namedtuple
from datetime import timedelta
from typing import Any, Dict, NamedTuple, Optional, Tuple

import gpxpy as gpx
import numpy as np
import pandas as pd
from folium import Circle, Map, PolyLine
from folium.plugins import PolyLineTextPath

# +
# | CUSTOM TYPES
# +

Location = namedtuple('Location', ['latitude', 'longitude'])

# +
# | CONSTANTS
# +

DFLT_PW_OPTIONS = {
    'repeat': True,
    'offset': 4,
    'attributes': {'font-size': '10', 'fill': 'black'}
    }

DFLT_FP_OPTIONS = {
    'radius': 10,
    'weight': 1,
    'color': 'black', 
    'fill_color': 'black', 
    'fill': True, 
    'fill_opacity': 1.0, 
    'opacity':1.0
}

# +
# | FUNCTIONS
# +

def from_gpx(
    filepath:str, 
    track:int = 0, 
    segment:int = 0, 
    index:Optional[str] = None
    ) -> pd.DataFrame:
    """Extract and load GPS data points contained in a GPX file into a Pandas DataFrame object.
    
    Parameters
    ----------
    filepath : str
        The filename and path of the GPX file.
    track : int
        The index for the track from which data will be extracted.
    segment : int
        The index of the segment contained in the track from which data will be extracted.
    index : str, optional
        The name of the column used for indexing

    Returns
    -------
    DataFrame
        A Pandas DataFrame object containing the GPS data points.
    """
    # [1. Parsing and extracting data from the given GPX file
    #  -> The file only contains one track with one segment.
    with open(filepath, 'r') as gpx_file:
        gpx_content = gpx.parse(gpx_file)
        gpx_points = gpx_content.tracks[track].segments[segment].points

    # [2. Building and filling a dictionnary from the GPX point previously extrated
    #     in order to build the pandas DataFrame object.
    entries = {"latitude": [], "longitude": [], "elevation": [], "datetime": [],}
    for point in gpx_points:
        entries["latitude"].append(point.latitude)
        entries["longitude"].append(point.longitude)
        entries["elevation"].append(point.elevation)
        entries["datetime"].append(point.time)

    # [3. Finally building the pandas DataFrame object and set the index if necessary.
    df = pd.DataFrame.from_dict(entries)
    if index is not None:
        df = df.set_index(index)

    return df

def build_map(
    data:pd.DataFrame,
    location:Tuple[float, float], 
    pathway:Optional[pd.DataFrame]=None,
    fixed_points:Optional[pd.DataFrame]=None,
    bounded:bool=False,
    data_options:Dict[str, Any]={},
    pw_options:Dict[str, Any]=DFLT_PW_OPTIONS,
    fp_options:Dict[str, Any]=DFLT_FP_OPTIONS,
    **kwargs:Dict[str, Any]
    ) -> Map:
    """Build and return a Folium map printing the given data points as circle.
    
    Parameters
    ----------
    data : DataFrame
        The Pandas DataFrame containing the main data points to display on the map.
    location : Tuple[float, float]
        The location at which the map is centered.
    pathway : DataFrame, optional
        A Pandas DataFrame containing GPS data points used to draw a pathway as polylines with
        arrows indicating the direction.
    fixed_points : DataFrame, optional
        A Pandas DataFrame containing GPS data points used to draw fixed objects as circle.
    bounded : bool
        If true, fit the map to contain a bounding box with the maximum zoom level possible.
    data_options : Dict[str, Any]
        Drawing option for the main data points.
    pw_options : Dict[str, Any]
        Drawing option for the pathway.
    fp_options : Dict[str, Any]
        Drawing option for the fixed data points.
    **kwargs : Dict[str, Any]
        Additional optional sent to Folium Map object.

    Returns
    -------
    Folium.Map
        The map containing all given data.
    """
    # [1. Building the map
    map = Map(location=location, **kwargs)

    # [2. Add fixed points, if defined
    if fixed_points is not None:
        for _, row in fixed_points.iterrows():
            tooltip = f"({row.latitude}, {row.longitude}) - {row.name}"
            Circle(location=[row.latitude, row.longitude], tooltip=tooltip, **fp_options).add_to(map)

    # [3. Add pathway, if defined
    if pathway is not None:
        locations = [(row.latitude, row.longitude) for _, row in pathway.iterrows()]
        polyline = PolyLine(locations, color='#565352').add_to(map)
        PolyLineTextPath(
            polyline,
            '\u25BA', 
            **pw_options
            ).add_to(map)

    # [4. Add the data points
    for _, row in data.iterrows():
        tooltip = f"({row.latitude}, {row.longitude}): {row.name}"
        Circle(location=[row.latitude, row.longitude], tooltip=tooltip, **data_options).add_to(map)

    # [5. Setting map's boundaries, if necessary
    if bounded:
        map.fit_bounds(map.get_bounds())

    return map

def remove_intervals(df:pd.DataFrame, time_intervals:Tuple[str, str]) -> pd.DataFrame:
    """Remove data contained in the given intervals.
    
    Notes
    -----
    For each time interval, the starting time is included and the ending time is excluded.

    Parameters
    ----------
    df : DataFrame
        The DataFrame object from which data points contained in the intervals will be removed.
    time_interval : Tuple[str, str]
        The time intervals
    
    Returns
    -------
    DataFrame
        The new DataFrame object which no longer contained the data points contained in the
        given time intervals.
    """
    # [1. Concatenate labels to remove
    to_remove = []
    for start, end in time_intervals:
        to_remove += df.loc[(df.index >= start) & (df.index < end)].index.tolist()
    # [2. Drop labels and return the result
    return df.drop(labels=to_remove)

def haversine_distance(point1:pd.Series, point2:pd.Series) -> float:
    """Compute distance between two GPS points.
    
    Notes
    -----
    The code comes from: https://www.movable-type.co.uk/scripts/latlong.html
    """
    R = 6371e3
    
    phi1 = point1.latitude * np.pi / 180
    phi2 = point2.latitude * np.pi / 180
    
    deltaPhi    = (point2.latitude - point1.latitude) * np.pi / 180
    deltaLambda = (point2.longitude - point1.longitude) * np.pi / 180
    
    a = np.sin(0.5 * deltaPhi)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(0.5 * deltaLambda)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def space_method(
    emitter_df:pd.DataFrame, 
    boat_df:pd.DataFrame, 
    reference_point: Location,
    dt:int=15) -> pd.DataFrame:
    """Compute the difference in space and time between emitter and boat GPS point using the space based method.
    
    Parameters
    ----------
    emitter_df : DataFrame
        A DataFrame object containing the GPS data points of the acoustic emitter.
    boat_df : DataFrame
        A DataFrame object containing the GPS data points of the boat.
    dt : int
        Time in seconds used to restrict the search for the corresponding boat's GPS data points.

    Returns
    -------
    DataFrame
        A DataFrame object containing the space and time difference with the nearest boat's GPS data
        point for every emitter's GPS data points.
    """
    computations = {'datetime': [], 'distance': [], 'space_diff': [], 'time_diff':[], 'method': []}
    
    # [1. For each GPS point from the emitter:
    #  -> Find the nearest GPS point of the boat within a narrow time interval
    #  -> Compute and find the minimal distance
    for idx, emitter_row in emitter_df.iterrows():
        # [1.1 Extract datetime and compute the time interval
        date_time = emitter_row.name
        low, high = date_time - timedelta(seconds=dt), date_time + timedelta(seconds=dt)
        
        # [1.2 Use the time interval to restrict the search of potential GPS points
        #  -> Because the pathway describe by the boat has intersection, time restriction
        #     has to be applied to ensure coherent GPS points are considered for comparaison.
        candidates = boat_df.loc[(boat_df.index >= low) & (boat_df.index < high)]
        empty = len(candidates) == 0

        # [1.3 For each GPS point, compute the haversine distance and the time difference between
        #      the emitter point and the nearest GPS point found in the time interval.
        if not empty:
            distances = candidates.apply(haversine_distance, args=(emitter_row,), axis=1)
            nearest_dist = distances.min()
            nearest_idx = distances.idxmin()
            time_diff = abs(idx - nearest_idx).total_seconds()
        else:
            nearest_dist = time_diff = None

        # [1.4 Compute the distance between the emitter's location and the reference point.
        distance = haversine_distance(emitter_row, reference_point)

        # [1.5 Record the result
        computations['datetime'].append(idx)
        computations['distance'].append(distance)
        computations['space_diff'].append(nearest_dist)
        computations['time_diff'].append(time_diff)
        computations['method'].append('space')

    # [2. Build and return Pandas Series object
    result_df = pd.DataFrame.from_dict(computations)
    result_df = result_df.set_index("datetime")
    return result_df

def time_method(
    emitter_df:pd.DataFrame, 
    boat_df:pd.DataFrame, 
    reference_point: Location,
    dt:int=15) -> pd.DataFrame:
    """Compute the difference in space and time between emitter and boat GPS point using the time based method.
    
    Parameters
    ----------
    emitter_df : DataFrame
        A DataFrame object containing the GPS data points of the acoustic emitter.
    boat_df : DataFrame
        A DataFrame object containing the GPS data points of the boat.
    dt : int
        Time in seconds used to restrict the search for the corresponding boat's GPS data points.

    Returns
    -------
    DataFrame
        A DataFrame object containing the space and time difference with the nearest boat's GPS data
        point for every emitter's GPS data points.
    """
    computations = {'datetime': [], 'distance': [], 'space_diff': [], 'time_diff':[], 'method': []}
    
    # [1. For each GPS point from the emitter:
    #  -> Find the nearest GPS point of the boat in time within a narrow time interval
    #  -> Compute and find the distance
    for idx, emitter_row in emitter_df.iterrows():
        # [1.1 Extract datetime and compute the time interval
        date_time = emitter_row.name
        low, high = date_time - timedelta(seconds=dt), date_time + timedelta(seconds=dt)
        
        # [1.2 Use the time interval to restrict the search of potential GPS points
        #  -> Because between the nearest data point in time and emitter point may be
        #     too large.
        candidates = boat_df.loc[(boat_df.index >= low) & (boat_df.index < high)]
        empty = len(candidates) == 0

        # [1.3 For each GPS point, compute the haversine distance and the time difference between
        #      the emitter point and the nearest GPS point found in the time interval.
        if not empty:
            time_differences = candidates.apply(lambda boat_row: abs(boat_row.name - date_time).total_seconds(), axis=1)
            time_diff = time_differences.min()
            time_idx = time_differences.idxmin()
            space_diff = haversine_distance(emitter_row, boat_df.loc[time_idx])
        else:
            time_diff = space_diff = None

        # [1.4 Compute the distance between the emitter's location and the reference point.
        distance = haversine_distance(emitter_row, reference_point)

        # [1.5 Record the result
        computations['datetime'].append(idx)
        computations['distance'].append(distance)
        computations['space_diff'].append(space_diff)
        computations['time_diff'].append(time_diff)
        computations['method'].append('time')
        
    # [2. Build and return Pandas Series object
    result_df = pd.DataFrame.from_dict(computations)
    result_df = result_df.set_index("datetime")
    return result_df

def space_time_method(
    emitter_df:pd.DataFrame,
    boat_df:pd.DataFrame,
    reference_point: Location, 
    dt:int=15) -> pd.DataFrame:
    """Compute the difference in space and time between emitter and boat GPS point using the time based method.
    
    Parameters
    ----------
    emitter_df : DataFrame
        A DataFrame object containing the GPS data points of the acoustic emitter.
    boat_df : DataFrame
        A DataFrame object containing the GPS data points of the boat.
    dt : int
        Time in seconds used to restrict the search for the corresponding boat's GPS data points.

    Returns
    -------
    DataFrame
        A DataFrame object containing the space and time difference with the nearest boat's GPS data
        point for every emitter's GPS data points.
    """
    computations = {'datetime': [], 'distance': [], 'space_diff': [], 'time_diff':[], 'method': []}
    
    # [1. For each GPS point from the emitter:
    #  -> Find the nearest GPS point of the boat in time within a narrow time interval
    #  -> Compute and find the distance
    for idx, emitter_row in emitter_df.iterrows():
        # [1.1 Extract datetime and compute the time interval
        date_time = emitter_row.name
        low, high = date_time - timedelta(seconds=dt), date_time + timedelta(seconds=dt)
        
        # [1.2 Use the time interval to restrict the search of potential GPS points
        #  -> Because between the nearest data point in time and emitter point may be
        #     too large.
        candidates = boat_df.loc[(boat_df.index >= low) & (boat_df.index < high)]
        empty = len(candidates) == 0

        # [1.3 For each GPS point, compute the haversine distance and the time difference between
        #      the emitter point and the nearest GPS point found in the time interval.
        if not empty:
            time_diffs = candidates.apply(lambda boat_row: abs(boat_row.name - date_time).total_seconds(), axis=1)
            space_diffs = candidates.apply(lambda boat_row: haversine_distance(emitter_row, boat_row), axis=1)
            euclidian_dist = np.sqrt(time_diffs**2 + space_diffs**2)
            min_idx = euclidian_dist.idxmin()
            space_diff = space_diffs.loc[min_idx]
            time_diff = time_diffs.loc[min_idx]
        else:
            space_diff = time_diff = None

        # [1.4 Compute the distance between the emitter's location and the reference point.
        distance = haversine_distance(emitter_row, reference_point)

        # [1.5 Record the result
        computations['datetime'].append(idx)
        computations['distance'].append(distance)
        computations['space_diff'].append(space_diff)
        computations['time_diff'].append(time_diff)
        computations['method'].append('space_time')

    # [2. Build and return Pandas Series object
    result_df = pd.DataFrame.from_dict(computations)
    result_df = result_df.set_index("datetime")
    return result_df