#!/usr/bin/env python3

import collections
import pathlib
from typing import Iterable

import cartopy.io.shapereader as shapereader
import numpy
import pandas
import xarray
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

HERE = pathlib.Path(__file__).parent


def get_intersecting_shapes(region: BaseGeometry) -> Iterable[BaseGeometry]:
    path = shapereader.gshhs('h', 1)
    for geometry in shapereader.Reader(path).geometries():
        if region.intersects(geometry):
            yield geometry


def floor_multiple(num, multiple):
    return int(num // multiple) * multiple


def main():
    output = HERE / 'smc.nc'

    # The name of the smc cell dimension
    dimension_name = 'cell'

    # The minimum size of a cell, in degrees
    base_lon_size, base_lat_size = 2 ** -3, 2 ** -3

    # The maximum span of a cell. Must be a power of two.
    max_lon_size_factor, max_lat_size_factor = 2 ** 4, 2 ** 4

    # Generate a grid that cover this region of the world
    bounds = (100, -50, 164, -5)
    region = box(*bounds)

    # Find all land polygons in this region
    land_polygons = numpy.array(list(get_intersecting_shapes(region)), dtype=numpy.object_)
    spatial_index = STRtree(land_polygons)

    # The number of total cells spanning the longitude and latitude of the region
    base_lon_count = int((bounds[2] - bounds[0]) // base_lon_size)
    base_lat_count = int((bounds[3] - bounds[1]) // base_lat_size)

    # A double-resolution grid of points within the region
    base_lon_grid = numpy.linspace(bounds[0], bounds[2], base_lon_count * 2 + 1)
    base_lat_grid = numpy.linspace(bounds[1], bounds[3], base_lat_count * 2 + 1)

    # Make a queue of possible cells to add.
    # A possible cell is made up of the bottom left index of the cell
    # and its size factor.
    # Each possible cell is popped from the queue,
    # and either added to the dataset, subdivided, or discarded.
    possible_cells = collections.deque(
        (
            x0, y0,
            min(max_lon_size_factor, base_lon_count - x0),
            min(max_lat_size_factor, base_lat_count - y0),
        )
        for y0 in numpy.arange(0, floor_multiple(base_lat_count, max_lat_size_factor), max_lat_size_factor)
        for x0 in numpy.arange(0, floor_multiple(base_lon_count, max_lon_size_factor), max_lon_size_factor)
    )
    good_cells = []

    while possible_cells:
        possible_cell = possible_cells.popleft()
        (x0, y0, cx, cy) = possible_cell
        lon_min, lon_max = base_lon_grid[x0 * 2], base_lon_grid[(x0 + cx) * 2]
        lat_min, lat_max = base_lat_grid[y0 * 2], base_lat_grid[(y0 + cy) * 2]
        polygon = box(lon_min, lat_min, lon_max, lat_max)

        # Find all land polygons that are near this polygon.
        # Near means within one cx/cy step of the polygon,
        # which should give higher resolution around shores
        # without needing to intersect the land
        add_cell = False
        close_polygons = land_polygons[
            spatial_index.query(
                polygon, predicate='dwithin',
                distance=max(cx * base_lon_size, cy * base_lat_size)
            )
        ]
        if close_polygons.size == 0:
            # Polygon is no where near land! Add it to the good cells.
            add_cell = True

        elif any(i.covers(polygon) for i in close_polygons):
            # Discard this cell entirely, as it is completely covered by land
            pass

        elif cx > 1 and cy > 1:
            # Subdivide it and add the quarters to the queue,
            # if the cell is not at the minimum size already
            cx = cx // 2
            cy = cy // 2
            possible_cells.extendleft([
                (x0 + cx, y0 + cy, cx, cy),
                (x0, y0 + cy, cx, cy),
                (x0 + cx, y0, cx, cy),
                (x0, y0, cx, cy),
            ])
        else:
            # Add this cell. It is close to (and possibly intersecting) land,
            # but isn't covered entirely by land,
            # but can't be subdivided further.
            add_cell = True

        if add_cell:
            lon = base_lon_grid[x0 * 2 + cx]
            lat = base_lat_grid[y0 * 2 + cy]
            good_cells.append((lon, lat, cx, cy))

    good_cells_df = pandas.DataFrame(
        good_cells, columns=['longitude', 'latitude', 'cx', 'cy'])
    days = 10
    dataset = xarray.Dataset(
        data_vars={
            'cx': xarray.DataArray(
                good_cells_df['cx'],
                dims=[dimension_name],
                attrs={
                    'long_name': 'longitude cell size factor',
                },
            ),
            'cy': xarray.DataArray(
                good_cells_df['cy'],
                dims=[dimension_name],
                attrs={
                    'long_name': 'latitude cell size factor',
                },
            ),
            'foo': xarray.DataArray(
                numpy.arange(len(good_cells)),
                dims=[dimension_name],
                attrs={
                    'long_name': 'Some arbitrary index',
                },
            ),
            'bar': xarray.DataArray(
                numpy.arange(len(good_cells) * days).reshape(days, -1),
                dims=['time', dimension_name],
                attrs={'long_name': 'Some other arbitrary value'},
            ),
        },
        coords={
            'longitude': xarray.DataArray(
                good_cells_df['longitude'],
                dims=[dimension_name],
                attrs={'standard_name': 'longitude'},
            ),
            'latitude': xarray.DataArray(
                good_cells_df['latitude'],
                dims=[dimension_name],
                attrs={'standard_name': 'latitude'},
            ),
            'time': xarray.DataArray(
                pandas.date_range("2023/01/01", periods=days, freq="D"),
                dims=['time'],
                attrs={'standard_name': 'time'},
            ),
        },
        attrs={
            'base_lon_size': base_lon_size,
            'base_lat_size': base_lat_size,
            'westernmost_longitude': bounds[0],
            'southernmost_latitude': bounds[1],
            'easternmost_longitude': bounds[2],
            'northernmost_latitude': bounds[3],
            'SMC_grid_type': dimension_name,
        },
    )

    dataset.to_netcdf(output)


if __name__ == '__main__':
    main()
