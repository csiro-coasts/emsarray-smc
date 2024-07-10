import pathlib

import numpy as np
import pytest
import pandas
import xarray
import xarray.testing
from emsarray.conventions import Specificity
from emsarray.conventions._registry import registry
from shapely.geometry import Polygon, box
import shapely.testing

from emsarray_smc import SMC, SMCGridKind

example_smc_file = pathlib.Path('./smc-australia.nc')


def test_convention_registered():
    assert SMC in registry.entry_point_conventions


def test_check_dataset_match(datasets):
    assert SMC.check_dataset(xarray.open_dataset(datasets / 'smc.nc')) is Specificity.HIGH


def test_open_dataset(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    assert isinstance(ds.ems, SMC)


def test_check_dataset_no_match(
    datasets: pathlib.Path,
):
    # Make a basic 1D CF grid dataset
    ds = xarray.Dataset(
        data_vars={'value': (['y', 'x'], np.arange(100).reshape(10, 10))},
        coords={
            'x': ('x', np.arange(10), {'standard_name': 'longitude'}),
            'y': ('y', np.arange(10), {'standard_name': 'latitude'}),
        },
    )

    assert SMC.check_dataset(ds) is None


def test_cell_dimension_autodetect(
    datasets: pathlib.Path,
):
    """
    The name of the cell dimension should be taken from the `SMC_grid_type`
    global attribute.
    """
    ds = xarray.Dataset(
        data_vars={'values': ('smc_dimension', np.arange(10))},
        attrs={'SMC_grid_type': 'smc_dimension'},
    )
    convention = SMC(ds)
    assert convention.topology.cell_dimension == 'smc_dimension'
    assert convention.topology.cell_count == 10


def test_ravel_index(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    assert ds.ems.ravel_index((SMCGridKind.cell, 10)) == 10


def test_wind_index(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    assert ds.ems.wind_index(10) == (SMCGridKind.cell, 10)


def test_grid_kinds(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    assert ds.ems.grid_kinds == frozenset([SMCGridKind.cell])


def test_default_grid_kind(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    assert ds.ems.default_grid_kind is SMCGridKind.cell


def test_get_grid_kind_and_size(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    assert ds.ems.get_grid_kind_and_size(ds['foo']) \
        == (SMCGridKind.cell, 20103)


def test_get_grid_kind_and_size_no_match(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    with pytest.raises(ValueError):
        ds.ems.get_grid_kind_and_size(ds['time'])


def test_ravel(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    xarray.testing.assert_equal(
        ds.ems.ravel(ds['foo']),
        xarray.DataArray(ds['foo'].values, dims=['index']),
    )


def test_ravel_no_match(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    with pytest.raises(ValueError):
        ds.ems.ravel(ds['time'])


def test_get_selector_for_index(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    selector = ds.ems.selector_for_index((SMCGridKind, 20))
    assert selector == {ds.ems.topology.cell_dimension: 20}


def test_drop_geometry(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    topology = ds.ems.topology

    dropped_ds = ds.ems.drop_geometry()
    assert topology.cell_dimension in dropped_ds.dims
    assert topology.longitude_name not in dropped_ds.variables
    assert topology.latitude_name not in dropped_ds.variables
    assert topology.longitude_cell_size_factor_name not in dropped_ds.variables
    assert topology.latitude_cell_size_factor_name not in dropped_ds.variables


def test_polygons(
    datasets: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    polygons = ds.ems.polygons
    assert len(polygons) == ds.ems.topology.cell_count
    assert isinstance(polygons[0], Polygon)
    shapely.testing.assert_geometries_equal(
        polygons[0],
        Polygon([[100, -50], [102, -50], [102, -48], [100, -48]]))
    shapely.testing.assert_geometries_equal(
        polygons[-1],
        Polygon([[163, -7], [164, -7], [164, -6], [163, -6]]))


def test_clip(
    datasets: pathlib.Path,
    tmp_path: pathlib.Path,
):
    ds = xarray.open_dataset(datasets / 'smc.nc')
    tasmania = box(140, -44, 150, -39)
    clipped = ds.ems.clip(tasmania, work_dir=tmp_path)
    clipped.load()
    clipped.to_netcdf(tmp_path / 'clipped.nc')
    assert isinstance(clipped.ems, SMC)
    assert clipped.ems.topology.cell_count < ds.ems.topology.cell_count
    # TODO Test this geometry makes sense
