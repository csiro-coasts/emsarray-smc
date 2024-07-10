import dataclasses
from collections.abc import Hashable
from enum import Enum
from functools import cached_property
from typing import Sequence

import numpy as np
import shapely
import xarray as xr
from emsarray import utils
from emsarray.conventions import DimensionConvention, Specificity
from emsarray.exceptions import ConventionViolationError
from emsarray.types import Pathish
from shapely.geometry.base import BaseGeometry


class SMCGridKind(str, Enum):
    cell = 'cell'


SMCIndex = tuple[SMCGridKind, int]


@dataclasses.dataclass
class SMCTopology:
    dataset: xr.Dataset

    #: The name of the global attribute that names the cell dimension
    cell_dimension_attribute = 'SMC_grid_type'

    def __init__(
        self,
        dataset: xr.Dataset,
        *,
        cell_dimension: Hashable | None = None,
        longitude: Hashable | None = None,
        latitude: Hashable | None = None,
        longitude_cell_size_factor: Hashable | None = None,
        latitude_cell_size_factor: Hashable | None = None,
    ):
        """
        Construct a new :class:`CFGridTopology` instance for a dataset.

        By default this will introspect the dataset
        looking for longitude and latitude coordinate variables,
        and lookinf for longitude and latitude cell size variables.
        The ``longitude``, ``latitude``,
        ``longitude_cell_size``, and ``latitude_cell_size`` parameters
        allow you to manually specify the correct variable names
        if the automatic detection fails.
        """
        self.dataset = dataset
        if cell_dimension is not None:
            self.cell_dimension = cell_dimension
        if longitude is not None:
            self.longitude_name = longitude
        if latitude is not None:
            self.latitude_name = latitude
        if longitude_cell_size_factor is not None:
            self.longitude_cell_size_factor_name = longitude_cell_size_factor
        if latitude_cell_size_factor is not None:
            self.latitude_cell_size_factor_name = latitude_cell_size_factor

    @cached_property
    def cell_dimension(self) -> Hashable:
        """The dimension name that indexes each cell"""
        try:
            name = self.dataset.attrs[self.cell_dimension_attribute]
        except KeyError:
            raise ConventionViolationError(
                "Global cell dimension attribute "
                f"{self.cell_dimension_attribute!r} not set!")
        if name not in self.dataset.dims:
            raise ConventionViolationError(
                f"Cell dimension {name!r} does not exist in the dataset!")
        return name

    @property
    def cell_count(self) -> int:
        """The size of the cell dimension"""
        return self.dataset.sizes[self.cell_dimension]

    @cached_property
    def longitude_name(self) -> Hashable:
        """
        The name of the longitude coordinate variable.
        Found by looking for a variable with either a
        ``standard_name = "longitude"`` or
        ``units = "degree_east"``
        attribute.
        """
        try:
            return next(
                name for name, variable in self.dataset.variables.items()
                if variable.attrs.get('standard_name') == 'longitude'
                or variable.attrs.get('coordinate_type') == 'longitude'
                or variable.attrs.get('units') == 'degree_east'
            )
        except StopIteration:
            raise ValueError("Could not find longitude coordinate")

    @cached_property
    def latitude_name(self) -> Hashable:
        """
        The name of the latitude coordinate variable.
        Found by looking for a variable with either a
        ``standard_name = "latitude"`` or
        ``units = "degree_north"``
        attribute.
        """
        try:
            return next(
                name
                for name, variable in self.dataset.variables.items()
                if variable.attrs.get('standard_name') == 'latitude'
                or variable.attrs.get('coordinate_type') == 'latitude'
                or variable.attrs.get('units') == 'degree_north'
            )
        except StopIteration:
            raise ValueError("Could not find latitude coordinate")

    @cached_property
    def longitude_cell_size_factor_name(self) -> Hashable:
        """
        The name of the longitude cell size variable.
        Found by looking for a variable with a
        ``long_name = "longitude cell size factor"`` attribute.
        """
        try:
            return next(
                name for name, variable in self.dataset.variables.items()
                if variable.attrs.get('long_name') == 'longitude cell size factor'
            )
        except StopIteration:
            raise ValueError("Could not find longitude cell size variable")

    @cached_property
    def latitude_cell_size_factor_name(self) -> Hashable:
        """
        The name of the latitude cell size variable.
        Found by looking for a variable with a
        ``long_name = "latitude cell size factor"`` attribute.
        """
        try:
            return next(
                name for name, variable in self.dataset.variables.items()
                if variable.attrs.get('long_name') == 'latitude cell size factor'
            )
        except StopIteration:
            raise ValueError("Could not find latitude cell size variable")

    @property
    def longitude(self) -> xr.DataArray:
        """The longitude coordinate variable"""
        return self.dataset[self.longitude_name]

    @property
    def latitude(self) -> xr.DataArray:
        """The latitude coordinate variable"""
        return self.dataset[self.latitude_name]

    @property
    def longitude_cell_size_factor(self) -> xr.DataArray:
        """The longitude cell size variable"""
        return self.dataset[self.longitude_cell_size_factor_name]

    @property
    def latitude_cell_size_factor(self) -> xr.DataArray:
        """The latitude cell size variable"""
        return self.dataset[self.latitude_cell_size_factor_name]


class SMC(DimensionConvention[SMCGridKind, SMCIndex]):
    """
    Spherical multiple-cell (SMC) datasets consist of non overlapping,
    axis aligned, rectangular cells of varying sizes.
    Smaller cells are used where increased resolution is desired
    (e.g. around coastlines).
    Cells are indexed by the :attr:`SMC.cell_dimension` dimension,
    which is a one-dimensional index in an arbitrary order.
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        *,
        topology: SMCTopology | None = None,
    ):
        super().__init__(dataset)
        if topology is not None:
            self.topology = topology

    @classmethod
    def check_dataset(cls, dataset: xr.Dataset) -> int | None:
        # The following dataset attributes are required to identify this as an
        # SMC dataset
        required_attrs = [
            'base_lat_size', 'base_lon_size',
            'southernmost_latitude',
            'northernmost_latitude',
            'westernmost_longitude',
            'easternmost_longitude',
            SMCTopology.cell_dimension_attribute,
        ]

        if not all(attr in dataset.attrs for attr in required_attrs):
            return None

        return Specificity.HIGH

    @cached_property
    def topology(self) -> SMCTopology:
        return SMCTopology(self.dataset)

    @cached_property
    def grid_kinds(self) -> frozenset[SMCGridKind]:
        return frozenset(SMCGridKind)

    @cached_property
    def grid_dimensions(self) -> dict[SMCGridKind, Sequence[Hashable]]:
        return {SMCGridKind.cell: (self.topology.cell_dimension,)}

    @cached_property
    def default_grid_kind(self) -> SMCGridKind:
        return SMCGridKind.cell

    def unpack_index(self, index: SMCIndex) -> tuple[SMCGridKind, Sequence[int]]:
        return SMCGridKind.cell, [index[1]]

    def pack_index(self, grid_kind: SMCGridKind, indexes: Sequence[int]) -> SMCIndex:
        return (grid_kind, indexes[0])

    def get_all_geometry_names(self) -> list[Hashable]:
        return [
            self.topology.longitude_name,
            self.topology.latitude_name,
            self.topology.longitude_cell_size_factor_name,
            self.topology.latitude_cell_size_factor_name,
        ]

    def drop_geometry(self) -> xr.Dataset:
        # Drop geometry variables
        dataset = super().drop_geometry()

        # Drop geometry attributes
        required_attrs = [
            'base_lat_size', 'base_lon_size',
            'southernmost_latitude',
            'northernmost_latitude',
            'westernmost_longitude',
            'easternmost_longitude',
            'SMC_grid_type',
        ]
        for key in sorted(dataset.attrs.keys() & required_attrs):
            del dataset.attrs[key]

        return dataset

    @cached_property
    @utils.timed_func
    def polygons(self) -> np.ndarray:
        """
        SMC polygons are lat/lon boxes centred at a point, with a size given by
        cx/cy and the base cell size.
        """
        # In the SMC datasets I have seen, the step values have all been
        # exactly representable as floats - i.e. a power of two.
        # The following calculations are exact because of this.
        # If any SMC datasets are encountered that do _not_ use an exactly
        # representable power of two this will have to be modified.
        lon_size = float(self.dataset.attrs['base_lon_size'])
        lat_size = float(self.dataset.attrs['base_lat_size'])

        lons = self.topology.longitude.values
        lats = self.topology.latitude.values
        cx = self.topology.longitude_cell_size_factor.values
        cy = self.topology.latitude_cell_size_factor.values

        # Cells have size (cx * lon_size, cy * lat_size),
        # centred at (longitde, latitude)
        lon_cell_size = lon_size * cx / 2
        lat_cell_size = lat_size * cy / 2
        lon_min = lons - lon_cell_size
        lon_max = lons + lon_cell_size
        lat_min = lats - lat_cell_size
        lat_max = lats + lat_cell_size

        # points is an array of shape (cell_count, 5, 2),
        # where each row is a set of five points defining the cell polygon.
        points = np.array([
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ], dtype=lons.dtype)
        points = np.transpose(points, (2, 0, 1))

        return shapely.polygons(points)

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xr.Dataset:
        if buffer > 0:
            raise ValueError("Buffering SMC datasets is not yet implemented")
        included_indexes = [
            index
            for index in self.strtree.query(clip_geometry)
            if self.polygons[index].intersects(clip_geometry)
            and not self.polygons[index].touches(clip_geometry)
        ]
        mask = np.zeros(self.topology.cell_count, dtype=bool)
        mask[included_indexes] = True

        return xr.Dataset(
            data_vars={
                'mask': xr.DataArray(
                    data=mask,
                    dims=[self.topology.cell_dimension],
                ),
            },
        )

    def apply_clip_mask(self, clip_mask: xr.Dataset, work_dir: Pathish) -> xr.Dataset:
        return self.dataset.isel({self.topology.cell_dimension: clip_mask['mask'].values})
