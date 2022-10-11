from enum import Enum
from functools import cached_property
from typing import Dict, List, Optional, Tuple

import emsarray
import numpy as np
import xarray as xr
from emsarray.formats._helpers import Specificity
from emsarray.types import Pathish
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


class SMCGridKind(str, Enum):
    seapoint = 'seapoint'


SMCIndex = Tuple[SMCGridKind, int]


class SMC(emsarray.Format[SMCGridKind, int]):
    """
    SMC datasets consist of non overlapping, axis aligned, rectangular cells
    of varying sizes.
    Smaller cells are used where increased resolution is desired
    (e.g. around coastlines).
    Cells are indexed by the ``seapoint`` dimension,
    which is a one-dimensional index in an arbitrary order.
    """
    #: The dimension name for each point
    seapoint_dimension: str = 'seapoint'

    @property
    def seapoint_count(self) -> int:
        return self.dataset.dims[self.seapoint_dimension]

    @classmethod
    def check_dataset(cls, dataset: xr.Dataset) -> Optional[int]:
        # The following dataset attributes are required to identify this as an
        # SMC dataset
        required_attrs = [
            'base_lat_size', 'base_lon_size',
            'southernmost_latitude',
            'northernmost_latitude',
            'westernmost_longitude',
            'easternmost_longitude',
            'SMC_grid_type',
        ]

        if not all(attr in dataset.attrs for attr in required_attrs):
            return None

        return Specificity.HIGH

    def ravel_index(self, index: SMCIndex) -> int:
        _kind, linear_index = index
        return linear_index

    def unravel_index(
        self,
        linear_index: int,
        grid_kind: Optional[SMCGridKind] = None,
    ) -> SMCIndex:
        return (SMCGridKind.seapoint, linear_index)

    @property
    def grid_kinds(self) -> List[SMCGridKind]:
        return list(SMCGridKind)

    @property
    def default_grid_kind(self) -> SMCGridKind:
        return SMCGridKind.seapoint

    def get_grid_kind_and_size(
        self, data_array: xr.DataArray,
    ) -> Tuple[SMCGridKind, int]:
        if self.seapoint_dimension not in data_array.dims:
            raise ValueError("Unknown grid kind")
        return (SMCGridKind.seapoint, self.seapoint_count)

    def make_linear(self, data_array: xr.DataArray) -> xr.DataArray:
        kind, size = self.get_grid_kind_and_size(data_array)
        if kind is not SMCGridKind.seapoint:
            raise ValueError("Unknown grid kind")

        # The dataset is already linear
        return data_array

    def selector_for_index(self, index: SMCIndex) -> Dict[str, int]:
        _kind, linear_index = index
        return {self.seapoint_dimension: linear_index}

    def drop_geometry(self) -> xr.Dataset:
        # Drop geometry variables
        dataset = self.dataset.drop_vars([
            'longitude', 'latitude', 'cx', 'cy',
        ])

        # Drop geometry attributes
        required_attrs = [
            'base_lat_size', 'base_lon_size',
            'southernmost_latitude',
            'northernmost_latitude',
            'westernmost_longitude',
            'easternmost_longitude',
            'SMC_grid_type',
        ]
        for key in (dataset.attrs.keys() & required_attrs):
            del dataset[key]

        return dataset

    @cached_property
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

        lons = self.dataset['longitude'].values
        lats = self.dataset['latitude'].values
        cx = self.dataset['cx'].values
        cy = self.dataset['cy'].values

        # Cells have size (cx * lon_size, cy * lat_size),
        # centred at (longitde, latitude)
        lon_cell_size = lon_size * cx / 2
        lat_cell_size = lat_size * cy / 2
        lon_min = lons - lon_cell_size
        lon_max = lons + lon_cell_size
        lat_min = lats - lat_cell_size
        lat_max = lats + lat_cell_size

        # points is an array of shape (seapoint_count, 5, 2),
        # where each row is a set of five points defining the cell polygon.
        points = np.array([
            [lon_min, lat_min],
            [lon_max, lat_min],
            [lon_max, lat_max],
            [lon_min, lat_max],
            [lon_min, lat_min],
        ], dtype=lons.dtype)
        points = np.transpose(points, (2, 0, 1))

        # Construct a polygon per row
        return np.array([Polygon(row) for row in points])

    def make_clip_mask(
        self,
        clip_geometry: BaseGeometry,
        buffer: int = 0,
    ) -> xr.Dataset:
        if buffer > 0:
            raise ValueError("Buffering SMC datasets is not yet implemented")

        # Construct this outside the timed section, as it is timed also
        spatial_index = self.spatial_index

        included_indices = [
            hit.linear_index
            for hit in spatial_index.query(clip_geometry)
            if hit.polygon.intersects(clip_geometry)
        ]
        mask = np.zeros(self.seapoint_count, dtype=bool)
        mask[included_indices] = True

        return xr.Dataset(
            data_vars={
                'mask': xr.DataArray(
                    data=mask,
                    dims=[self.seapoint_dimension],
                ),
            },
        )

    def apply_clip_mask(self, clip_mask: xr.Dataset, work_dir: Pathish) -> xr.Dataset:
        return self.dataset.isel({self.seapoint_dimension: clip_mask['mask'].values})
