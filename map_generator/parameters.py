from dataclasses import dataclass
from typing import Any
import dataclasses
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import yaml

from map_generator.globals import REPO_ROOT
from map_generator.landscape import landscape_gen
from map_generator import backend_switch as np


def from_dict_helper(cls, data : dict[str , Any]):
    valid_field_names = [f.name for f in dataclasses.fields(cls)]
    filtered_data = {k:v for k, v in data.items() if k in valid_field_names}
    return cls(**filtered_data)

@dataclass(frozen=True)
class WorldParams:
    world_size: int  # Linear scale of the world in km
    mountain_heights: float   # Height of mountains in km 0..1
    river_scale: float   # Height of rivers in m / Dendry height (0..1000)
    centroids: Optional[List[List[float]]] = None  # plate centroids as relative [[x, y], ...] both x and y in 0..1
    heights_tectonic_plates: Optional[List[float]] = None  # list with values between [-1..1]
    slopes_x: Optional[List[float]] = None  # km across world size
    slopes_y: Optional[List[float]] = None  # km across world size
    river_density: Optional[int] = None  # major branching points per world size in Dendry noise

    @staticmethod
    def from_yaml(path: str | Path) -> "WorldParams":
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        return WorldParams.from_dict(data)
    @staticmethod
    def from_dict(data: dict) -> "WorldParams":
        return from_dict_helper(cls=WorldParams, data=data)
        # filtered_data = {k:v for k, v in data.items() if k in [f.name for f in dataclasses.fields(WorldParams)]}
        # return WorldParams(**filtered_data)


@dataclass(frozen=True)
class ImagingParams:
    offset_x: float  # Offset of the map from a world center
    offset_y: float
    zoom: float  # Relative zoom, 1 = whole world without boundaries
    resolution: int  # resolution in each direction
    tiling: int = -1 # number of splits for memory management (2 means 2 tiles total)
    tile_size : int = 0 # resolution of each tile, to be used when rendering the image in chunks. If <1, generate the whole image at once

    @staticmethod
    def from_yaml(path: str | Path) -> "ImagingParams":
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        return ImagingParams.from_dict(data)
    @staticmethod
    def from_dict(data: dict) -> "ImagingParams":
        return from_dict_helper(cls=ImagingParams, data=data)
        # return ImagingParams(**data)

def to_yaml(obj: WorldParams | ImagingParams, path: str | Path) -> None:
    payload = dataclasses.asdict(obj)
    # Use a custom SafeDumper that writes lists in flow (inline) style but keeps mappings in block style
    class _FlowSeqDumper(yaml.SafeDumper):
        pass
    def _repr_flow_seq(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    _FlowSeqDumper.add_representer(list, _repr_flow_seq)
    text = yaml.dump(payload, Dumper=_FlowSeqDumper, sort_keys=False)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding='utf-8')


@dataclass(frozen=True)
class RootParams:
    world: WorldParams
    imaging: Optional[ImagingParams] = None
    output_folder: Path = None
    timestamp: datetime = datetime.now()

    @property
    def timestamped_output_folder(self) -> Path:
        if self.output_folder is None:
            raise ValueError("No output_folder defined")
        else:
            return self.output_folder / self.timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    def min_pos(self) -> float:
        return self.world.world_size // 2 - self.world.world_size // (2 * self.imaging.zoom)

    def max_pos(self) -> float:
        return self.world.world_size // 2 + self.world.world_size // (2 * self.imaging.zoom)

    def create_mesh_grid(self):
        if self.imaging is None:
            raise ValueError("Imaging parameters required to generate a mesh_grid")
        x = np.linspace(self.min_pos(), self.max_pos(), self.imaging.resolution)
        y = np.linspace(self.min_pos(), self.max_pos(), self.imaging.resolution)
        x += self.imaging.offset_x
        y += self.imaging.offset_y

        X, Y = np.meshgrid(x, y)
        if self.imaging.tile_size > 0: # split the mesh grid into tiles of size tile_size x tile_size
            num_tiles = self.imaging.resolution // self.imaging.tile_size # number of tiles in each axis
            effective_res = num_tiles * self.imaging.tile_size
            if effective_res != self.imaging.resolution:
                print(f"Warning: resolution {self.imaging.resolution} is not a multiple of tile_size {self.imaging.tile_size}. Upsampling to {effective_res}.")
                x = np.linspace(self.min_pos(), self.max_pos(), effective_res)
                y = np.linspace(self.min_pos(), self.max_pos(), effective_res)
                x += self.imaging.offset_x
                y += self.imaging.offset_y
                X, Y = np.meshgrid(x, y)
            X = X.reshape((num_tiles, self.imaging.tile_size, num_tiles, self.imaging.tile_size)).swapaxes(1,2)
            Y = Y.reshape((num_tiles, self.imaging.tile_size, num_tiles, self.imaging.tile_size)).swapaxes(1,2)
        return X, Y

    def save(self) -> None:
        to_yaml(self.world, self.timestamped_output_folder / "world_params.yaml")
        to_yaml(self.imaging, self.timestamped_output_folder / "imaging_params.yaml")

@dataclass(frozen=True)
class Params:
    root_params: RootParams
    X: np.ndarray
    Y: np.ndarray

def create_landscape(world: WorldParams) -> landscape_gen:
    wc = world

    # TODO figure out what it means for these to be None and maybe move the default values to WorldConfig dataclass or default.yaml or fallback.yaml
    if wc.centroids is not None and wc.heights_tectonic_plates is not None and wc.slopes_x is not None and wc.slopes_y is not None:
        num_plates = max(1, len(wc.centroids) - 8)
        lg = landscape_gen(wc.world_size, wc.world_size, num_plates=num_plates, boundaries=False)

        # TODO: figure out why these are not part of the constructor
        lg.centroids = np.asarray(wc.centroids) * wc.world_size
        lg.heights = np.asarray(wc.heights_tectonic_plates)
        lg.slopes_x = np.asarray(wc.slopes_x)
        lg.slopes_y = np.asarray(wc.slopes_y)
        if wc.river_density is not None:
            lg.river_density = int(wc.river_density)
    else:
        lg = landscape_gen(wc.world_size, wc.world_size, num_plates=15, boundaries=True)

        # TODO: figure out why this one is not part of the constructor
        lg.river_density = 10
    return lg


def load_params(
        input_folder: Path,
        world_params_filename: str = "world_params.yaml",
        imaging_params_filename: str = "imaging_params.yaml",
        sampling_coords_filename: str = "sampling_coordinates.csv",
        output_folder: Path = None,
):
    world_params_path = input_folder / world_params_filename
    imaging_params_path = input_folder / imaging_params_filename
    sampling_coords_path = input_folder / sampling_coords_filename

    if not world_params_path.exists():
        # fallback_world_params = REPO_ROOT / "params/fallback-old/world_params.yaml"
        fallback_world_params = REPO_ROOT / "params/fallback/world_params.yaml"
        world_params = WorldParams.from_yaml(fallback_world_params)
    else:
        world_params = WorldParams.from_yaml(world_params_path)

    if not sampling_coords_path.exists():  # create sampling coordinates in a grid
        # Set params for image generation
        if not imaging_params_path.exists():
            fallback_imaging_params = REPO_ROOT / "params/fallback/imaging_params.yaml"
            imaging_params = ImagingParams.from_yaml(fallback_imaging_params)
        else:
            imaging_params = ImagingParams.from_yaml(imaging_params_path)

        root_params = RootParams(
            world=world_params,
            imaging=imaging_params,
            output_folder=output_folder,
        )

        X, Y = root_params.create_mesh_grid()
        params = Params(root_params, X, Y)

    else:  # Load (arbitrarily shaped) sampling coordinates
        root_params = RootParams(
            world=world_params,
            imaging=None,
            output_folder=output_folder,
        )

        import pandas as pd
        coords = pd.read_csv(sampling_coords_path)
        X = coords['x'].values
        Y = coords['y'].values

        params = Params(root_params, X, Y)

    return params
