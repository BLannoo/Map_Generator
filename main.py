from pathlib import Path

from map_generator import backend_switch as np
from map_generator.globals import REPO_ROOT
from map_generator.imaging_functions import normalize
from map_generator.parameters import load_params, Params, create_landscape


def run(params: Params, autosave=True, jax_available = np.jax_available) -> None:
    root_params = params.root_params
    my_landscape = create_landscape(root_params.world)
    mountainsca = root_params.world.mountain_heights
    riversca = root_params.world.river_scale
    X = params.X
    Y = params.Y
    zoom = root_params.imaging.zoom
    res = root_params.imaging.resolution
    tiling = root_params.imaging.tiling
    tile_size = root_params.imaging.tile_size
    min_pos = root_params.min_pos()
    max_pos = root_params.max_pos()
    neg_octaves = int(np.log2(my_landscape.lin_sca)-4)

    print(f"{(1000 * (max_pos - min_pos) / res):.2f} meters per pixel at zoom {zoom}")
    import time
    start_time = time.time()
    if tile_size > 0 and tile_size <= res: # if tile_size is set and less than resolution, use tiling
        import tqdm
        num_tiles = res // tile_size # number of tiles in each axis
        # X and Y are already split into tiles in the create_mesh_grid function, so we can just iterate over them
        Z = [np.zeros((tile_size * num_tiles, tile_size)) for _ in range(num_tiles)]
        base = [np.zeros((tile_size * num_tiles, tile_size)) for _ in range(num_tiles)]
        mountains = [np.zeros((tile_size * num_tiles, tile_size)) for _ in range(num_tiles)]
        river_z = [np.zeros((tile_size * num_tiles, tile_size)) for _ in range(num_tiles)]
        secondary = [np.zeros((tile_size * num_tiles, tile_size)) for _ in range(num_tiles)]

        if jax_available:
            import jax
            height_func = jax.jit(lambda X_i,Y_i: my_landscape.get_height(
                X_i, Y_i, offs=0.5, fine_offs=1.0, 
                mountainsca=mountainsca, neg_octave = neg_octaves,
                riversca=riversca, rivernoise=0.4)
                ).lower(X[0,0], Y[0,0]).compile()
        else:
            height_func = my_landscape.get_height
        for i in tqdm.tqdm(range(num_tiles), desc="Generating map tiles"):
            base_i = [np.zeros((tile_size, tile_size)) for _ in range(num_tiles)]
            mountains_i = [np.zeros((tile_size, tile_size)) for _ in range(num_tiles)]
            Z_i = [np.zeros((tile_size, tile_size)) for _ in range(num_tiles)]
            river_z_i = [np.zeros((tile_size, tile_size)) for _ in range(num_tiles)]
            secondary_i = [np.zeros((tile_size, tile_size)) for _ in range(num_tiles)]
            for j in range(num_tiles):
                base_i[j], mountains_i[j], Z_i[j], river_z_i[j], secondary_i[j] = height_func(X[i,j], Y[i,j])
            base[i] = np.concatenate(base_i, axis=1)
            mountains[i] = np.concatenate(mountains_i, axis=1)
            Z[i] = np.concatenate(Z_i, axis=1)
            river_z[i] = np.concatenate(river_z_i, axis=1)
            secondary[i] = np.concatenate(secondary_i, axis=1)
        base = np.concatenate(base, axis=0)
        mountains = np.concatenate(mountains, axis=0)
        Z = np.concatenate(Z, axis=0)
        river_z = np.concatenate(river_z, axis=0)
        secondary = np.concatenate(secondary, axis=0)

            
    elif tiling > 1:
        import tqdm
        print("Warning: tiling is deprecated and will be removed in a future version. Please use tile_size instead.")
        print(f"Tiling the map into {tiling} parts for memory management")
        X = np.array_split(X, tiling)
        Y = np.array_split(Y, tiling)
        Z = [np.zeros_like(X[i]) for i in range(tiling)]
        base = [np.zeros_like(X[i]) for i in range(tiling)]
        mountains = [np.zeros_like(X[i]) for i in range(tiling)]
        river_z = [np.zeros_like(X[i]) for i in range(tiling)]
        secondary = [np.zeros_like(X[i]) for i in range(tiling)]
        if jax_available:
            import jax
            height_func = jax.jit(lambda X_i,Y_i: my_landscape.get_height(
                X_i, Y_i, offs=0.5, fine_offs=1.0, 
                mountainsca=mountainsca, neg_octave = neg_octaves,
                riversca=riversca, rivernoise=0.4)
                ).lower(X[0], Y[0]).compile()
        else:
            height_func = my_landscape.get_height
        for i in tqdm.tqdm(range(tiling), desc="Generating map sections"):
            base[i], mountains[i], Z[i], river_z[i], secondary[i] = height_func(X[i], Y[i])# my_landscape.get_height(X[i], Y[i], offs=0.5, fine_offs=1.0, mountainsca=mountainsca, riversca=riversca, rivernoise=0.4, neg_octave = neg_octaves)
        Z = np.concatenate(Z, axis=0)
        base = np.concatenate(base, axis=0)
        mountains = np.concatenate(mountains, axis=0)
        river_z = np.concatenate(river_z, axis=0)
        secondary = np.concatenate(secondary, axis=0)
    else:
        if jax_available:
            import jax
            height_func = jax.jit(lambda X,Y: my_landscape.get_height(
                X, Y, offs=0.5, fine_offs=1.0, 
                mountainsca=mountainsca, neg_octave = neg_octaves,
                riversca=riversca, rivernoise=0.4)
                ).lower(X, Y).compile()
            base, mountains, Z, river_z, secondary = height_func(X, Y)
        else:
            height_func = my_landscape.get_height
        base, mountains, Z, river_z, secondary = height_func(X,Y) # my_landscape.get_height(X, Y, offs=0.5, fine_offs=1.0, mountainsca=mountainsca, riversca=riversca, rivernoise=0.4)#, octaves=2,neg_octaves=0, fade=0.5,voron=True,ndims=1)
    print(f"Time taken to generate the map: {time.time() - start_time:.2f} seconds")
    print(np.min(Z))
    print(np.max(Z))
    if autosave:
        Z = normalize(Z, root_params.timestamped_output_folder)
    else:
        Z = normalize(Z, output_folder=None)
    return base, mountains, Z, river_z, secondary


def main(
        input_folder: Path = REPO_ROOT / "params/default",
        output_folder: Path = REPO_ROOT / "output",
):
    params = load_params(input_folder=input_folder, output_folder=output_folder)
    _,_,_,_,_ = run(params)
    params.root_params.save()

if __name__ == "__main__":
    main()
