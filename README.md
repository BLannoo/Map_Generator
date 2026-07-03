# Map_Generator
 World Generator for use in gaming projects
Generates heightfields for terrain maps. Heightfields are continuous and fully procedural. 
A demo can be run on google colab at: https://colab.research.google.com/github/EddieMcT/Map_Generator/blob/main/Parallelised_world_demo.ipynb 


Heightfields entirely made in parallel and all calculations should be vectorised. This means that every step in the calculations is fully deterministic, using ​basic arithmetic and a lot of fractal noise, with no simulations or random processes involved after setting initial parameters. Each point is calculated entirely independently of the others, and any number of maps can be made of a world at different zoom levels and locations without discontinuities.

This calculation starts with an approximation of tectonic plates as voronoi cells, samples using jittered coordinates to create fractal coastlines, then blends between these to create boundaries and mountain range, before using a new implementation of Dendry noise (Gaillard ​et al.​ 2019) to add rivers.

Currently the script (main.py) operates by reading parameters from world_parameters.txt and imaging_parameters.txt before sampling a grid of coordinates. Future plans include adding a GUI, and allowing for different sampling methods (ie spirals or other uneven patterns with baked-in LOD), gpu implementations with cupy, and assorted design improvements.

Heightfield outputs are automatically saved in a 0-1 range, in order to make optimal use of bit depth in game engines. For now, values to undo normalisation are stored with image names in order to allow automated corrections.
