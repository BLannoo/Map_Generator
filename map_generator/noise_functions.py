from pathlib import Path
import map_generator.backend_switch as np
from map_generator.backend_switch import named_scope, scoped, fine_scope
import math
# from scipy.interpolate import CubicSpline
import gc

from map_generator.globals import REPO_ROOT

# =========================
# Helper functions (to move to another file?)
# =========================

@scoped
def generate_spline_points(x1, y1, x2, y2, n_points=10, delta=0.05, method = 'count', control_start_x = None, control_start_y = None, control_end_x = None, control_end_y = None):
    """
    Generate points along a spline between two points.
    For now, this is implemented as a simple linear interpolation (placeholder).
    """
    if method == 'count': #TODO implement other methods (e.g., based on distance)
        t = np.linspace(0, 1, n_points)
    spline_x = x1[...,np.newaxis] + t * (x2[...,np.newaxis] - x1[...,np.newaxis])
    spline_y = y1[...,np.newaxis] + t * (y2[...,np.newaxis] - y1[...,np.newaxis])

    return spline_x, spline_y

@scoped
def find_nearest_points(x, y, candidate_x, candidate_y):
    #Given query points (x, y) (of any shape) and a set of candidate points per query point (candidate_x, candidate_y),
    #returns nearest tree point coordinates and square distances.
    
    for _ in range(x.ndim-2):  # Add dimensions for broadcasting in the case that input query points have many points per coordinate
        candidate_x = candidate_x[..., np.newaxis, :]
        candidate_y = candidate_y[..., np.newaxis, :]

    x_dist = x[...,np.newaxis] - candidate_x
    y_dist = y[...,np.newaxis] - candidate_y
    sq_distances = np.add(x_dist**2 , y_dist**2)
    nearest_idx = np.argmin(sq_distances, axis=-1)

    return nearest_idx, sq_distances**0.5

@scoped
def bezier_points(t, p0, p1, p2, p3):
    # For an array of input control points and an array of values of t, computes locations along cubic Bezier curve for each t value for each set of control points
    # Control points are assumed to be of the same (arbitrary) shape (e.g., 3D vectors), eg (a,b,..., n)
    # t is assumed to be one-dimensional array or list (e.g., np.linspace(0,1,num_points))
    # output is of shape (a,b,...,n,num_points) where n is the number of dimensions for each control point (e.g., 3 for 3D vectors)
    with named_scope("t_powers"):
        t3 = np.power(t,3)
        t2 = np.power(t,2)
        t0 = np.ones_like(t)
    with named_scope("cubic_polynomial"):
        out = ((-p0 + 3*p1 - 3*p2 + p3)[..., np.newaxis]*t3
            + (p0 - 2*p1 + p2)[..., np.newaxis]*t2*3
            + (p1 -p0 )[..., np.newaxis] *t*3
            + p0[..., np.newaxis]*t0)
    return out
@scoped
def bezier_tangents(t, p0, p1, p2, p3):
    # For an array of input control points and an array of values of t, computes directions along cubic Bezier curve for each t value for each set of control points
    # Control points are assumed to be of the same (arbitrary) shape (e.g., 3D vectors), eg (a,b,..., n)
    # t is assumed to be one-dimensional array or list (e.g., np.linspace(0,1,num_points))
    # output is of shape (a,b,...,n,num_points) where n is the number of dimensions for each control point (e.g., 3 for 3D vectors)
    with named_scope("t_powers"):
        t2 = np.power(t,2)
        t0 = np.ones_like(t)
    with named_scope("derivative_polynomial"):
        out = ((-p0 + 3*p1 - 3*p2 + p3)[..., np.newaxis]*t2*3 # Polynomial form
            + (p0 - 2*p1 + p2)[..., np.newaxis]*t*6
            + (p1 -p0)[..., np.newaxis]*t0*3
            )
    return out

@scoped
def subdivide_bezier(p0, p1, p2, p3, num_segments=2, mode = "cubic"):
    # Given start and end points of a bezier curve with control points in between, subdivides it into num_points segments.
    # Returns series of coordinates of nodes for each segment as well as the direction of the curve at that point to be used later control points.
    
    t = np.linspace(0, 1, num_segments+1)
    scale_factor = 1/ (3*num_segments)

    if mode == 'linear':
        points = p0[..., np.newaxis]*(1-t) + p3[..., np.newaxis]*t
        tangents = p1[..., np.newaxis]*(1-t) + p2[..., np.newaxis]*t
    elif mode == 'cubic':
        points = bezier_points(t, p0, p1, p2, p3)
        # tangent directions for each new point (in the forward direction)
        tangents = bezier_tangents(t, p0, p1, p2, p3)

    start_points = points[..., :-1]
    end_points = points[..., 1:]
    start_control =  start_points + tangents[..., :-1]* scale_factor
    end_control = end_points - tangents[..., 1:]* scale_factor 

    return(start_points, end_points, start_control, end_control)



@scoped
def index_within_subgrid(val, z, w=None, offset_z = 0, offset_w = 0, mask_value = -9999):
    # for a 4D array of x,y,z,w (ie a subgrid per point in array) take the values indicated by z (in the last dimension) and w (in the second to last dimension)
    # val is therefore candidate values, and z*w indicate the correct candidate per xy point
    # Shape of val is (a,b,c,d), shape of z, w, and output = (a,b,e,f)
    # for a 3d array of x,y,z take the values indicated by z (in the last dimension) only
    # When the indices are not valid (ie outside the size of val), they should be masked out with a value that is not in val (like -9999). This can then be used to mask invalid indices.

    # Create two placeholder arrays, values = 0:a and 0:b, to index the first two (outer) axes and allow for fancy indexing
    with fine_scope("outer_index_grids"):
        index_1, index_2 = np.ogrid[:val.shape[0], :val.shape[1]]
    #Project these along the later axes to match output shape
    if w is not None: # 4 dimensional case
        with fine_scope("broadcast_indices"):
            index_1 = np.broadcast_to(index_1[:,:,np.newaxis, np.newaxis], z.shape)
            index_2 = np.broadcast_to(index_2[:,:,np.newaxis, np.newaxis], z.shape)
        with fine_scope("masked_gather"):
            return ( np.where( ((z >= -offset_z) & (w >= -offset_w)) & ((z < (val.shape[3]-offset_z)) & (w < (val.shape[2]-offset_w))),
                          val[index_1, index_2, np.minimum(val.shape[2]-1, np.maximum(0,w+offset_w)),  np.minimum(val.shape[3]-1, np.maximum(0,z+offset_z))], # Needed to made indexing possible even in cases where its replaced with the mask value
                          mask_value))

    else: # 3 dimensional case
        with fine_scope("broadcast_indices"):
            index_1 = np.broadcast_to(index_1[:,:,np.newaxis], z.shape)
            index_2 = np.broadcast_to(index_2[:,:,np.newaxis], z.shape)
        with fine_scope("masked_gather"):
            return (np.where( (z < (val.shape[2]-offset_z)) & (z >= -offset_z), val[index_1, index_2,np.minimum(val.shape[2]-1, np.maximum(0,z+offset_z))], mask_value))

@scoped
def check_inner_grid(c = None):
    # For each of the inner 3*3 grid points in a 5*5 matrix, find which of its eight neighbours has the lowest value of c
    # If no value of c is provided, use distance?
    # Expected shape of  c is (X, Y, n_points, n_points)
    # Shape of output is (X, Y, n_points-2, n_points-2) because we exclude the outer two rows and columns from consideration

    # Find the c values of neighbours for each point being calculated
    # shape=(X,Y,n_points-2, n_points-2, 3,3)
    # originally created using a sliding window, but changing to indexing/stacking to allow for more robust ducktyping options
    # neighbour_c = np.lib.stride_tricks.sliding_window_view(c, (3, 3), axis=(-2, -1))
    neighbour_c = np.stack([
        np.stack([
            c[:,:, i:c.shape[-2]-(2-i), j:c.shape[-1]-(2-j)] 
            for i in range(3)
        ], axis=-1) for j in range(3)
    ], axis=-1)

    # Take the index of the lowest neighbour in the n_points-2 by n_points-2 grid
    neighbour_x = np.argmin(neighbour_c, axis=-1) 
    min_x = np.min(neighbour_c, axis=-1)
    neighbour_y = np.argmin(min_x, axis=-1) 
    # Take the neighbour_yth entry for each of neighbour_x entries
    neighbour_x = np.take_along_axis(neighbour_x, neighbour_y[:,:,:, :, None], axis=-1).squeeze()

    # This gives the index of the lowest neighbour in the 3*3 grid for each point being calculated
    # Correct to indices within original 5*5 grid:
    indices = np.indices(neighbour_x.shape)
    neighbour_x += indices[-1]
    neighbour_y += indices[-2]
    
    return neighbour_x, neighbour_y

def inneficient_flatten(data : np.ndarray, shape_target = None):
    while len(data.shape) > 3:
        data = data.reshape(*data.shape[:-2], -1)
    return data

@scoped
def closest_point_on_lines(px, py, x0, y0, x1, y1):
    """
    For each pixel (X,Y), computes all pairwise closest points between N query points and M line segments.

    Parameters
    __________
    px, py: (X, Y, N)
        Input points to query from
    x0, y0, x1, y1: (X, Y, M)
        Start and end points of the lines to examine
    upres : int
        number of cubic splines to split the curves into before approximating each segment as a line. If 1, a straight line between start and end is used (no conversion to a curve)
    x0c, y0c, x1c, y1c: (X,Y,M)
        Optional, control points to use when subsampling bezier curves. Required if upres > 1
    Returns
    _______
        t: (X, Y, N, M) - fraction along each segment
    """
    
    # Expand dims for broadcasting: px,py: (X,Y,N,1), x0,y0,x1,y1: (X,Y,1,M)
    with fine_scope("expand_dims"):
        px = px[..., :, np.newaxis]
        py = py[..., :, np.newaxis]
        x0 = x0[..., np.newaxis, :]
        y0 = y0[..., np.newaxis, :]
        x1 = x1[..., np.newaxis, :]
        y1 = y1[..., np.newaxis, :]

    with fine_scope("project_onto_segments"):
        dx = x1 - x0
        dy = y1 - y0
        dpx = px - x0
        dpy = py - y0

        denom = dx**2 + dy**2 + 1e-12  # avoid division by zero
        t = (dpx * dx + dpy * dy) / denom
        t = np.clip(t, 0, 1)

    return t

@scoped
def sample_pixelwise_bezier_at_t(t, p0, p1, p2, p3):
    """
    Vectorized Bézier sampler for per-pixel, per-curve, per-point t.
    For an image of shape m*n, with N points sampled per curve and M curves defined per pixel.

    Parameters
    __________
    t
        Array of shape (m, n, N, M)
        Control points to use at each curve. m & n are resolution of the image
    p0, p1, p2, p3
        Arrays of shape (m, n, M).
        Curves defined in each pixel.
    Returns
    _______
    coords
        Array of shape (m, n, N, M) indicating coordinates of sampled point
    tangents
        Array of shape (m, n, N, M) indicating tangents of sampled point
    """
    # Expand control points to (m, n, 1, M) for broadcasting with t (m, n, N, M)
    with fine_scope("expand_controls"):
        p0 = p0[..., np.newaxis, :]  # (m, n, 1, M)
        p1 = p1[..., np.newaxis, :]
        p2 = p2[..., np.newaxis, :]
        p3 = p3[..., np.newaxis, :]

    # t is (m, n, N, M)
    one_minus_t = 1 - t

    # Bézier position
    with fine_scope("bezier_position"):
        coords = (
            one_minus_t**3 * p0 +
            3 * one_minus_t**2 * t * p1 +
            3 * one_minus_t * t**2 * p2 +
            t**3 * p3
        )

    # Bézier tangent (derivative)
    with fine_scope("bezier_tangent"):
        tangents = (
            3 * one_minus_t**2 * (p1 - p0) +
            6 * one_minus_t * t * (p2 - p1) +
            3 * t**2 * (p3 - p2)
        )

    return coords, tangents

@scoped
def upres_bezier(x0,x1,x2,x3, upres=1):
    if upres>1: # Subsample the bezier curves before finding the nearest linear approximation within the subsampled curves
        # initial shape = X,Y,m
        x0, x3, x1, x2 = subdivide_bezier(p0 = x0, p1 = x1, p2 = x2, p3 = x3, num_segments=upres)
        # convert back to shape X,Y,m*upres
        x0 = inneficient_flatten(x0)
        x1 = inneficient_flatten(x1)
        x2 = inneficient_flatten(x2)
        x3 = inneficient_flatten(x3)
    return x0,x1,x2,x3
@scoped
def find_distances(x,y,x0,x1,x2,x3,y0,y1,y2,y3, upres=1, weight_t = 0.0):
    """For each pixel, find the distance to the nearest point in the corresponding bezier curves defined by x/y0-3
    """
    m = x0.shape[-1]
    if upres>1: # Subsample the bezier curves before finding the nearest linear approximation within the subsampled curves
        # initial shape = X,Y,m
        x0,x1,x2,x3 = upres_bezier(x0,x1,x2,x3, upres=upres)
        y0,y1,y2,y3 = upres_bezier(y0,y1,y2,y3, upres=upres)

    t = closest_point_on_lines(px=x[..., np.newaxis], py=y[..., np.newaxis], x0=x0, y0=y0, x1=x3, y1=y3)
    coords_x, _ = sample_pixelwise_bezier_at_t(
        t = t,
        p0 = x0,
        p1 = x1,
        p2 = x2,
        p3 = x3
    )
    coords_y, _ = sample_pixelwise_bezier_at_t(
        t = t,
        p0 = y0,
        p1 = y1,
        p2 = y2,
        p3 = y3
    )
    
    dists = np.power(x[..., np.newaxis, np.newaxis] - coords_x, 2) + np.power(y[..., np.newaxis, np.newaxis] - coords_y, 2)
    if weight_t > 0:
        if upres > 1:
            segment_indices = np.tile(np.arange(upres), m)
            segment_indices = np.broadcast_to(segment_indices, t.shape)
            # segment_fraction = np.zeros_like(t) #replace with integers indicating which subsempled segment a sampled line is on
            t = t+segment_indices
            t = t/upres #scale to 0-1 along original curve
        # t = np.power(t,0.5)
        t = 1-t# Distance upstream
        t = 1+t*weight_t 
        # dists = dists*(0.5+t)*weight_t
        dists = np.power(dists, t+1)
        # dists = dists +t*weight_t # Adding weight values is NOT stable
    dists = np.min(dists, axis = -1)
    dists = np.min(dists, axis = -1)
    return dists


@scoped
def broaden_channels(dists_full, intensity, base_scale=1):
    """Broadens the near-zero regions of the input distance maps per layer.
    Result has a derivative of near 0 at low values, and approaches original values at high values.

    Parameters
    __________
    dists_full : np.ndarray
        Input distance maps of shape (X, Y, num_layers).
    intensity : np.ndarray
        Intensity values of shape (X, Y) indicating the mountain intensity at each pixel.
        Assumed to be in the range [0, 1].
        Higher values result in narrower channels, lower values result in wider channels.
    Returns
    _______
    dists_full : np.ndarray
        Broadened distance maps of shape (X, Y, num_layers).
    """
    num_layers = dists_full.shape[-1]
    layer_biases = np.linspace(0, num_layers, num_layers)*0.5
    intensities_per_layer = layer_biases[np.newaxis, np.newaxis, :] - intensity[..., np.newaxis]
    intensities_per_layer = np.clip(intensities_per_layer, 0, 1)  # Ensure range is [0, 1]

    # Scale intensities to a suitable range for frequencies of the noise function, based on artistic preference when adjusting in Blender, may need to be adapted later for scales other than 1
    intensities_per_layer =intensities_per_layer*0.5  + intensity[..., np.newaxis] + 0.2
    intensities_per_layer = intensities_per_layer*250*base_scale
    rescaled_dists = np.multiply(dists_full, intensities_per_layer)

    weightings_broaden = 1 - rescaled_dists
    weightings_broaden = np.clip(weightings_broaden, 0, 1) # Extent to which the original distance values are adjusted, in order to fade to the original values at high distances

    # Convert the rescaled distances to a sinusoidal function which is subtracted from the original distances
    rescaled_dists = np.sin(rescaled_dists*2*np.pi)/intensities_per_layer
    rescaled_dists = rescaled_dists*(0.5/ np.pi) # Ensure that gradient is equal to 1 at small values for the offset value which will be subtracted
    weightings_broaden = weightings_broaden*rescaled_dists # Scale the sinusoidal function by the weightings, so that it is only applied to the low values
    dists_full = dists_full - weightings_broaden # Subtract the sinusoidal function from the original distances


    return dists_full

@scoped
def softcap_heights(dists_full, intensity, base_scale=1):
    """Flattens the distance maps so that they asymptotically approach a maximum value at high distances.
    Function is similar to (1 - 1/(1+cx))/c, with the maximum value c equal to (100*intensity+0.1)*base_scale.
    Derivative is 1 at low values, and approaches 0 at high values.
    Output is 0 at low valuess and approaches c at high values.

    Parameters
    __________
    dists_full : np.ndarray
        Input distance maps of shape (X, Y, num_layers).
    intensity : np.ndarray
        Intensity values of shape (X, Y) indicating the mountain intensity at each pixel.
        Assumed to be in the range [0, 1].
        Lower values result in lower maximum distances flattening features towards rolling hills, higher values result in higher maximum distances retaining sharp features of original data.
    base_scale : float
        Base frequency of the input noise function. Cap values are inversely proportional to this value.

    Returns
    _______
    dists_full : np.ndarray
        Soft-capped distance maps of shape (X, Y, num_layers).
    """
    intensity = 1 - intensity
    intensity = intensity*100 +0.1 # Convert to a suitable range for frequencies of the noise function (avoiding zeros), based on artistic preference when adjusting in Blender, may need to be adapted later for scales other than 1
    intensity = intensity[..., np.newaxis]*base_scale

    dists_full = 1 + dists_full*intensity
    dists_full = 1 - 1/(dists_full)  # Apply the soft cap function
    dists_full = dists_full/intensity  # Rescale the distance maps by the maximum intensity
    
    return dists_full

@scoped
def additive_mode_blending(dists_full, intensity, lacunarity=1.414, verbose=False):
    """Rescales each layer of the distance maps by a factor based on the intensity and lacunarity then adds and rescales them.
    Rescaling is an artistic choice based on experimentation in Blender, may need to be adapted later for scales other than 1 and lacunarity other than 1.414.
    Helps include detail and variation from all layers, including slopes that result in realistic higher-tier streams at the loss of specific stream shapes.
    
    Parameters
    __________
    dists_full : np.ndarray
        Input distance maps of shape (X, Y, num_layers).
    intensity : np.ndarray
        Intensity values of shape (X, Y) indicating the mountain intensity at each pixel.
        Assumed to be in the range [0, 1].
    lacunarity : float
        Lacunarity value associated with the fractal noise used to generate the distance maps, indicating the scale factor between each layer of noise. 2==octaves.
    Returns
    _______
    dists_full : np.ndarray
        Additively blended distance map of shape (X, Y).
    """
    intensity = intensity+1  # Convert intensity to a range [1, 2] to avoid zeros
    intensity = intensity*lacunarity
    if verbose:
        print(f"Intensity shape: {intensity.shape}, dists_full shape: {dists_full.shape}, lacunarity: {lacunarity}")
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
    intensity_per_layer = intensity[..., np.newaxis]
    intensity_per_layer = np.power(intensity_per_layer, np.arange(dists_full.shape[-1])[np.newaxis, np.newaxis, :]+1) +1 # Intensity is scaled once per layer index, with the first layer being scaled by lacunarity, second by lacunarity squared, and so on, plus one. 
    
    dists_full = dists_full * intensity_per_layer  # Scale each layer by the intensity factor
    # rescaling is based on the maximum noise scale, so using only the maximum frequency and the highest power
    intensity = np.power(intensity, dists_full.shape[-1])+1e-5
    dists_full = np.sum(dists_full, axis=-1)  # Sum the scaled layers


    if verbose:
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
    dists_full = dists_full/intensity  # Rescale the distance maps by the maximum intensity
    if verbose:
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
        print(f"dists_full shape: {dists_full.shape}, dists_full min: {np.min(dists_full)}, max: {np.max(dists_full)}")

    return dists_full

@scoped
def minimum_mode_blending(dists_full, intensity, bias_value = 0.005, verbose=False):
    """Rescales each layer of the distance maps by a factor based on the intensity and lacunarity then takes the minimum.
    Rescaling is an artistic choice based on experimentation in Blender, may need to be adapted later for scales other than 1 and lacunarity other than 1.414.
    Focuses on keeping the exact shapes of streams, at the cost of losing detail and variation when further away from them
    With a bias value of 0 recreates original dendry noise function and simply returns the minimum.
    
    Parameters
    __________
    dists_full : np.ndarray
        Input distance maps of shape (X, Y, num_layers).
    intensity : np.ndarray
        Intensity values of shape (X, Y) indicating the mountain intensity at each pixel.
        Assumed to be in the range [0, 1].
    bias_value : float
        Amount to add to each layer to shift weighting towards the first layer at lower intensities.
        This is a very slight bias, as it will rapidly cause higher layers to disappear.
    Returns
    _______
    dists_full : np.ndarray
        Minimum blended distance map of shape (X, Y).
    """
    intensity = 1-intensity
    intensity = intensity*bias_value
    if verbose:
        print(f"Intensity shape: {intensity.shape}, dists_full shape: {dists_full.shape}, bias_value: {bias_value}")
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")

    intensity_per_layer = intensity[..., np.newaxis]
    intensity_per_layer = intensity_per_layer* np.arange(dists_full.shape[-1])[np.newaxis, np.newaxis, :]

    dists_full = dists_full + intensity_per_layer # Add the intensity as a bias to each layer
    dists_full = np.min(dists_full, axis=-1)  # Take the minimum of the scaled layers

    return dists_full

@scoped
def blend_distance_layers(dists_full, intensity, lacunarity=1.414, bias_value=0.005, base_frequency=1, verbose=False):
    """Blends the distance maps using a combination of additive and minimum mode blending.
    This is an artistic choice based on experimentation in Blender, may need to be adapted later for scales other than 1 and lacunarity other than 1.414.
    Helps include detail and variation from all layers, including slopes that result in realistic higher-tier streams at the loss of specific stream shapes.
    Currently only uses one intensity value for all blending parameters to vary from lowlands to mountains, but could be adapted to use different values for each blending step to allow for two dimensional biome-space.
    
    Parameters
    __________
    dists_full : np.ndarray
        Input distance maps of shape (X, Y, num_layers).
    intensity : np.ndarray
        Intensity values of shape (X, Y) indicating the mountain intensity at each pixel.
        Assumed to be in the range [0, 1].
        High values results in narrower and more even channels with influence from all layers, low values results in wider channels with more influence from the first layer.
    lacunarity : float
        Lacunarity value associated with the fractal noise used to generate the distance maps, indicating the scale factor between each layer of noise. 2==octaves.
    bias_value : float
        Amount to add to each layer to shift weighting towards the first layer at lower intensities.
        This is a very slight bias, as it will rapidly cause higher layers to disappear.
    base_frequency : float
        Frequency of (the lowest component of) the noise function used to generate the distance maps.
        Channels will be broadened proportionally to the wavelength of this noise.
    
    Returns
    _______
    blended_dists : np.ndarray
        Blended distance map of shape (X, Y).
    """
    if verbose:
        print(f"Blending distance layers with intensity shape: {intensity.shape}, dists_full shape: {dists_full.shape}, lacunarity: {lacunarity}, bias_value: {bias_value}, base_frequency: {base_frequency}")
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
    dists_full = broaden_channels(dists_full, intensity, base_scale=base_frequency) 
    if verbose:
        print(f"After broadening channels, dists_full shape: {dists_full.shape}, dists_full min: {np.min(dists_full)}, max: {np.max(dists_full)}")
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
    dists_full = softcap_heights(dists_full, intensity, base_scale=base_frequency)
    if verbose:
        print(f"After softcapping heights, dists_full shape: {dists_full.shape}, dists_full min: {np.min(dists_full)}, max: {np.max(dists_full)}")
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
    sum_dists = additive_mode_blending(dists_full, intensity, lacunarity, verbose=verbose)
    if verbose:
        print(f"After additive blending, dists_full shape: {dists_full.shape}, dists_full min: {np.min(dists_full)}, max: {np.max(dists_full)}")
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
    min_dists = minimum_mode_blending(dists_full, intensity, bias_value, verbose=verbose)
    if verbose:
        print(f"After minimum blending, dists_full shape: {dists_full.shape}, dists_full min: {np.min(dists_full)}, max: {np.max(dists_full)}")
        print(f"Intensity min: {np.min(intensity)}, max: {np.max(intensity)}")
    min_dists = np.power(min_dists,0.75) 
    sum_of_modes = (sum_dists + min_dists*2)*0.4/3  # Combine the two modes additively, producing detailed shapes
    product_of_modes = 10*sum_dists * min_dists  # Combine the two modes multiplicatively, retaining 0 values eg streams

    # Blend the two modes together, with higher intensity biasing towards the sum of modes and lower intensity biasing towards the product of modes
    sum_of_modes = sum_of_modes*intensity
    product_of_modes = product_of_modes*(1-intensity)
    sum_of_modes = sum_of_modes + product_of_modes  # Combine the two modes additively

    return sum_of_modes
class smoothnoise_generator(): 
    
    def __init__(self,x=128,y=128,max_oct=32, random_seed=1, precomputed_pattern = None):
        if precomputed_pattern is not None:
            # load in existing values as constants. Preferable for jit compilation eg jax
            self.pattern_ref = precomputed_pattern
        else:
            np.random.seed(random_seed) #NB this will not work with Jax optimisation
            self.pattern_ref = np.random.rand(x,y,3)*2 -1 #array of random values between -1 and 1
        self.cos_lut = [math.cos(2*i) for i in range(max_oct)] #used instead of calculating trig functions per pixel at runtime
        self.sin_lut = [math.sin(2*i) for i in range(max_oct)]
        #for i in range(max_oct): #so that negative values of i correspond to negative angles
        #    self.cos_lut.append(math.cos(2*(i-max_oct)))
        #    self.sin_lut.append(math.sin(2*(i-max_oct)))
            
    def pattern(self,x,y,ndims=3):
        # return(self.pattern_ref[x.astype(int)%self.pattern_ref.shape[0],y.astype(int)%self.pattern_ref.shape[1],0:ndims])
        #Converts this version back to old method (pregen noise pattern) 
        
        output_by_sin_fraction = np.sin(x)[..., np.newaxis] + np.cos(y[..., np.newaxis]*1.414 + np.arange(ndims))
        output_by_sin_fraction = np.sin(output_by_sin_fraction* 1000) * 100
        # take only fractional component:
        output_by_sin_fraction = output_by_sin_fraction - np.floor(output_by_sin_fraction)
        return(output_by_sin_fraction)

        #This is an attempt at procedural noise, but it produces too many artifacts to replace the pattern method
        primes = [4937,6053,5843,6701,6133,7919,7823,5281,5407,5443]
        output = np.zeros(ndims)
        for i in range(ndims):
            output[i] = 0.5 - ((x*(y+i+3)*7717 + y*(x+10*i)*7907*7717)%primes[i])/primes[i] #Pseudorandom output between -0.5 and 0.5
        return(output)
    def find_grid(self, x, y, n=5, scale=1.0, epsilon=0.5, frequency = 1000, rotation = 0, recenter=True): 
        # TODO REVIEW FOR EFFICIENCY, can we just floor x*freq directly?
        # Generate a grid of jittered points around each input (x,y).
        # x and y are assumed to have shape (res, res).
        # Returns grid_centroids_x and grid_centroids_y with shape (res, res, n, n).

        # Generate a regular grid of points centered around 0. The set of offsets is the same in x and y
        #offsets = np.linspace(-0.5,0.5,n, endpoint=False) * scale*n

        with named_scope("find_grid_offsets"):
            offsets = np.linspace(-(n-1)/2, (n-1)/2, n) # * scale

            # Expand x and y to shape (res, res, n, n) including added offsets of (n*n)

            offset_x, offset_y = np.meshgrid(offsets, offsets)
            s = np.sin(rotation*2) # self.sin_lut[rotation] #
            c = np.cos(rotation*2) # self.cos_lut[rotation] #
            qx = c * offset_x - s * offset_y #rotate offsets
            qy = s * offset_x + c * offset_y
            offset_x = qx
            offset_y = qy

            grid_centroids_x = np.add(x[..., np.newaxis, np.newaxis]*frequency, offset_x[np.newaxis, np.newaxis, ...])
            grid_centroids_y = np.add(y[..., np.newaxis, np.newaxis]*frequency, offset_y[np.newaxis, np.newaxis, ...])

        with named_scope("find_grid_snap"):
            # To snap the centroids on a rotated grid, we rotate the entire set in reverse, then floor+center, then rotate forwards again
            #grid_centroids_x = grid_centroids_x*frequency
            #grid_centroids_y = grid_centroids_y*frequency
            qx = c * grid_centroids_x + s * grid_centroids_y # Reverse rotation
            qy = c * grid_centroids_y - s * grid_centroids_x
            grid_centroids_x = qx
            grid_centroids_y = qy
            if recenter:
                # Constant offset of 0.5 so that points are centred within their grid spaces, in a grid proportional to the frequency
                # TODO: check if this is necessary for dendry (not used in Voronoi)
                grid_centroids_x = np.floor(grid_centroids_x)+0.5
                grid_centroids_y = np.floor(grid_centroids_y)+0.5
            else:
                grid_centroids_x = np.floor(grid_centroids_x)
                grid_centroids_y = np.floor(grid_centroids_y)

            qx = c * grid_centroids_x - s * grid_centroids_y # Forward rotation
            qy = s * grid_centroids_x + c * grid_centroids_y
            grid_centroids_x = qx
            grid_centroids_y = qy

        with named_scope("find_grid_jitter"):
            # Compute jitter based on the pattern function
            #jitter = self.pattern(x_exp*frequency, y_exp*frequency, ndims=2) #having a grid size close to one means that cells receive the same jitter as their neighbours, as they are falling into the same bins in the pattern.
            #Frequency upsamples that so that nearby cells are less likely to have the same jitter

            jitter = self.pattern(grid_centroids_x*scale, grid_centroids_y*scale, ndims=2)
            # OPTIONAL: use jitter * abs(jitter) to weight towards lower values ie the middle of each cell. This is less uniform and unused edges may be visible, but it is less likely to hit edge cases when finding eg nearest neighbours
            grid_centroids_x = grid_centroids_x/frequency
            grid_centroids_y = grid_centroids_y/frequency

            # Rotate the jitter so that it is within the rotated grid squares
            jitter_x = (c*jitter[..., 0] - s*jitter[..., 1])  * epsilon
            jitter_y = (s*jitter[..., 0] + c*jitter[..., 1]) * epsilon

            # # Alternative: turn jitter into polar coordinates in radius epsilon to avoid hitting corners (UNTESTED)
            # jitter_x = np.sin(jitter_x) * epsilon
            # jitter_y = np.cos(jitter_y) * epsilon

            grid_centroids_x = np.add(grid_centroids_x,jitter_x/frequency)
            grid_centroids_y = np.add(grid_centroids_y,jitter_y/frequency)

        return grid_centroids_x, grid_centroids_y


    def dendry_higher_tiers(self,x,y, dists_full, spline_start_x,spline_start_control_x,spline_end_control_x,spline_end_x, spline_start_y,spline_start_control_y,spline_end_control_y,spline_end_y,
                        base_frequency, epsilon=0.4,skew=0.5, lacunarity=1.414, push_upstream=0.1, push_downstream=0.2, scale_factor_start = 0.250,
                        soften_start = 0.75, weight_t=0.0 , max_tier=3, upres_tier_max=0, upres=2, scale=1, verbose=False):
        if verbose:
            print("Generating higher tiers of dendry noise")
            print("Base frequency:", base_frequency)
            print("Lacunarity:", lacunarity)
            print("Epsilon:", epsilon)
            print("Skew:", skew)
            print("Push upstream:", push_upstream)
            print("Push downstream:", push_downstream)
            print("Scale factor start:", scale_factor_start)
            print("Soften start:", soften_start)
            print("Weight t:", weight_t)
            print("Upres", upres)
        skew1 = soften_start*skew
        tier_freq = base_frequency*lacunarity
        for tier in range(1,max_tier+1):
            with named_scope(f"dendry_tier_{tier}_grid"):
                new_points_x, new_points_y = self.find_grid(x, y, n=3,epsilon=epsilon, frequency=tier_freq, rotation=tier, scale=1)
                new_points_x = inneficient_flatten(new_points_x)
                new_points_y = inneficient_flatten(new_points_y)
            t = closest_point_on_lines(px=new_points_x, py=new_points_y, x0=spline_start_x, y0=spline_start_y, x1=spline_end_x, y1=spline_end_y)
            # This gives an output of shape x,y,N,M, where x and y are the resolution of the image, 
            # N is the number of new points, and M is the number of curves for each pixel that have already been defined.
            t = t + push_downstream
            coords_x, tangents_x = sample_pixelwise_bezier_at_t(
                t = t,
                p0 = spline_start_x,
                p1 = spline_start_control_x,
                p2 = spline_end_control_x,
                p3 = spline_end_x
            )
            coords_y, tangents_y = sample_pixelwise_bezier_at_t(
                t = t,
                p0 = spline_start_y,
                p1 = spline_start_control_y,
                p2 = spline_end_control_y,
                p3 = spline_end_y
            )
            dists = np.power(new_points_x[..., np.newaxis] - coords_x, 2) + np.power(new_points_y[..., np.newaxis] - coords_y, 2)
            chosen = np.argmin(dists, axis=-1)

            m, n, N, M = coords_x.shape

            # Build index arrays for advanced indexing
            i = np.arange(m)[:, None, None]
            j = np.arange(n)[None, :, None]
            k = np.arange(N)[None, None, :]

            # Use chosen as the index for the M axis
            coords_x = coords_x[i, j, k, chosen]
            coords_y = coords_y[i, j, k, chosen]
            tangents_x = tangents_x[i, j, k, chosen]
            tangents_y = tangents_y[i, j, k, chosen]
            # Normalise the tangents to length 1 (necessary to keep consistent shapes across lengths/tiers) # normally these would be equivalent to three times the length of the curve, but normalisation handles this
            tangent_length = np.power(tangents_x, 2) + np.power(tangents_y, 2)
            tangent_length = np.power(tangent_length, 0.5)+1e-10
            tangents_x = tangents_x/(tangent_length*tier_freq)
            tangents_y = tangents_y/(tangent_length*tier_freq)
            # construct new beziér curves, compute pixelwise distance for this tier, and append to existing tree
            tangents_x = skew*tangents_x#/tier_freq#*scale_factor_end
            tangents_y = skew*tangents_y#/tier_freq#*scale_factor_end
            # When building into the function, directly scale the tangent values to avoid repitition # in full version, avoid duplication
            
            new_x1 = (1-skew1)*new_points_x + (coords_x)*(skew1) - tangents_x*push_upstream
            new_y1 = (1-skew1)*new_points_y + (coords_y)*(skew1) - tangents_y*push_upstream
            new_x2 = coords_x - tangents_x # in full version, avoid duplication
            new_y2 = coords_y - tangents_y
            # # Experimental: interpolate towards the start when defining end handles for a gentler join
            # new_x25 = (skew*scale_factor_start)*new_points_x + (coords_x)*(1-skew*scale_factor_start) - tangents_x
            # new_y25 = (skew*scale_factor_start)*new_points_y + (coords_y)*(1-skew*scale_factor_start) - tangents_y
            new_x25 = (scale_factor_start)*new_points_x + (new_x2)*(1-scale_factor_start) 
            new_y25 = (scale_factor_start)*new_points_y + (new_y2)*(1-scale_factor_start) 

            new_points_x = new_points_x - tangents_x*push_upstream
            new_points_y = new_points_y - tangents_y*push_upstream
            # Distance calcs for this tier
            with named_scope(f"dendry_tier_{tier}_distances"):
                dists_new= find_distances(x=x, y=y, x0=new_points_x, y0=new_points_y,
                                    x1 = new_x1, y1=new_y1,
                                    x2=new_x25, y2=new_y25,
                                    x3=coords_x, y3=coords_y, upres=8, weight_t=weight_t)
            dists_full = np.concatenate([dists_full, dists_new[:,:, np.newaxis]], axis = 2)
            if tier < max_tier: #update variables and combine coordinate structures for next tier
            
                if (upres_tier_max < 0 or tier <= upres_tier_max) and upres>1:  # Subsample the new curves for accuracy in higher tiers (MEMORY INTENSIVE)
                    new_points_x,new_x1,new_x25,coords_x = upres_bezier(x0=new_points_x,
                                    x1 = new_x1, 
                                    x2=new_x25, 
                                    x3=coords_x, upres=upres)
                    new_points_y,new_y1,new_y25,coords_y = upres_bezier(x0=new_points_y,x1 = new_y1,x2=new_y25,x3=coords_y, upres=upres)
                # push_upstream = push_upstream/lacunarity
                # push_downstream = push_downstream/lacunarity
                tier_freq = tier_freq*lacunarity
                if verbose:
                    print(spline_start_x.shape)
                spline_start_x = np.concatenate([spline_start_x, new_points_x], axis = -1) # p0
                spline_start_y = np.concatenate([spline_start_y, new_points_y], axis = -1)
                spline_start_control_x = np.concatenate([spline_start_control_x, new_x1], axis = -1) # p1
                spline_start_control_y = np.concatenate([spline_start_control_y, new_y1], axis = -1)
                spline_end_control_x = np.concatenate([spline_end_control_x, new_x25], axis = -1) # p2
                spline_end_control_y = np.concatenate([spline_end_control_y, new_y25], axis = -1)
                spline_end_x = np.concatenate([spline_end_x, coords_x], axis = -1) # p3
                spline_end_y = np.concatenate([spline_end_y, coords_y], axis = -1)
        return dists_full
    def dendry(self, x, y, intensity=None,
               dendry_layers = 2, upres = 2, final_sample = 10, initial_method = 'b', upres_tier_max = 0,
               base_frequency = 1, epsilon = 0.4, skew = 0.5, lacunarity = 1.414, push_upstream = 0.1, push_downstream = 0.2,
               scale_factor_start = 0.250, soften_start = 0.75, weight_t=0.0, bias_value=0.005, verbose = False,
               control_function=None, return_full = False, scale=1, blend_scale=1, **kwargs):
        """
        Generate dendry (river) noise for input coordinates x, y (shape: (res, res)).
        
        Base level (octave 0):
          - Generates a 5x5 grid per evaluation point.
          - For each inner cell (i,j where i,j in 1..3) of that grid,
            finds the chosen neighbor from its 3x3 neighborhood.
            If a control_function is provided, it is used to determine
            the "height" at each grid point and the lowest height is chosen.
            If not, the nearest neighbor (by Euclidean distance) is chosen.
          - For each inner cell, a single spline is generated (using generate_spline_points)
            between the grid cell's center and its chosen neighbor.
            This yields 9 splines per evaluation point.
        
        Higher octaves:
          - For each octave, a new grid is generated using n=3 (the entire grid is used).
          - For each grid point, the nearest existing tree point (from lower octaves)
            is found and a spline is generated between them.
        
        Finally, the function computes, for each evaluation point, the minimum distance to any
        spline sample point in the union of all splines blended with the sum of these distances, and returns this distance field.
        """

        # First tier
        # skew = skew*(base_frequency**0.75)
        if verbose:
            print("Generating first tier of dendry noise")
            print("Base frequency:", base_frequency)
            print("Lacunarity:", lacunarity)
            print("Epsilon:", epsilon)
            print("Skew:", skew)
            print("Push upstream:", push_upstream)
            print("Push downstream:", push_downstream)
            print("Scale factor start:", scale_factor_start)
            print("Soften start:", soften_start)
            print("Weight t:", weight_t)
            print("Upres", upres)
            print("bias value:", bias_value)
        base_grid_size = 7
        tree_x, tree_y = self.find_grid(x, y, n=base_grid_size, epsilon=epsilon, frequency=base_frequency, rotation=1, scale=scale)
        if verbose:
            print("Tree shape:", tree_x.shape, tree_y.shape)
        # find values of initial points according to the control function
        if control_function is None:
            c = np.zeros_like(tree_x)
            for i in range(base_grid_size): 
                    if verbose:
                        print(f"i: {i}")
                    for j in range(base_grid_size):
                        if control_function is None:
                            c[:,:,i,j] = 1.75*tree_x[:,:,i,j] + 0.15*tree_y[:,:,i,j] + self.sample(tree_x[:,:,i,j], tree_y[:,:,i,j], ndims=1)[:,:,0] 
        else:
            c = control_function(tree_x, tree_y, **kwargs)
        chosen_idx_x, chosen_idx_y = check_inner_grid(c) # shape = rx, ry, 5,5, indices from 0 to 6 for tree_x and tree_y
        # Select points for first tier of splines
        # P0
        spline_start_x = tree_x[...,2:-2,2:-2]
        spline_start_y = tree_y[...,2:-2,2:-2]
        # P3
        spline_end_x = index_within_subgrid(tree_x, chosen_idx_x[...,1:-1,1:-1], chosen_idx_y[...,1:-1,1:-1])
        spline_end_y = index_within_subgrid(tree_y, chosen_idx_x[...,1:-1,1:-1], chosen_idx_y[...,1:-1,1:-1])
        # P6, aka the endpoint of the next spline that would start from p3
        # Target nodes from each end node are the forward direction and need to be reversed
        chosen_of_target_x = index_within_subgrid(chosen_idx_x, chosen_idx_x, chosen_idx_y, offset_z=-1, offset_w=-1, mask_value=chosen_idx_x)[...,1:-1,1:-1] # NB these are indices only, not the coordinates
        chosen_of_target_y = index_within_subgrid(chosen_idx_y, chosen_idx_x, chosen_idx_y, offset_z=-1, offset_w=-1, mask_value=chosen_idx_y)[...,1:-1,1:-1]
        spline_end_target_x = index_within_subgrid(tree_x, chosen_of_target_x, chosen_of_target_y)
        spline_end_target_y = index_within_subgrid(tree_y, chosen_of_target_x, chosen_of_target_y)
        # P2
        spline_end_control_x = (1+skew)*spline_end_x - skew*spline_end_target_x
        spline_end_control_y = (1+skew)*spline_end_y - skew*spline_end_target_y

        # P1, there are a few options here. B is the default, and gives slightly strange results but guarantees smooth joins (direction matches between segments)
        skew1 = skew*0.1 # Unclear why, but larger values cause instability
        if initial_method == 'a':
            spline_start_control_x = (1-skew) * spline_start_x + skew*(2*spline_end_x - spline_end_target_x)
            spline_start_control_y = (1-skew) * spline_start_y + skew*(2*spline_end_y - spline_end_target_y)
        elif initial_method == 'b':
            spline_start_control_x = (1-skew1) * spline_start_x + skew1*spline_end_x
            spline_start_control_y = (1-skew1) * spline_start_y + skew1*spline_end_y
        elif initial_method == 'c':
            spline_start_control_x = spline_start_x + skew*(spline_end_target_x - spline_end_x)
            spline_start_control_y = spline_start_y + skew*(spline_end_target_y - spline_end_y)
        elif initial_method == 'd':
            spline_start_control_x = (2-skew) * spline_start_x + (1+skew)*spline_end_x
            spline_start_control_y = (2-skew) * spline_start_y + (1+skew)*spline_end_y
        # Flatten all sets of points to m*n*9 and rescale to base frequency so that the control function values are in the original space but all splines are at the desired frequency
        spline_start_x = inneficient_flatten(spline_start_x)
        spline_start_control_x = inneficient_flatten(spline_start_control_x)
        spline_end_control_x = inneficient_flatten(spline_end_control_x)
        spline_end_x = inneficient_flatten(spline_end_x)
        spline_start_y = inneficient_flatten(spline_start_y)
        spline_start_control_y = inneficient_flatten(spline_start_control_y)
        spline_end_control_y = inneficient_flatten(spline_end_control_y)
        spline_end_y = inneficient_flatten(spline_end_y)

        with named_scope("dendry_first_tier_distances"):
            if upres>1: # Subsample the bezier curves before finding the nearest linear approximation within the subsampled curves
                spline_start_x,spline_start_control_x,spline_end_control_x,spline_end_x = upres_bezier(x0=spline_start_x,x1 = spline_start_control_x,x2=spline_end_control_x,x3=spline_end_x, upres=upres)
                spline_start_y,spline_start_control_y,spline_end_control_y,spline_end_y = upres_bezier(x0=spline_start_y,x1 = spline_start_control_y,x2=spline_end_control_y,x3=spline_end_y, upres=upres)
            dists_full = find_distances(x=x, y=y, x0=spline_start_x, y0=spline_start_y,
                           x1 = spline_start_control_x, y1=spline_start_control_y,
                           x2=spline_end_control_x, y2=spline_end_control_y,
                           x3=spline_end_x, y3=spline_end_y, upres=4)
        # higher tiers
        # Each curve is stored only in terms of its control points
        # Per tier (per pixel):
        #    create a new 3*3 grid of points in a smaller area 
        #    for each of these 9 points, for each of the existing curves in the tree, find the distance from that point to the nearest point on some approximation of that curve (eg a straight line)
        #    For each of the 9 points, define a curve that goes from that point to nearest of these determined points (no longer using approximation, translated in terms of eg fraction along line)
        #    Add (the control points of) these 9 curves to the existing tree
        with named_scope("dendry_higher_tiers"):
            dists_full = self.dendry_higher_tiers(x=x,y=y, dists_full=dists_full[..., np.newaxis],
                                     spline_start_x = spline_start_x,spline_start_control_x = spline_start_control_x,
                                     spline_end_control_x = spline_end_control_x,spline_end_x = spline_end_x,
                                     spline_start_y = spline_start_y,spline_start_control_y = spline_start_control_y,
                                     spline_end_control_y = spline_end_control_y,spline_end_y = spline_end_y,
                                     base_frequency = base_frequency,
                                     epsilon=epsilon,
                                     skew=skew,
                                     lacunarity=lacunarity,
                                     push_upstream=push_upstream,
                                     upres=upres,
                                     push_downstream=push_downstream,
                                     soften_start = soften_start, scale_factor_start = scale_factor_start,
                                     weight_t=weight_t, max_tier=dendry_layers, upres_tier_max=upres_tier_max, verbose=verbose, scale=scale)
        if return_full:
            return dists_full
        else:
            if intensity is None: #If intensity is not given, or is a single value, use a default intensity
                intensity = np.ones_like(x)*0.5
            elif np.isscalar(intensity): #If intensity is a single value, use it for all pixels
                intensity = np.ones_like(x)*intensity
            with named_scope("dendry_blend"):
                blended_dists = blend_distance_layers(dists_full, intensity, lacunarity=lacunarity, bias_value=bias_value/base_frequency, base_frequency=base_frequency*blend_scale)
            return blended_dists

    def base_sample(self,x,y,**kwargs): # currently simple interpolation, not true perlin noise
        with named_scope("base_sample_lattice"):
            with named_scope("grid_cell"):
                lox = np.floor(x)
                loy = np.floor(y)
            with named_scope("corner_noise"):
                a = self.pattern(lox,loy,**kwargs)
                b = self.pattern(lox,(loy+1),**kwargs)
                c = self.pattern((lox+1),loy,**kwargs)
                d = self.pattern((lox+1),(loy+1),**kwargs)

        #weights = [1-x%1, x%1, 1-y%1, y%1]
        #a = a*(weights[0]*weights[2])
        #b = b*(weights[0]*weights[3])
        #c = c*(weights[1]*weights[2])
        #d = d*(weights[1]*weights[3])
        with named_scope("base_sample_weights"):
            with named_scope("smoothstep_x"):
                weight1 = x%1
                # weight1 = weight1[:,:,None]
                weight1 = weight1 * weight1 * weight1 * (weight1 * (6 * weight1 - 15) + 10)
                weight0 = 1-weight1
            with named_scope("smoothstep_y"):
                weight3 = y%1
                # weight3 = weight3[:,:,None]
                weight3 = weight3 * weight3 * weight3 * (weight3 * (6 * weight3 - 15) + 10)
                weight2 = 1-weight3

            weight0 = np.asarray(weight0)
            weight1 = np.asarray(weight1)
            weight2 = np.asarray(weight2)
            weight3 = np.asarray(weight3)

        with named_scope("base_sample_interpolate"):
            with named_scope("weight_corners"):
                a *= weight0[..., np.newaxis]
                a *= weight2[..., np.newaxis]
                b *= weight0[..., np.newaxis]
                b *= weight3[..., np.newaxis]
                c *= weight1[..., np.newaxis]
                c *= weight2[..., np.newaxis]
                d *= weight1[..., np.newaxis]
                d *= weight3[..., np.newaxis]
            with named_scope("bilinear_sum"):
                a = np.add(a,b)
                c = np.add(c,d)
                s = np.add(a,c)
        return(s)#abs(s)*s*(3-2*s))
    
    
    def sample(self,x,y,octaves=1,neg_octaves=0, fade=0.5,voron=False,ndims=3, **kwargs) -> np.ndarray:
        with named_scope("sample_octave_coords"):
            i_range = np.arange(neg_octaves*-1, octaves)
            base_scale = 2.0 ** i_range[0] # scale for the first octave, to avoid integers to negative powers
            i_range_from_0 = np.arange(len(i_range))
            with named_scope("octave_scales"):
                ax =  x[..., np.newaxis] * np.power(2, i_range_from_0) * base_scale # np.stack([x*base_scale*2**i for i in i_range_from_0], axis=-1) #
                ay =  y[..., np.newaxis] * np.power(2, i_range_from_0) * base_scale # np.stack([y*base_scale*2**i for i in i_range_from_0], axis=-1) #
            with named_scope("octave_rotation"):
                c = np.cos(2*i_range) # self.cos_lut[i]
                s = np.sin(2*i_range) # self.sin_lut[i]

                qx =   np.subtract(ax*c, ay*s)
                qy =   np.add(ax*s, ay*c)
        if voron:
            with named_scope("sample_voron"):
                output = self.voron(qx,qy, randomness = 0.5)
            # Previous version of this function returned identical values for each dimension, which resulted in the same shape for voron vs base_sample
            # TODO: check if this is needed/assumed, as it is clearly redundant
            # output = np.stack([output for _ in range(ndims)], axis=-1)
        else:
            with named_scope("sample_base"):
                output = self.base_sample(qx,qy,ndims=ndims)

        with named_scope("sample_octave_blend"):
            with named_scope("octave_weights"):
                weights = np.power(fade, i_range)
            # for i in range(len(i_range)):
            #     output[..., i, :] = output[..., i, :]# *weights[i]
            with named_scope("weighted_octave_sum"):
                if voron:
                    axis_of_iters = -2
                    output = np.stack([output[...,i]*weights[i] for i in range(len(i_range))], axis=-1)
                    output = np.stack([output for _ in range(ndims)], axis=-1) # duplicate the voron output across all dimensions, as it is a single value per pixel
                else:
                    axis_of_iters = -2
                    output = np.stack([output[...,i, :]*weights[i] for i in range(len(i_range))], axis=-2)
                # sample_stack = output.copy()
                output = np.sum(output, axis=axis_of_iters)
        return output # , weights, c, s, qx, qy, ax, ay, sample_stack
    
    def get_height(self,x,y,channel=-1, **kwargs):
        return(self.sample(x,y,**kwargs)[:,:,channel]) #Note that ndims >1 is irrelevant if channel is not a list/array, as only one channel will be selected



    def find_nearest(self,x,y,randomness = 0.5): #Create a voronoi (or Worley noise) pattern from the same starting pattern, returning distance to nearest centroid
        lox = np.floor(x)
        loy = np.floor(y)
        sqdist = np.zeros_like(x) + 1000
        for x_off in range(-2,4):
            for y_off in range(-2,4):
                #get the random offset of that location in the pattern. Pattern is -1 to 1, scale by 0.5 keeps points from overlapping
                centroid = self.pattern(lox + x_off, loy + y_off,ndims = 1)*0.5
                centroid += [x_off, y_off]
                
                centroid = np.square(centroid)
                centroid = np.sum(centroid, axis = -1)
                
                sqdist = np.minimum(sqdist, centroid)#If this centroid is closer than previous, keep this distance (always a closest neighbour search, second closest neighbour not used here
        return(np.sqrt(sqdist)[:,:,None])
    def voron(self,x,y,randomness = 0.5):

        # new implementation using find_grid and check_inner_grid, intended to be dimensionality agnostic
        lox = np.floor(x)
        loy = np.floor(y)
        # frac_x = x-lox
        # frac_y = y-loy
        with named_scope("voron_grid"):
            grid_centroids_x, grid_centroids_y = self.find_grid(lox, loy, n=5, scale=1.0, epsilon=randomness*2, frequency = 1.0, rotation = 0, recenter=False)
            # get the coordinates of the centroids of the grid squares around each point, with some randomness to avoid regularity.
            # These are in absolute coordinates, not relative to the input point

        with named_scope("voron_nearest_distance"):
            grid_centroids_x = grid_centroids_x - x[..., np.newaxis, np.newaxis] # coordinates of centroids relative to the input point
            grid_centroids_y = grid_centroids_y - y[..., np.newaxis, np.newaxis]
            grid_centroids_x = np.power(grid_centroids_x, 2)
            grid_centroids_y = np.power(grid_centroids_y, 2)
            sqdist = np.add(grid_centroids_x, grid_centroids_y)
            sqdist = np.min(sqdist, axis=-1)
            sqdist = np.min(sqdist, axis=-1) # minimum distance to any centroid in the per-pixel grid
        return np.sqrt(sqdist)# return the distance to the nearest centroid, as a simple voronoi function
    
    
    def profile_base_sample(self, x, y, ndims=3): #Not to be used in production code, only for profiling current version of base_sample
        import time
        # Time each major operation
        timings = {}
        
        start = time.perf_counter()
        lox = np.floor(x)
        loy = np.floor(y)
        timings['floor'] = time.perf_counter() - start
        
        start = time.perf_counter()
        a = self.pattern(lox, loy, ndims=ndims)
        b = self.pattern(lox, (loy+1), ndims=ndims)
        c = self.pattern((lox+1), loy, ndims=ndims)
        d = self.pattern((lox+1), (loy+1), ndims=ndims)
        timings['pattern_lookup'] = time.perf_counter() - start
        
        start = time.perf_counter()
        weight1 = x % 1
        weight1 = weight1[:,:,None]
        weight1 = weight1 * weight1 * weight1 * (weight1 * (6 * weight1 - 15) + 10)
        weight0 = 1-weight1
        weight3 = y % 1
        weight3 = weight3[:,:,None]
        weight3 = weight3 * weight3 * weight3 * (weight3 * (6 * weight3 - 15) + 10)
        weight2 = 1-weight3
        timings['weight_calc'] = time.perf_counter() - start
        
        start = time.perf_counter()
        a *= weight0
        a *= weight2
        b *= weight0
        b *= weight3
        c *= weight1
        c *= weight2
        d *= weight1
        d *= weight3
        a = np.add(a,b)
        c = np.add(c,d)
        s = np.add(a,c)
        timings['interpolation'] = time.perf_counter() - start
        
        return s, timings
precomputed_noise = None
precomputed_noise_path = Path(REPO_ROOT) / "map_generator" / "resources" / "pattern_ref.npy"
if precomputed_noise_path.exists():
    # This is used to have static noise that is not precomputed each time, as this is essentially a constant in practice, and loading it as such is better for jit compilation
    # the noise file used here is generated with the following code (the same as the backup generator):
    # numpy.random.seed(1)
    # pattern_ref = numpy.random.rand(256,256,3) * 2 - 1
    precomputed_noise = np.load(precomputed_noise_path)
    # for other noise, or larger noise if tiling issues emerge, then this can be recreated with different settings or resolution
    # optionally this could all be changed for a procedural method of pseudorandom calculations, but this will need to be highly performant


my_perl = smoothnoise_generator(256,256, precomputed_pattern=precomputed_noise)

def julia(x,y,c = 1j, n = 5,cap = 2):
    z = x + y*1j
    start = abs(z)
    val = 0
    new = abs(z)
    for reps in range(n):
        if new > cap:
            ec = reps + 1 - math.log2(new)#*3.32192809489*0.5
            ec = max(ec,1)
            return(1-(1/ec))
        z = z**2 + c
        new = abs(z)
        if new == start:
            return(1)
    return(1)
        
class julia_gen():
    def __init__(self, c=1j,n=5):
        self.c = c
        self.n = n
    
    def get_height(self,x,y,channel=-1, octaves=1,neg_octaves=0, fade=0.5,rotation = 4): #Channel is unused, this is just to keep in line with other functions
        height = 0
        cap_factor = fade ** (neg_octaves) #TO DO: Move all possible calculations OUTSIDE of per-pixel loop
        for octave in range(-1*neg_octaves,octaves):
            c, s = np.cos(rotation*octave), np.sin(rotation*octave)
            a = c*x - s*y
            b = s*x + c*y
            a = a*1.2**octave
            b = b*1.2**octave
            height += julia(a,b,self.c,self.n)*fade**octave
            
            
        return (height*cap_factor)

class fractal_julia():
    def __init__(self, c=1j,n=5):
        self.c = c
        self.n = n
    
    def get_height(self,x,y,channel=-1, octaves=1,neg_octaves=0, fade=0.5,rotation = 4): #Channel is unused, this is just to keep in line with other functions
        height = 0
        #cap_factor = fade ** (neg_octaves) #TO DO: Move all possible calculations OUTSIDE of per-pixel loop
        angl = np.arctan2(x,y)
        new_height = julia(x,y,self.c,self.n)
        for octave in range(-1*neg_octaves,octaves):
            if new_height > 0:
                height += new_height/(octaves+neg_octaves)
                angl += rotation
                c, s = np.cos(angl), np.sin(angl)
                c = c*(1+fade-new_height)/(new_height+fade)
                s = s*(1+fade-new_height)/(new_height+fade)
                new_height = julia(c,s,self.c,self.n)
            
            
        return (height)
if __name__ == '__main__':
    import cv2
    x_size = 256
    y_size = 256
    zoom = 1.5
    x_off = 100
    y_off = - 100
    subsamples = 2
    base_frequency = 10
    epsilon = 0.4# 0.4 is upper limit on stability. 0.5 is entirely random, but introduces instability
    dendry_layers = 4
    final_sample = 10

    mode = "cubic"
    skew = 0.5
    skew1 = 0.0
    initial_method = 'b'

    graphical_debug = False # Whether to show the intermediate results of each layer
    control_function = None
    eval_per_tier = False # Whether to perform distance calculations of new points to each previous tier in succession vs all at once. Not used in current version
    # This is slower but should result in more even memory usage


    base_grid_size = 7

    x, y = np.meshgrid(np.linspace(-zoom*0.5, zoom*0.5, x_size), np.linspace(-zoom*0.5, zoom*0.5, y_size))
    x = x + x_off
    y = y + y_off
    blended_dists = my_perl.dendry(x,y,intensity=0.5, dendry_layers = dendry_layers, return_full = True, base_frequency = base_frequency, epsilon = epsilon, skew = skew,)
    print(blended_dists.shape)
    for i in range(blended_dists.shape[2]): # save each layer of the blended distances as a greyscale image
        layer = blended_dists[:,:,i]
        print(f'Layer {i} min: {np.min(layer)}, max: {np.max(layer)}')
        layer = np.power(layer, 0.5)*255
        cv2.imwrite(f'f10_dendry_layer_{i}.png', layer)


