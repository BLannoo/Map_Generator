### Experimental code
# to quickly test gpu acceleration, uses cupy instead of numpy
# not expected to work on windows systems, can be enabled for WSL systems but is untested

gpu_enabled = False
debug = True
# detect if cuda is available
try:
    import cupy
    # Check if CUDA is available and functional
    if debug:
        print("cupy imported. Checking for CUDA availability...")
    if hasattr(cupy, 'cuda') and cupy.cuda.is_available():
        if debug:
            print("CUDA is available. Testing functionality.")
        try:
            # Attempt to allocate a small array on the GPU
            _ = cupy.random.rand(2,2,3)
            if debug:
                print("CUDA is functional. Enabling GPU acceleration.")
            gpu_enabled = True
        except Exception as e:
            if debug:
                print(f"CUDA test failed: {e}. Disabling GPU acceleration.")
            gpu_enabled = False
    else:
        if debug:
            print("CUDA is not available. Disabling GPU acceleration.")
        gpu_enabled = False
except:
    if debug:
        print("cupy import failed. Disabling GPU acceleration.")
    gpu_enabled = False
jax_available = False
try:
    import jax
    jax_available =  True # Experimental feature, disabled for testing
except:
    if debug:
        print("jax import failed. JAX will not be available.")
    jax_available = False

if jax_available: # disabling for testing
    if debug:
        print("Using jax for graph based acceleration.")
    from jax.numpy import newaxis, array_split, asarray, min, power, max, ndarray, add, multiply, sin, linspace, floor, concatenate, arange, zeros_like, zeros, cos, clip, sum, argmin, minimum, sqrt, maximum, broadcast_to, pi
    from jax.numpy import subtract, ones_like, meshgrid, log2, array, abs, arctan2, where, square, uint16, divide, sort, absolute, linalg, ogrid, take_along_axis, indices, tile, isscalar, stack 
    # from numpy import lib
    from jax.numpy import load
    from numpy import random

elif gpu_enabled:
    # from cupy import *
    if debug:
        print("Using cupy for GPU acceleration.")
    from cupy import newaxis, array_split, asarray, min, power, max, ndarray, add, multiply, sin, linspace, floor, concatenate, arange, zeros_like, zeros, cos, clip, sum, argmin, minimum, sqrt, maximum, broadcast_to, random, pi
    from cupy import subtract, ones_like, meshgrid, log2, array, abs, arctan2, where, square, uint16, divide, sort, absolute, linalg, ogrid, lib, take_along_axis, indices, tile, isscalar, stack 
    from cupy import load
else:
    # from numpy import *
    if debug:
        print("Using numpy for CPU processing.")
    from numpy import newaxis, array_split, asarray, min, power, max, ndarray, add, multiply, sin, linspace, floor, concatenate, arange, zeros_like, zeros, cos, clip, sum, argmin, minimum, sqrt, maximum, broadcast_to, random, pi 
    from numpy import subtract, ones_like, meshgrid, log2, array, abs, arctan2, where, square, uint16, divide, sort, absolute, linalg, ogrid, lib, take_along_axis, indices, tile, isscalar, stack
    from numpy import load

# named_scope tags blocks of computation so their names show up in the HLO/XLA
# graph (see tools/visualize_jax_graph.py). It only exists in jax; on the
# numpy/cupy backends we fall back to a no-op context manager so the same
# `with named_scope(...)` calls work everywhere.
if jax_available:
    from jax import named_scope
else:
    from contextlib import contextmanager as _contextmanager

    @_contextmanager
    def named_scope(name):
        yield

import functools as _functools
from contextlib import contextmanager as _ctxmgr

# --- suppressible "fine" scopes ---------------------------------------------
# Some helpers (closest_point_on_lines, sample_pixelwise_bezier_at_t,
# index_within_subgrid) are only ever called deep inside the river/dendry
# computation. We still want their internal sub-boxes in principle, but NOT
# rendered inside the already-deep scale_rivers section. `fine_scope` behaves
# like named_scope, except it becomes a no-op while `suppress_fine_scopes()` is
# active (which scale_rivers turns on).
_suppress_fine = False


@_ctxmgr
def suppress_fine_scopes():
    global _suppress_fine
    prev = _suppress_fine
    _suppress_fine = True
    try:
        yield
    finally:
        _suppress_fine = prev


@_ctxmgr
def fine_scope(name):
    if _suppress_fine:
        yield
    else:
        with named_scope(name):
            yield


def scoped(fn):
    """Decorator: wrap a function's whole body in named_scope(fn.__name__).

    A cheap way to make each helper appear as its own labelled box in the
    computation graph without re-indenting its body. No-op cost on the
    numpy/cupy backends (named_scope is a no-op there)."""
    name = fn.__name__

    @_functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with named_scope(name):
            return fn(*args, **kwargs)
    return wrapper