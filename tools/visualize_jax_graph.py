"""Visualize the XLA/HLO computation graph of a JAX function.

Modern replacement for the removed `jax.xla_computation(fn)(*args)` API used in
https://bnikolic.co.uk/blog/python/jax/2022/02/22/jax-outputgraph-rev.html

In current JAX (tested on 0.10.2) both `jax.xla_computation` and the
`xla_client._xla.hlo_module_to_dot_graph` helper have been removed. The
equivalent object (an `XlaComputation`, which still exposes `as_hlo_dot_graph()`
and `as_hlo_text()`) is now obtained via the lowering API:

    jax.jit(fn).lower(*example_args).compiler_ir("hlo")

Usage:
    # 1) Reproduce the blog's toy example:
    python tools/visualize_jax_graph.py demo --out /tmp/demo

    # 2) Graph this project's JAX height function (WARNING: ~6800 ops, huge):
    python tools/visualize_jax_graph.py height --out /tmp/height --resolution 4

Rendering the .dot to an image needs graphviz (`brew install graphviz`):
    dot  -Tpng graph.dot -o graph.png     # small graphs
    sfdp -Tsvg graph.dot -o graph.svg     # large graphs (scalable layout)
"""
import argparse
from pathlib import Path

import jax
import jax.numpy as jnp


def hlo_computation(fn, *example_args):
    """Return the XlaComputation for `fn` at the given example args.

    `.as_hlo_dot_graph()` and `.as_hlo_text()` on the result are the modern
    equivalents of the blog's `z.as_hlo_dot_graph()` / `z.as_hlo_text()`.
    """
    return jax.jit(fn).lower(*example_args).compiler_ir("hlo")


def dump(z, out_stem: Path):
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    dot_path = out_stem.with_suffix(".dot")
    txt_path = out_stem.with_suffix(".hlo.txt")
    dot_path.write_text(z.as_hlo_dot_graph())
    txt_path.write_text(z.as_hlo_text())
    print(f"wrote {dot_path} ({dot_path.stat().st_size} bytes)")
    print(f"wrote {txt_path}")
    print(f"HLO instruction count (approx): {z.as_hlo_text().count(' = ')}")
    return dot_path


def demo(args):
    """The blog's tanh-gradient example, modern API."""
    def tanh(x):
        y = jnp.exp(-2.0 * x)
        return (1.0 - y) / (1.0 + y)

    def lfn(x):
        return jnp.log(tanh(x).sum())

    z = hlo_computation(jax.grad(lfn), jnp.ones(100))
    dump(z, Path(args.out))


def height(args):
    """This project's JAX height function, mirroring main.py's jit call."""
    import tempfile
    import shutil
    import yaml
    from map_generator.globals import REPO_ROOT
    from map_generator.parameters import load_params, create_landscape
    import map_generator.backend_switch as bk

    src = REPO_ROOT / "params/default"
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(src / "world_params.yaml", tmp / "world_params.yaml")
    img = yaml.safe_load((src / "imaging_params.yaml").read_text())
    img["resolution"] = args.resolution
    img["tile_size"] = -1
    (tmp / "imaging_params.yaml").write_text(yaml.safe_dump(img))

    params = load_params(input_folder=tmp)
    rp = params.root_params
    land = create_landscape(rp.world)
    neg_oct = int(bk.log2(land.lin_sca) - 4)

    # Pass inputs as a dict so the HLO parameter nodes carry real names.
    # x, y and rivernoise flow through as data -> traced named inputs.
    # mountainsca / riversca / neg_octave gate Python if-branches and octave
    # counts, so they MUST stay static (compile-time constants) — they change
    # the graph's structure, not its data flow.
    mountainsca = rp.world.mountain_heights
    riversca = rp.world.river_scale
    fn = lambda d: land.get_height(
        d["x"], d["y"], offs=0.5, fine_offs=1.0,
        mountainsca=mountainsca, neg_octave=neg_oct,
        riversca=riversca, rivernoise=d["rivernoise"],
    )
    z = hlo_computation(
        fn, {"x": params.X, "y": params.Y, "rivernoise": jnp.asarray(0.4)}
    )
    dump(z, Path(args.out))

    # Record the full entry signature (traced inputs + static constants) so the
    # functional view can render it as a legend.
    legend = (
        "get_height parameters\\l"
        "  x, y            = traced inputs (grid coords)\\l"
        "  rivernoise=0.4  = traced input\\l"
        f"  mountainsca={mountainsca:g}   = static constant\\l"
        f"  riversca={riversca:g}      = static constant\\l"
        f"  neg_octave={neg_oct}       = static constant\\l"
    )
    Path(str(args.out) + ".legend.txt").write_text(legend)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="blog tanh-gradient example")
    d.add_argument("--out", default="/tmp/jax_demo", help="output path stem (no extension)")
    d.set_defaults(func=demo)

    h = sub.add_parser("height", help="this project's JAX height function")
    h.add_argument("--out", default="/tmp/jax_height", help="output path stem (no extension)")
    h.add_argument("--resolution", type=int, default=4, help="grid resolution (graph size is op-count, not resolution, dependent)")
    h.set_defaults(func=height)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
