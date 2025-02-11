#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

"""

"""

import os
import icepack
import firedrake
import icepack.plot
from meshpy import triangle
from firedrake import as_vector
from firedrake.__future__ import interpolate
from firedrake import Constant
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
)
from firedrake.pyplot import tripcolor, tricontour
import matplotlib.pyplot as plt
from icepackaccs.mismip import Lx, Ly, mismip_bed_topography
from icepackaccs.friction import regularized_coulomb_mismip
from libchannels import run_simulation, fast_opts, ts_name

from partial_stream_hybrid_init import get_C, get_a

field_names = ["velocity", "thickness", "surface"]
points = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]

facets = [(i, (i + 1) % len(points)) for i in range(len(points))]
markers = list(range(1, len(points) + 1))

mesh_info = triangle.MeshInfo()
mesh_info.set_points(points)
mesh_info.set_facets(facets, facet_markers=markers)


def subplots(*args, **kwargs):
    fig, axes = icepack.plot.subplots(*args, **kwargs)
    if hasattr(axes, "__len__"):
        for ax in axes:
            ax.set_aspect(2)
            ax.set_xlim((0, Lx))
            ax.set_ylim((0, Ly))
    else:
        axes.set_aspect(2)
        axes.set_xlim((0, Lx))
        axes.set_ylim((0, Ly))
    return fig, axes


def colorbar(fig, colors, **kwargs):
    return fig.colorbar(colors, fraction=0.012, pad=0.025, **kwargs)


model = icepack.models.IceStream(friction=regularized_coulomb_mismip)
field_names = ["velocity", "thickness", "surface"]

fields_3 = {}
with firedrake.CheckpointFile("outputs/mismip-fine-ssa-15.0kyr.h5", "r") as chk:
    fine_mesh2d = chk.load_mesh(name="fine_mesh2d")
    for key in field_names:
        fields_3[key] = chk.load_function(fine_mesh2d, key)

if not os.path.exists(f"outputs/{ts_name}-fine-ssa-17.5kyr.h5"):
    x = firedrake.SpatialCoordinate(fine_mesh2d)[0]
    Q2 = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
    V2 = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2)

    h_0 = firedrake.project(fields_3["thickness"], Q2)
    u_0 = firedrake.project(fields_3["velocity"], V2)

    z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh2d), Q2))
    s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)

    A = Constant(20)
    C = get_C(fine_mesh2d, Q2)
    a = get_a(fine_mesh2d, Q2)

    u_init = firedrake.assemble(interpolate(as_vector((90 * x / Lx, 0)), V2))

    solver = icepack.solvers.FlowSolver(model, **fast_opts)
    u_0 = solver.diagnostic_solve(velocity=u_init, thickness=h_0, surface=s_0, fluidity=A, friction=C)

    fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

    time = 1000
    dt = 0.3125

    print("Running 1000 years with ultra-fine time steps")
    exception, fields_0 = run_simulation(solver, time, dt, bed=z_b, a=a, A=A, C=C, return_all=False, **fields)

    time = 1500
    dt = 0.625

    print("Running 1500 years with fine time steps")
    exception, fields_init = run_simulation(solver, time, dt, bed=z_b, a=a, A=A, C=C, return_all=False, **fields_0)

    fine_mesh2d.name = "fine_mesh2d"
    with firedrake.CheckpointFile(f"outputs/{ts_name}-fine-ssa-17.5kyr.h5", "w") as chk:
        chk.save_mesh(fine_mesh2d)
        for key in fields_init:
            chk.save_function(fields_init[key], name=key)
else:
    print("Reloading results at 17.5 kyr")
    with firedrake.CheckpointFile(f"outputs/{ts_name}-fine-ssa-17.5kyr.h5", "r") as chk:
        fine_mesh2d = chk.load_mesh("fine_mesh2d")
        fields_init = {}
        for key in field_names:
            fields_init[key] = chk.load_function(fine_mesh2d, key)

        Q2 = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
        V2 = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2)
        z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh2d), Q2))

        A = Constant(20)
        C = get_C(fine_mesh2d, Q2)
        a = get_a(fine_mesh2d, Q2)

        solver = icepack.solvers.FlowSolver(model, **fast_opts)

fn = f"outputs/{ts_name}-fine-ssa-20kyr.h5"
if not os.path.exists(fn):
    time = 2500
    dt = 0.625

    print("Running another 2500 years with fine time steps")
    exception, fields_mid = run_simulation(solver, time, dt, bed=z_b, a=a, A=A, C=C, return_all=False, **fields_init)

    fine_mesh2d.name = "fine_mesh2d"
    with firedrake.CheckpointFile(fn, "w") as chk:
        chk.save_mesh(fine_mesh2d)
        for key in fields_mid:
            chk.save_function(fields_mid[key], name=key)
else:
    print("Reloading results at 20 kyr")
    with firedrake.CheckpointFile(fn, "r") as chk:
        fine_mesh2d = chk.load_mesh("fine_mesh2d")
        fields_mid = {}
        for key in field_names:
            fields_mid[key] = chk.load_function(fine_mesh2d, key)

        Q2 = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
        V2 = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2)
        z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh2d), Q2))

        A = Constant(20)
        C = get_C(fine_mesh2d, Q2)
        a = get_a(fine_mesh2d, Q2)

        solver = icepack.solvers.FlowSolver(model, **fast_opts)
        # fields_mid["velocity"] = solver.diagnostic_solve(velocity=firedrake.project(fields_mid["velocity"] * 0.8, V2), thickness=fields_mid["thickness"], surface=fields_mid["surface"], fluidity=A, friction=C)
        # solver = icepack.solvers.FlowSolver(model, **reliable_opts)
fn = f"outputs/{ts_name}-fine-ssa-22.5kyr.h5"
if not os.path.exists(fn):
    time = 2500
    dt = 1.25 / 2.0

    fig, axes = subplots(2, 1, figsize=(10, 7))
    colors = tripcolor(fields_mid["thickness"], axes=axes[0], vmin=0, vmax=1500)
    plt.colorbar(colors, label="Thickness [m]", ax=axes[0], extend="max")
    colors = tripcolor(
        firedrake.project(fields_mid["velocity"][0], Q2),
        axes=axes[1],
        cmap="Reds",
        vmin=0,
    )
    plt.savefig(f"plots/ssa/init/{ts_name}_at_22.5kyr.png", dpi=300)

    print("Running another 2500 years with fine time steps")
    exception, fields_fine = run_simulation(solver, time, dt, bed=z_b, a=a, A=A, C=C, return_all=False, **fields_mid)
    # solver = icepack.solvers.FlowSolver(model, **reliable_opts)
    # exception, fields_fine = run_simulation(solver, time - 200, dt, bed=z_b, a=a, A=A, C=C, return_all=False, **fields_fine)
    if exception:
        fine_mesh2d.name = "fine_mesh2d"
        with firedrake.CheckpointFile(fn, "w") as chk:
            chk.save_mesh(fine_mesh2d)
            for key in fields_fine:
                chk.save_function(fields_fine[key], name=key)
        raise exception

    fine_mesh2d.name = "fine_mesh2d"
    with firedrake.CheckpointFile(fn, "w") as chk:
        chk.save_mesh(fine_mesh2d)
        for key in fields_fine:
            chk.save_function(fields_fine[key], name=key)
else:
    print("Reloading results at 22.5 kyr")
    with firedrake.CheckpointFile(fn, "r") as chk:
        fine_mesh2d = chk.load_mesh("fine_mesh2d")
        fields_fine = {}
        for key in field_names:
            fields_fine[key] = chk.load_function(fine_mesh2d, key)

        Q2 = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
        V2 = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2)
        z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh2d), Q2))

        A = Constant(20)
        C = get_C(fine_mesh2d, Q2)
        a = get_a(fine_mesh2d, Q2)

        solver = icepack.solvers.FlowSolver(model, **fast_opts)
        # fields_fine["velocity"] = solver.diagnostic_solve(velocity=firedrake.project(fields_fine["velocity"] * 0.8, V2), thickness=fields_fine["thickness"], surface=fields_fine["surface"], fluidity=A, friction=C)
        # solver = icepack.solvers.FlowSolver(model, **reliable_opts)

if not os.path.exists(f"outputs/{ts_name}-fine-ssa-30kyr.h5"):
    time = 7500
    dt = 0.625

    print("Running another 7500 years with fine time steps")
    exception, fields_final = run_simulation(solver, time, dt, bed=z_b, a=a, A=A, C=C, return_all=False, **fields_fine)
    # solver = icepack.solvers.FlowSolver(model, **reliable_opts)
    # exception, fields_final = run_simulation(solver, time - 200, dt, bed=z_b, a=a, A=A, C=C, return_all=False, **fields_final)
    if exception:
        fine_mesh2d.name = "fine_mesh2d"
        with firedrake.CheckpointFile(f"outputs/{ts_name}-fine-ssa-30kyr_error.h5", "w") as chk:
            chk.save_mesh(fine_mesh2d)
            for key in fields_final:
                chk.save_function(fields_final[key], name=key)
        raise exception

    fine_mesh2d.name = "fine_mesh2d"
    with firedrake.CheckpointFile(f"outputs/{ts_name}-fine-ssa-30kyr.h5", "w") as chk:
        chk.save_mesh(fine_mesh2d)
        for key in fields_final:
            chk.save_function(fields_final[key], name=key)
else:
    print("Reloading results at 20 kyr")
    with firedrake.CheckpointFile(f"outputs/{ts_name}-fine-ssa-30kyr.h5", "r") as chk:
        fine_mesh2d = chk.load_mesh("fine_mesh2d")
        fields_final = {}
        for key in field_names:
            fields_final[key] = chk.load_function(fine_mesh2d, key)

        Q2 = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
        V2 = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2)
        z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh2d), Q2))

        A = Constant(20)
        C = get_C(fine_mesh2d, Q2)
        a = get_a(fine_mesh2d, Q2)

        solver = icepack.solvers.FlowSolver(model, **fast_opts)
        # fields_final["velocity"] = solver.diagnostic_solve(velocity=firedrake.project(fields_final["velocity"] * 0.8, V2), thickness=fields_final["thickness"], surface=fields_final["surface"], fluidity=A, friction=C)
        # solver = icepack.solvers.FlowSolver(model, **reliable_opts)

fig, axes = subplots(2, 1, figsize=(10, 7))
colors = tripcolor(fields_final["thickness"], axes=axes[0], vmin=0, vmax=1500)
plt.colorbar(colors, label="Thickness [m]", ax=axes[0], extend="max")
colors = tripcolor(
    firedrake.project(fields_final["velocity"][0], Q2),
    axes=axes[1],
    cmap="Reds",
    vmin=0,
)
plt.colorbar(colors, label=r"Velocity [m yr$^{-1}$]", ax=axes[1], extend="max")

s = fields_final["surface"]
h = fields_final["thickness"]
height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h, Q2))
levels = [5]
tricontour(height_above_flotation, levels=levels, axes=axes[0], colors=["k"])
tricontour(height_above_flotation, levels=levels, axes=axes[1], colors=["k"])
fig.savefig(f"plots/ssa/init/{ts_name}_fine_thick_reinit.png", dpi=300)
