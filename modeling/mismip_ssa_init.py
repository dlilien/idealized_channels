#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

import os
from firedrake import Constant, dx, dS, as_vector
from firedrake.__future__ import interpolate
from icepackaccs.friction import regularized_coulomb_mismip
from icepackaccs.mismip import mismip_bed_topography, Lx, Ly
from libchannels import (
    trapezoidal_channel,
    sidehugging_trapezoidal_channels,
    run_simulation,
    fast_opts,
)
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
)
import icepack
import firedrake
import icepack.plot
from meshpy import triangle
from firedrake.pyplot import triplot, tripcolor, tricontour
from hybrid_channels import margin_y, standard_x


chk_template = "outputs/mismip-fine-ssa-{:2.1f}kyr.h5"


def subplots(**kwargs):
    fig, axes = icepack.plot.subplots(**kwargs)
    axes.set_aspect(2)
    axes.set_xlim((0, Lx))
    axes.set_ylim((0, Ly))
    return fig, axes


def colorbar(fig, colors, **kwargs):
    return fig.colorbar(colors, fraction=0.012, pad=0.025, **kwargs)


total_time = 0
field_names = ["thickness", "velocity", "surface"]

points = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]

facets = [(i, (i + 1) % len(points)) for i in range(len(points))]
markers = list(range(1, len(points) + 1))

mesh_info = triangle.MeshInfo()
mesh_info.set_points(points)
mesh_info.set_facets(facets, facet_markers=markers)

δy = Ly / 32
area = δy**2 / 2
triangle_mesh = triangle.build(mesh_info, max_volume=area)
coarse_mesh = icepack.meshing.triangle_to_firedrake(triangle_mesh)
coarse_mesh.name = "coarse_mesh"

A = Constant(20)
C = Constant(1e-2)
a = Constant(0.3)

model = icepack.models.IceStream(friction=regularized_coulomb_mismip)

# ### First run
# In order to get a better idea of where we need to refine the mesh, we'll start by looking at the results of relatively low-resolution simulations.
# Since we'll be running the same simulation many times, we'll again wrap up the code in a function that we can call repeatedly.

coarse_fn = "outputs/mismip-mesh-coarse2d.h5"
med_mesh_fn = "outputs/mismip-mesh-medium2d.h5"
incremental_time = 5000.0

Q1c = firedrake.FunctionSpace(coarse_mesh, "CG", 1)
V1c = firedrake.VectorFunctionSpace(coarse_mesh, "CG", 1)
Q2c = firedrake.FunctionSpace(coarse_mesh, "CG", 2)
V2c = firedrake.VectorFunctionSpace(coarse_mesh, "CG", 2)

if (not os.path.exists(coarse_fn)) or (not os.path.exists(med_mesh_fn)):
    print("Doing two coarse runs to make medium mesh")

    z_b1 = firedrake.assemble(interpolate(mismip_bed_topography(coarse_mesh), Q1c))
    h_01 = firedrake.assemble(interpolate(Constant(100), Q1c))
    s_01 = icepack.compute_surface(thickness=h_01, bed=z_b1)
    x = firedrake.SpatialCoordinate(coarse_mesh)[0]
    solver_1c = icepack.solvers.FlowSolver(model, **fast_opts)
    u_01 = solver_1c.diagnostic_solve(
        velocity=firedrake.assemble(interpolate(as_vector((90 * x / Lx, 0)), V1c)),
        thickness=h_01,
        surface=s_01,
        fluidity=A,
        friction=C,
    )
    fields_0c1 = {"surface": s_01, "thickness": h_01, "velocity": u_01}

    z_b = firedrake.assemble(interpolate(mismip_bed_topography(coarse_mesh), Q2c))
    h_0 = firedrake.assemble(interpolate(Constant(100), Q2c))
    s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
    x = firedrake.SpatialCoordinate(coarse_mesh)[0]
    solver_2c = icepack.solvers.FlowSolver(model, **fast_opts)
    u_0 = solver_2c.diagnostic_solve(
        velocity=firedrake.assemble(interpolate(as_vector((90 * x / Lx, 0)), V2c)),
        thickness=h_0,
        surface=s_0,
        fluidity=A,
        friction=C,
    )
    fields_0c2 = {"thickness": h_0, "surface": s_0, "velocity": u_0}

    dt = 2.0

    print("Running coarse, piecewise-linear simulation for {:d} years".format(int(incremental_time)))
    exception, fields_1c = run_simulation(solver_1c, incremental_time, dt, bed=z_b1, a=a, A=A, C=C, **fields_0c1)

    fig, axes = subplots()
    colors = tripcolor(fields_1c["thickness"], axes=axes)
    colorbar(fig, colors, label="Thickness [m]")
    fig.savefig("plots/ssa/init/coarse_thick_1stOrder.png", dpi=300)

    print("Running coarse, piecewise-quadratic simulation for {:d} years".format(int(incremental_time)))
    exception, fields_2c = run_simulation(solver_2c, incremental_time, dt, bed=z_b, a=a, A=A, C=C, **fields_0c2)

    fig, axes = subplots()
    colors = tripcolor(fields_2c["thickness"], axes=axes)
    colorbar(fig, colors, label="Thickness [m]")
    fig.savefig("plots/ssa/init/coarse_thick_2ndOrder.png", dpi=300)

    expr = abs(fields_2c["thickness"] - fields_1c["thickness"])
    δhc = firedrake.assemble(interpolate(expr, Q2c))

    fig, axes = subplots()
    colors = tripcolor(δhc, axes=axes)
    colorbar(fig, colors, label="Thickness difference [m]")
    fig.savefig("plots/ssa/init/thickness_difference_coarse.png", dpi=300)

    s = fields_2c["surface"]
    h = fields_2c["thickness"]
    height_above_flotationc = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h, Q2c))
    x, y = firedrake.SpatialCoordinate(coarse_mesh)
    refine_glc = firedrake.project(
        firedrake.conditional(height_above_flotationc < 75.0, 275, 0)
        * firedrake.conditional(height_above_flotationc > 2.0, 1, 0)
        * firedrake.conditional(x > 5e5, 1, 0),
        Q2c,
    )

    δh2c = firedrake.project(δhc + refine_glc, Q2c)

    DG0 = firedrake.FunctionSpace(coarse_mesh, "DG", 0)
    ϵc = firedrake.Function(DG0)
    J = 0.5 * ((ϵc - δh2c) ** 2 * dx + (Ly / 2) * (ϵc("+") - ϵc("-")) ** 2 * dS)
    F = firedrake.derivative(J, ϵc)
    firedrake.solve(F == 0, ϵc)

    fig, axes = subplots()
    colors = tripcolor(ϵc, axes=axes)
    colorbar(fig, colors, label="Thickness difference [m]")
    fig.savefig("plots/ssa/init/thickness_difference_smooth_coarse.png", dpi=300)

    # Make medium mesh
    triangle_mesh.element_volumes.setup()
    expr = firedrake.CellVolume(coarse_mesh)
    areas = firedrake.project(expr, DG0)
    shrink = 5
    exponent = 2
    max_err = ϵc.dat.data_ro[:].max()
    num_triangles = len(triangle_mesh.elements)
    for index, err in enumerate(ϵc.dat.data_ro[:]):
        area = areas.dat.data_ro[index]
        shrink_factor = shrink * (err / max_err) ** exponent
        triangle_mesh.element_volumes[index] = area / (1 + shrink_factor)
    refined_triangle_mesh = triangle.refine(triangle_mesh)
    med_mesh = icepack.meshing.triangle_to_firedrake(refined_triangle_mesh)
    med_mesh.name = "med_mesh"

    with firedrake.CheckpointFile(coarse_fn, "w") as chk:
        chk.save_mesh(coarse_mesh)
        chk.save_function(δhc, name="dhc")
        chk.save_function(refine_glc, name="refine_glc")
        chk.save_function(ϵc, name="eps_c")
        for key, val in fields_2c.items():
            chk.save_function(val, name=key)
    with firedrake.CheckpointFile(med_mesh_fn, "w") as chk:
        chk.save_mesh(med_mesh)
else:
    print("Loading " + coarse_fn + " and " + med_mesh_fn)
total_time += incremental_time

with firedrake.CheckpointFile(coarse_fn, "r") as chk:
    coarse_mesh = chk.load_mesh("coarse_mesh")
    δhc = chk.load_function(coarse_mesh, "dhc")
    refine_glc = chk.load_function(coarse_mesh, "refine_glc")
    ϵc = chk.load_function(coarse_mesh, "eps_c")
    fields_2c = {}
    for key in field_names:
        fields_2c[key] = chk.load_function(coarse_mesh, key)
with firedrake.CheckpointFile(med_mesh_fn, "r") as chk:
    med_mesh = chk.load_mesh("med_mesh")

# Recreate these to avoid weird errors
Q1c = firedrake.FunctionSpace(coarse_mesh, "CG", 1)
V1c = firedrake.VectorFunctionSpace(coarse_mesh, "CG", 1)
Q2c = firedrake.FunctionSpace(coarse_mesh, "CG", 2)
V2c = firedrake.VectorFunctionSpace(coarse_mesh, "CG", 2)

# ### Second run
#
# Now that we have a refined mesh, we can project our old solutions on the coarse mesh to it and run the physics out for a further several thousand years to get even closer to the equilibrium solution. Note that this is necessary because the grounding line is moving substantially, so we are not going to get it in the correct place otherwise.
fine_input_fn = "outputs/mismip-fine-ssa-10kyr.h5"
med_err_fn = "outputs/mismip-error-med.h5"
incremental_time = 5000.0
if (not os.path.exists(fine_input_fn)) or (not os.path.exists(med_err_fn)):
    print(
        "Running medium resolution to refine mesh further (final time will be {:d} years)".format(
            int(total_time + incremental_time)
        )
    )
    dt = 1.0

    Q1 = firedrake.FunctionSpace(med_mesh, "CG", 1)
    V1 = firedrake.VectorFunctionSpace(med_mesh, "CG", 1)

    h_0 = firedrake.project(fields_2c["thickness"], Q1)
    u_0 = firedrake.project(fields_2c["velocity"], V1)
    z_b = firedrake.assemble(interpolate(mismip_bed_topography(med_mesh), Q1))
    s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
    solver = icepack.solvers.FlowSolver(model, **fast_opts)
    u_0 = solver.diagnostic_solve(velocity=u_0, thickness=h_0, surface=s_0, fluidity=A, friction=C)
    fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

    print("Running medium, piecewise-linear simulation for {:d} years".format(int(incremental_time)))
    exception, fields_1 = run_simulation(solver, incremental_time, dt, bed=z_b, a=a, A=A, C=C, **fields)

    Q2 = firedrake.FunctionSpace(med_mesh, "CG", 2)
    V2 = firedrake.VectorFunctionSpace(med_mesh, "CG", 2)

    h_0 = firedrake.project(fields_2c["thickness"], Q2)
    u_0 = firedrake.project(fields_2c["velocity"], V2)
    z_b = firedrake.assemble(interpolate(mismip_bed_topography(med_mesh), Q2))
    s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
    solver = icepack.solvers.FlowSolver(model, **fast_opts)
    u_0 = solver.diagnostic_solve(velocity=u_0, thickness=h_0, surface=s_0, fluidity=A, friction=C)
    fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

    print("Running medium, piecewise-quadratic simulation for {:d} years".format(int(incremental_time)))
    exception, fields_2 = run_simulation(solver, incremental_time, dt, bed=z_b, a=a, A=A, C=C, **fields)

    expr = abs(fields_2["thickness"] - firedrake.assemble(interpolate(fields_1["thickness"], Q2)))
    δh = firedrake.assemble(interpolate(expr, Q2))

    s = firedrake.project(fields_2["surface"], Q2c)
    h = firedrake.project(fields_2["thickness"], Q2c)
    height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h, Q2c))
    x, y = firedrake.SpatialCoordinate(coarse_mesh)
    refine_gl = firedrake.project(
        firedrake.conditional(height_above_flotation < 75.0, 150, 0)
        * firedrake.conditional(height_above_flotation > 2.0, 1, 0)
        * firedrake.conditional(x > 5e5, 1, 0),
        Q2c,
    )

    δhf = firedrake.project(δh, Q2c)

    refine_chans = firedrake.assemble(
        interpolate(
            sidehugging_trapezoidal_channels(coarse_mesh, 250.0, standard_x, margin_y, y_width=5000.0),
            Q2c,
        )
    )
    δh2 = firedrake.project(δhf + refine_chans + refine_gl, Q2c)

    DG0 = firedrake.FunctionSpace(coarse_mesh, "DG", 0)
    ϵ = firedrake.Function(DG0)
    J = 0.5 * ((ϵ - δh2) ** 2 * dx + (Ly / 2) * (ϵ("+") - ϵ("-")) ** 2 * dS)
    F = firedrake.derivative(J, ϵ)
    firedrake.solve(F == 0, ϵ)
    fig, axes = subplots()
    colors = tripcolor(firedrake.project(ϵ / ϵ.dat.data_ro[:].max(), DG0), axes=axes)
    colorbar(fig, colors)
    fig.savefig("plots/ssa/init/normalized_smooth_err_med.png", dpi=300)

    fig, axes = subplots()
    colors = tripcolor(firedrake.project(ϵc / ϵc.dat.data_ro[:].max(), DG0), axes=axes)
    colorbar(fig, colors)
    fig.savefig("plots/ssa/init/normalized_smooth_err_coarse.png", dpi=300)

    triangle_mesh = triangle.build(mesh_info, max_volume=area)
    triangle_mesh.element_volumes.setup()
    coarse_mesh = icepack.meshing.triangle_to_firedrake(triangle_mesh)
    coarse_mesh.name = "coarse_mesh"

    DG0c = firedrake.FunctionSpace(coarse_mesh, "DG", 0)
    ϵ = firedrake.project(ϵ, DG0c)

    expr = firedrake.CellVolume(coarse_mesh)
    areas = firedrake.project(expr, DG0c)

    shrink = 10
    exponent = 2
    max_err = ϵ.dat.data_ro[:].max()

    num_triangles = len(triangle_mesh.elements)
    for index, err in enumerate(ϵ.dat.data_ro[:]):
        area = areas.dat.data_ro[index]
        shrink_factor = shrink * (err / max_err) ** exponent
        triangle_mesh.element_volumes[index] = area / (1 + shrink_factor)

    refined_triangle_mesh = triangle.refine(triangle_mesh)

    fine_mesh = icepack.meshing.triangle_to_firedrake(refined_triangle_mesh)
    fine_mesh.name = "fine_mesh"
    with firedrake.CheckpointFile("outputs/mismip-mesh-fine2d.h5", "w") as chk:
        chk.save_mesh(fine_mesh)

    Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2)
    V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2)
    fields_2["velocity"] = firedrake.project(fields_2["velocity"], V2)
    fields_2["thickness"] = firedrake.project(fields_2["thickness"], Q2)
    fields_2["surface"] = firedrake.project(fields_2["surface"], Q2)

    fig, axes = subplots(figsize=(14, 6))
    colors = tripcolor(fields_2["thickness"], axes=axes, vmin=200, vmax=800)
    colorbar(fig, colors, extend="both", label="Thickness [m]")
    s = fields_2["surface"]
    h = fields_2["thickness"]
    height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h, Q2))
    levels = [5]
    contours = tricontour(height_above_flotation, levels=levels, axes=axes, colors=["k"])
    axes.set_xlim(40e4, 64e4)

    fig, axes = subplots(figsize=(14, 6))
    colors = tripcolor(
        firedrake.project(fields_2["velocity"][0], Q2),
        axes=axes,
        cmap="Reds",
        vmin=0,
        vmax=750,
    )
    colorbar(fig, colors, label="Speed [m yr$^{-1}$]", extend="max")
    s = fields_2["surface"]
    h = fields_2["thickness"]
    height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h, Q2))
    levels = [5]
    contours = tricontour(height_above_flotation, levels=levels, axes=axes, colors=["k"])
    axes.set_xlim(40e4, 64e4)

    with firedrake.CheckpointFile(fine_input_fn, "w") as chk:
        chk.save_mesh(fine_mesh)
        for key in fields_2:
            chk.save_function(fields_2[key], name=key)

    with firedrake.CheckpointFile(med_err_fn, "w") as chk:
        chk.save_mesh(coarse_mesh)
        chk.save_function(δhf, name="dhf")
        chk.save_function(δh2, name="dh2")
        chk.save_function(refine_gl, name="refine_gl")
        chk.save_function(ϵ, name="eps")

total_time += incremental_time

fields_3 = {}
with firedrake.CheckpointFile(fine_input_fn, "r") as chk:
    fine_mesh2d = chk.load_mesh(name="fine_mesh")
    for key in field_names:
        fields_3[key] = chk.load_function(fine_mesh2d, key)
with firedrake.CheckpointFile(med_err_fn, "r") as chk:
    coarse_mesh = chk.load_mesh("coarse_mesh")
    δhf = chk.load_function(coarse_mesh, "dhf")
    δh2 = chk.load_function(coarse_mesh, "dh2")
    refine_gl = chk.load_function(coarse_mesh, "refine_gl")
    # ϵ = chk.load_function(coarse_mesh, "eps")

Q2 = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
V2 = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2)

h_0 = firedrake.project(fields_3["thickness"], Q2)
u_0 = fields_3["velocity"]
z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh2d), Q2))
s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
x = firedrake.SpatialCoordinate(fine_mesh2d)[0]
# u_init = firedrake.assemble(interpolate(as_vector((90 * x / Lx, 0)), V2))
solver = icepack.solvers.FlowSolver(model, **fast_opts)
u_0 = solver.diagnostic_solve(velocity=u_0, thickness=h_0, surface=s_0, fluidity=A, friction=C)
fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

incremental_time = 2500.0
total_time += incremental_time
dt = 1.0

fine_fn = "outputs/mismip-fine-ssa-12.5kyr.h5"
if not os.path.exists(fine_fn):
    print("Running fine, piecewise-quadratic simulation for {:d} years".format(int(incremental_time)))
    exception, fields_2 = run_simulation(solver, incremental_time, dt, bed=z_b, a=a, A=A, C=C, **fields)

    fine_mesh2d.name = "fine_mesh2d"
    with firedrake.CheckpointFile(fine_fn, "w") as chk:
        chk.save_mesh(fine_mesh2d)
        for key in fields_2:
            chk.save_function(fields_2[key], name=key)

print("Loading fine results")
with firedrake.CheckpointFile(fine_fn, "r") as chk:
    fine_mesh2d = chk.load_mesh("fine_mesh2d")
    fields_2 = {}
    for key in field_names:
        fields_2[key] = chk.load_function(fine_mesh2d, key)

Q2 = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
V2 = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2)
z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh2d), Q2))

Q2c2 = firedrake.FunctionSpace(coarse_mesh, "CG", 2)

s = fields_2["surface"]
h = fields_2["thickness"]
height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h, Q2))
height_above_flotation = firedrake.project(height_above_flotation, Q2c2)
x, y = firedrake.SpatialCoordinate(coarse_mesh)
refine_gl = firedrake.project(
    firedrake.conditional(height_above_flotation < 75.0, 500, 0)
    * firedrake.conditional(height_above_flotation > 2.0, 1, 0)
    * firedrake.conditional(x > 5e5, 1, 0),
    Q2c2,
)

refine_chans = firedrake.assemble(
    interpolate(
        trapezoidal_channel(coarse_mesh, 750.0, 4.5e5, 40e3, y_width=5000.0, ramp_x=5e4)
        + sidehugging_trapezoidal_channels(coarse_mesh, 750.0, standard_x, margin_y, y_width=5000.0, ramp_x=5e4),
        Q2c2,
    )
)
δh2_2 = firedrake.project(
    firedrake.project(δhf, Q2c2) + firedrake.project(δhc, Q2c2) + refine_chans + refine_gl,
    Q2c2,
)

triangle_mesh = triangle.build(mesh_info, max_volume=area)
triangle_mesh.element_volumes.setup()
coarse_mesh = icepack.meshing.triangle_to_firedrake(triangle_mesh)
coarse_mesh.name = "coarse_mesh"

DG02 = firedrake.FunctionSpace(coarse_mesh, "DG", 0)
ϵ2 = firedrake.Function(DG02)
δh2_2c = firedrake.project(δh2_2, DG02)
J = 0.5 * ((ϵ2 - δh2_2c) ** 2 * dx + (Ly / 2) * (ϵ2("+") - ϵ2("-")) ** 2 * dS)
F = firedrake.derivative(J, ϵ2)
firedrake.solve(F == 0, ϵ2)
fig, axes = subplots()
colors = tripcolor(firedrake.project(ϵ2 / ϵ2.dat.data_ro[:].max(), DG02), axes=axes)
colorbar(fig, colors)

expr = firedrake.CellVolume(coarse_mesh)
areas = firedrake.project(expr, DG02)

shrink = 10
exponent = 2
max_err = ϵ2.dat.data_ro[:].max()

num_triangles = len(triangle_mesh.elements)
for index, err in enumerate(ϵ2.dat.data_ro[:]):
    area = areas.dat.data_ro[index]
    shrink_factor = shrink * (err / max_err) ** exponent
    triangle_mesh.element_volumes[index] = area / (1 + shrink_factor)

refined_triangle_mesh2 = triangle.refine(triangle_mesh)
refine_mesh = icepack.meshing.triangle_to_firedrake(refined_triangle_mesh2)
refine_mesh.name = "fine_mesh"

with firedrake.CheckpointFile("outputs/mismip-mesh-refine2d.h5", "w") as chk:
    chk.save_mesh(refine_mesh)

fig, axes = icepack.plot.subplots(figsize=(8, 4))
axes.set_xlim((400e3, 600e3))
axes.set_ylim((0, 80e3))
axes.get_yaxis().set_visible(False)

levels = [5]
contours = tricontour(height_above_flotation, levels=levels, axes=axes, colors=["r"])
triplot(refine_mesh, axes=axes)
fig.savefig("plots/ssa/init/mesh_refined_ssa.pdf")

Q2 = firedrake.FunctionSpace(refine_mesh, "CG", 2)
V2 = firedrake.VectorFunctionSpace(refine_mesh, "CG", 2)

solver = icepack.solvers.FlowSolver(model, **fast_opts)
h_0 = firedrake.project(fields_3["thickness"], Q2)
u_0 = firedrake.project(fields_3["velocity"], V2)
z_b = firedrake.assemble(interpolate(mismip_bed_topography(refine_mesh), Q2))
s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
x = firedrake.SpatialCoordinate(refine_mesh)[0]
u_0 = solver.diagnostic_solve(velocity=u_0, thickness=h_0, surface=s_0, fluidity=A, friction=C)
# u_init = firedrake.assemble(interpolate(as_vector((90 * x / Lx, 0)), V2))

for real_time in [15.0, 17.5, 20.0]:
    if not os.path.exists(chk_template.format(real_time)):
        fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

        final_time = 2500.0
        dt = 1.25

        print("Running (re)fine, piecewise-quadratic simulation for {:d} years".format(int(final_time)))
        exception, fields_4 = run_simulation(solver, final_time, dt, bed=z_b, a=a, A=A, C=C, **fields)

        refine_mesh.name = "fine_mesh2d"
        with firedrake.CheckpointFile(chk_template.format(real_time), "w") as chk:
            chk.save_mesh(refine_mesh)
            for key in fields_4:
                chk.save_function(fields_4[key], name=key)
    else:
        print("Loading (re)fine, piecewise-quadratic simulation at {:2.1f} kyr".format(real_time))
        with firedrake.CheckpointFile(chk_template.format(real_time), "r") as chk:
            refine_mesh = chk.load_mesh("fine_mesh2d")
            fields_4 = {}
            for key in field_names:
                fields_4[key] = chk.load_function(refine_mesh, key)
    h_0 = fields_4["thickness"]
    u_0 = fields_4["velocity"]
    s_0 = fields_4["surface"]
    Q2 = h_0.function_space()
    V2 = u_0.function_space()
    z_b = firedrake.assemble(interpolate(mismip_bed_topography(refine_mesh), Q2))
    solver = icepack.solvers.FlowSolver(model, **fast_opts)
    u_0 = solver.diagnostic_solve(velocity=u_0, thickness=h_0, surface=s_0, fluidity=A, friction=C)
