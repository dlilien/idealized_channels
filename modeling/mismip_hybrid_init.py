#!/usr/bin/env python
# coding: utf-8

import os
from meshpy import triangle

import firedrake
from firedrake.__future__ import interpolate
from firedrake.pyplot import tripcolor
import icepack
import icepack.plot
from icepack.utilities import lift3d

from icepackaccs import extract_surface
from icepackaccs.mismip import Lx, Ly, mismip_bed_topography
from icepackaccs.friction import regularized_coulomb_mismip
from icepackaccs.stress import von_mises_stress_vel

from libchannels import run_simulation, fast_opts
from firedrake.petsc import PETSc


field_names = ["surface", "thickness", "velocity"]

points = [(0, 0), (Lx, 0), (Lx, Ly), (0, Ly)]
δy = Ly / 10
area = δy**2 / 2

fields_ssa = None


def subplots(**kwargs):
    fig, axes = icepack.plot.subplots(figsize=(10, 4))
    axes.set_aspect(2)
    axes.set_xlim((0, Lx))
    axes.set_ylim((0, Ly))
    return fig, axes


def colorbar(fig, colors, **kwargs):
    return fig.colorbar(colors, fraction=0.012, pad=0.025, **kwargs)


A = firedrake.Constant(20)
C = firedrake.Constant(1e-2)
a = firedrake.Constant(0.3)

model = icepack.models.HybridModel(friction=regularized_coulomb_mismip)

dt = 5.0
final_time = 3600.0

opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "icepack",
    "diagnostic_solver_parameters": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "tolerance": 1e-8,
    },
}
solver = icepack.solvers.FlowSolver(model, **opts)


if not os.path.exists("outputs/mismip-fine-degree2.h5"):
    if not os.path.exists("outputs/mismip-fine-ssa-15.0kyr.h5"):
        facets = [(i, (i + 1) % len(points)) for i in range(len(points))]
        markers = list(range(1, len(points) + 1))

        mesh_info = triangle.MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets, facet_markers=markers)

        triangle_mesh = triangle.build(mesh_info, max_volume=area)
        coarse_mesh2d = icepack.meshing.triangle_to_firedrake(triangle_mesh)
        coarse_mesh = firedrake.ExtrudedMesh(coarse_mesh2d, layers=1, name="coarse_mesh")

        Q1 = firedrake.FunctionSpace(coarse_mesh, "CG", 1, vfamily="R", vdegree=0)
        V1 = firedrake.VectorFunctionSpace(coarse_mesh, "CG", 1, dim=2, vfamily="GL", vdegree=0)

        z_b = firedrake.assemble(interpolate(mismip_bed_topography(coarse_mesh), Q1))
        h_0 = firedrake.assemble(interpolate(firedrake.Constant(100), Q1))
        s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)

        x = firedrake.SpatialCoordinate(coarse_mesh)[0]
        u_0 = solver.diagnostic_solve(
            velocity=firedrake.assemble(interpolate(firedrake.as_vector((90 * x / Lx, 0)), V1)),
            thickness=h_0,
            surface=s_0,
            fluidity=A,
            friction=C,
        )

        fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

        # In[14]:

        exception, fields_1 = run_simulation(solver, final_time, dt, bed=z_b, a=a, C=C, A=A, **fields)

        with firedrake.CheckpointFile("outputs/mismip-coarse-degree1.h5", "w") as chk:
            chk.save_mesh(coarse_mesh)
            for key in fields:
                chk.save_function(fields[key], name=key)

        # We were able to compute the results fairly fast, but the coarse mesh resolution is very obvious in the plot of the solution below.
        # There are clearly spurious artifacts in the shear margins at the top and bottom of the domain.

        # In[15]:

        fig, axes = subplots()
        colors = firedrake.tripcolor(extract_surface(fields_1["thickness"]), axes=axes)
        colorbar(fig, colors)

        # Now we'll repeat the same simulation again at higher resolution by using piecewise quadratic instead of piecewise linear basis functions.

        # In[16]:

        Q2 = firedrake.FunctionSpace(coarse_mesh, "CG", 2, vfamily="R", vdegree=0)
        V2 = firedrake.VectorFunctionSpace(coarse_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)

        # In[17]:

        z_b = firedrake.assemble(interpolate(mismip_bed_topography(coarse_mesh), Q2))

        # In[18]:

        h_0 = firedrake.assemble(interpolate(firedrake.Constant(100), Q2))
        s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)

        x = firedrake.SpatialCoordinate(coarse_mesh)[0]
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_0 = solver.diagnostic_solve(
            velocity=firedrake.assemble(interpolate(firedrake.as_vector((90 * x / Lx, 0)), V2)),
            thickness=h_0,
            surface=s_0,
            fluidity=A,
            friction=C,
        )

        fields = {"thickness": h_0, "surface": s_0, "velocity": u_0}

        # In[19]:

        exception, fields_2 = run_simulation(solver, final_time, dt, bed=z_b, a=a, C=C, A=A, **fields)

        # To get an idea of where we're making the largest errors, we can look at the discrepancy between the degree-1 and degree-2 simulations.

        # In[20]:

        expr = abs(fields_2["thickness"] - fields_1["thickness"])
        δh = firedrake.assemble(interpolate(expr, Q2))

        fig, axes = subplots()
        colors = firedrake.tripcolor(extract_surface(δh), axes=axes)
        colorbar(fig, colors)

        DG0 = firedrake.FunctionSpace(coarse_mesh2d, "DG", 0)
        ϵ = firedrake.Function(DG0)
        J = 0.5 * ((ϵ - extract_surface(δh)) ** 2 * firedrake.dx + (Ly / 2) * (ϵ("+") - ϵ("-")) ** 2 * firedrake.dS)
        F = firedrake.derivative(J, ϵ)
        firedrake.solve(F == 0, ϵ)

        # In[22]:

        fig, axes = subplots()
        colors = firedrake.tripcolor(ϵ, axes=axes)
        colorbar(fig, colors)

        # The `element_volumes` member of the Triangle mesh data structure contains an array that we'll fill in order to specify the desired triangle areas in the refined mesh.
        # This array isn't initialized by default.
        # The setup routine below allocates space for it.

        # In[23]:

        triangle_mesh.element_volumes.setup()

        # Now we have to make some decisions about how much to actually refine the mesh.
        # Here we'll specify arbitrarily that the triangles with the largest errors will have their areas shrunk by a factor of 8.
        # We then have to decide how much to shrink the areas of triangles with less than the largest error.
        # The scaling could be linear, or quadratic, or the square root -- this is up to us.
        # For this problem, we'll use a quadratic scaling; this makes for fewer triangles than if we had used linear scaling.

        # In[24]:

        expr = firedrake.CellVolume(coarse_mesh2d)
        areas = firedrake.project(expr, DG0)

        shrink = 8
        exponent = 2
        max_err = ϵ.dat.data_ro[:].max()

        num_triangles = len(triangle_mesh.elements)
        for index, err in enumerate(ϵ.dat.data_ro[:]):
            area = areas.dat.data_ro[index]
            shrink_factor = shrink * (err / max_err) ** exponent
            triangle_mesh.element_volumes[index] = area / (1 + shrink_factor)

        refined_triangle_mesh = triangle.refine(triangle_mesh)

        # Once again we'll use the convenience function `triangle_to_firedrake` to convert the Triangle data structure into a Firedrake data structure.

        # In[25]:

        fine_mesh2d = icepack.meshing.triangle_to_firedrake(refined_triangle_mesh)
    else:
        PETSc.Sys.Print("Loading SSA")
        with firedrake.CheckpointFile("outputs/mismip-fine-ssa-15.0kyr.h5", "r") as chk:
            fields_ssa = {}
            fine_mesh2d = chk.load_mesh("fine_mesh2d")
            Q2_2d = firedrake.FunctionSpace(fine_mesh2d, "CG", 2)
            V2_2d = firedrake.VectorFunctionSpace(fine_mesh2d, "CG", 2, dim=2)
            for key in field_names:
                if key == "velocity":
                    fields_ssa[key] = firedrake.project(chk.load_function(fine_mesh2d, key), V2_2d)
                else:
                    fields_ssa[key] = firedrake.project(chk.load_function(fine_mesh2d, key), Q2_2d)

    fine_mesh = firedrake.ExtrudedMesh(fine_mesh2d, layers=1, name="fine_mesh")

    target_time = 17500.0

    Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
    V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)

    z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))
    if fields_ssa is not None:
        PETSc.Sys.Print("Interpolating thickness")
        h_0 = lift3d(fields_ssa["thickness"], Q2)
        time_so_far = 15000.0
        dt = 0.5
    else:
        h_0 = firedrake.assemble(interpolate(firedrake.Constant(100), Q2))
        time_so_far = 0.0
        dt = 1.0

    s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
    x = firedrake.SpatialCoordinate(fine_mesh)[0]
    u_init = firedrake.assemble(interpolate(firedrake.as_vector((90 * x / Lx, 0)), V2))
    solver = icepack.solvers.FlowSolver(model, **opts)

    PETSc.Sys.Print("Getting initial velocity")
    u_0 = solver.diagnostic_solve(
        velocity=u_init,
        thickness=h_0,
        surface=s_0,
        fluidity=A,
        friction=C,
    )

    fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

    PETSc.Sys.Print(f"Initializing to {target_time} kyr with a {(target_time - time_so_far) / 1000} kyr run")
    exception, fields_2 = run_simulation(solver, target_time - time_so_far, dt, bed=z_b, a=a, C=C, A=A, **fields)

    with firedrake.CheckpointFile("outputs/mismip-fine-degree2.h5", "w") as chk:
        chk.save_mesh(fine_mesh)
        for key in fields_2:
            chk.save_function(fields_2[key], name=key)

    fig, axes = subplots(figsize=(14, 6))
    colors = tripcolor(extract_surface(fields_2["thickness"]), axes=axes)
    colorbar(fig, colors, label="H [m]")
    fig.savefig("initialized_mismip_thickness.png", dpi=300)
else:
    PETSc.Sys.Print("Found initialized h5 checkpoint")


first_hybrid_fn = "outputs/mismip-fine-degree2.h5"
output_fn = "outputs/mismip-fine-degree2_comp.h5"

fields_init = {}
if not os.path.exists(output_fn):
    PETSc.Sys.Print("Reloading init results...")
    time_now = 17500
    with firedrake.CheckpointFile(first_hybrid_fn, "r") as chk:
        fine_mesh = chk.load_mesh("fine_mesh")
        for key in field_names:
            fields_init[key] = chk.load_function(fine_mesh, key)
else:
    PETSc.Sys.Print("Reloading looped results...")
    with firedrake.CheckpointFile(output_fn, "r") as chk:
        fine_mesh = chk.load_mesh("fine_mesh")
        time_now = chk.get_attr("metadata", "total_time")
        for key in field_names:
            fields_init[key] = chk.load_function(fine_mesh, key)

Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)
x = firedrake.SpatialCoordinate(fine_mesh)[0]
# u_init = firedrake.assemble(interpolate(firedrake.as_vector((90 * x / Lx, 0)), V2))
u_init = firedrake.assemble(0.9 * fields_init["velocity"])

solver = icepack.solvers.FlowSolver(model, **fast_opts)
PETSc.Sys.Print("Re-initializing velocity")
u_0 = solver.diagnostic_solve(
    velocity=u_init,
    thickness=fields_init["thickness"],
    surface=fields_init["surface"],
    fluidity=A,
    friction=C,
)


solver = icepack.solvers.FlowSolver(model, **fast_opts)

fields_complete = {
    "surface": fields_init["surface"],
    "thickness": fields_init["thickness"],
    "velocity": u_0,
}
z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))

final_time = 5000.0
increment = 500
dt = 1.0
for i in range(int(final_time / increment)):
    PETSc.Sys.Print("Running and saving years {:d} to {:d}".format(int(i * increment + time_now), int(i + 1) * increment + int(time_now)))
    exception, fields_complete = run_simulation(solver, increment, dt, bed=z_b, a=a, C=C, A=A, plot=False, **fields_complete)

    fields_complete["tvm"] = firedrake.project(von_mises_stress_vel(velocity=fields_complete["velocity"], fluidity=A) * 1000, Q2)
    if exception:
        with firedrake.CheckpointFile("outputs/mismip-fine-degree2_comp_err.h5", "w") as chk:
            chk.create_group("metadata")
            chk.set_attr("metadata", "total_time", increment * (i + 1) + time_now)
            chk.save_mesh(fine_mesh)
            for key in fields_complete:
                chk.save_function(fields_complete[key], name=key)
        raise exception

    with firedrake.CheckpointFile("outputs/mismip-fine-degree2_comp.h5", "w") as chk:
        chk.create_group("metadata")
        chk.set_attr("metadata", "total_time", increment * (i + 1) + time_now)
        chk.save_mesh(fine_mesh)
        for key in fields_complete:
            chk.save_function(fields_complete[key], name=key)
