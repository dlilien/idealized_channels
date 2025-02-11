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
from firedrake.__future__ import interpolate
from firedrake import Constant
from icepack.utilities import lift3d

from firedrake.pyplot import tripcolor
import matplotlib.pyplot as plt

from libchannels import run_simulation, fast_opts, ts_name

from icepackaccs import extract_surface
from icepackaccs.mismip import Lx, Ly, mismip_bed_topography
from icepackaccs.friction import regularized_coulomb_mismip
from icepackaccs.stress import von_mises_stress_vel

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


def get_C(mesh, Q, y_w=2.5e3, y0=1.0e-2, y1=1.0e-1):
    y = firedrake.SpatialCoordinate(mesh)[1]
    return firedrake.project(
        firedrake.max_value(firedrake.min_value(y0 + (y - Ly / 2 + y_w) / (2 * y_w) * (y1 - y0), y1), y0),
        Q,
    )


def get_a(mesh, Q, y_w=2.5e3, a0=0.051595188, a1=0.65, x_w=2.5e3):
    x = firedrake.SpatialCoordinate(mesh)[0]
    y = firedrake.SpatialCoordinate(mesh)[1]
    return firedrake.project(
        firedrake.max_value(firedrake.min_value(a0 + (y - Ly / 2 + y_w) / (2 * y_w) * firedrake.max_value((5.0e5 - x + x_w) / (2 * x_w), 0.0) * (a1 - a0), a1), a0),
        Q,
    )


if __name__ == "__main__":
    model = icepack.models.HybridModel(friction=regularized_coulomb_mismip)
    field_names = ["velocity", "thickness", "surface"]

    ssa_fn = "outputs/{:s}-fine-ssa-30kyr.h5".format(ts_name)

    first_hybrid_fn = "outputs/{:s}-fine-degree2.h5".format(ts_name)

    if not os.path.exists(first_hybrid_fn):
        if True:
            if not os.path.exists(ssa_fn):
                raise FileNotFoundError("Need an SSA result at 30kyr ({:s}) to work off".format(ssa_fn))
            else:
                print("Loading SSA at {:s}".format(ssa_fn))
                with firedrake.CheckpointFile(ssa_fn, "r") as chk:
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

                target_time = 26000.0

                Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
                V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)

                z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))
                print("Interpolating thickness")
                h_0 = lift3d(fields_ssa["thickness"], Q2)
                time_so_far = 25000.0
                dt = 0.5

                x, y, z = firedrake.SpatialCoordinate(fine_mesh)

                A = Constant(20)
                C = get_C(fine_mesh, Q2)
                a = get_a(fine_mesh, Q2)

                s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
                x = firedrake.SpatialCoordinate(fine_mesh)[0]
                u_init = firedrake.assemble(interpolate(firedrake.as_vector((90 * x / Lx, 0)), V2))
                solver = icepack.solvers.FlowSolver(model, **fast_opts)

                print("Getting initial velocity")
                u_0 = solver.diagnostic_solve(
                    velocity=u_init,
                    thickness=h_0,
                    surface=s_0,
                    fluidity=A,
                    friction=C,
                )

                fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

                print(f"Initializing to {target_time} kyr with a {(target_time - time_so_far) / 1000} kyr run")
                exception, fields_2 = run_simulation(solver, target_time - time_so_far, dt, bed=z_b, a=a, C=C, A=A, **fields)

                with firedrake.CheckpointFile("outputs/{:s}-fine-degree2.h5".format(ts_name), "w") as chk:
                    chk.save_mesh(fine_mesh)
                    for key in fields_2:
                        chk.save_function(fields_2[key], name=key)
        else:
            fields_init = {}
            with firedrake.CheckpointFile("outputs/mismip-fine-degree2_comp.h5", "r") as chk:
                fine_mesh = chk.load_mesh("fine_mesh")
                time_now = chk.get_attr("metadata", "total_time")
                for key in field_names:
                    fields_init[key] = chk.load_function(fine_mesh, key)
            Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
            V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)
            z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))
            h_0 = fields_init["thickness"]
            target_time = 2500.0
            time_so_far = 0.0
            dt = 0.5

            x, y, z = firedrake.SpatialCoordinate(fine_mesh)

            A = Constant(20)
            C = get_C(fine_mesh, Q2)
            a = get_a(fine_mesh, Q2)

            s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
            x = firedrake.SpatialCoordinate(fine_mesh)[0]
            solver = icepack.solvers.FlowSolver(model, **fast_opts)

            print("Getting initial velocity")
            u_0 = solver.diagnostic_solve(
                velocity=fields_init["velocity"],
                thickness=h_0,
                surface=s_0,
                fluidity=A,
                friction=C,
            )

            fields = {"surface": s_0, "thickness": h_0, "velocity": u_0}

            print(f"Initializing to {target_time} kyr with a {(target_time - time_so_far) / 1000} kyr run")
            exception, fields_2 = run_simulation(solver, target_time - time_so_far, dt, bed=z_b, a=a, C=C, A=A, **fields)

            with firedrake.CheckpointFile("outputs/{:s}-fine-degree2.h5".format(ts_name), "w") as chk:
                chk.save_mesh(fine_mesh)
                for key in fields_2:
                    chk.save_function(fields_2[key], name=key)

        fields_init = {}
        with firedrake.CheckpointFile(first_hybrid_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh")
            for key in field_names:
                fields_init[key] = chk.load_function(fine_mesh, key)

        fig, axes = subplots(figsize=(14, 6))
        colors = tripcolor(extract_surface(fields_init["thickness"]), axes=axes)
        colorbar(fig, colors, label="H [m]")
        fig.savefig("plots/hybrid/{:s}_thickness_init.png".format(ts_name), dpi=300)

    output_fn = "outputs/{:s}-fine-degree2_comp.h5".format(ts_name)

    fields_init = {}
    if not os.path.exists(output_fn):
        print("Reloading init results...")
        # time_now = 30000
        time_now = 2500
        with firedrake.CheckpointFile(first_hybrid_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh")
            for key in field_names:
                fields_init[key] = chk.load_function(fine_mesh, key)
    else:
        print("Reloading looped results...")
        with firedrake.CheckpointFile(output_fn, "r") as chk:
            fine_mesh = chk.load_mesh("fine_mesh")
            time_now = chk.get_attr("metadata", "total_time")
            for key in field_names:
                fields_init[key] = chk.load_function(fine_mesh, key)
    print("Starting at {:d}".format(int(time_now)))


    solver = icepack.solvers.FlowSolver(model, **fast_opts)
    Q2 = firedrake.FunctionSpace(fine_mesh, "CG", 2, vfamily="R", vdegree=0)
    V2 = firedrake.VectorFunctionSpace(fine_mesh, "CG", 2, dim=2, vfamily="GL", vdegree=2)
    x = firedrake.SpatialCoordinate(fine_mesh)[0]
    # u_init = firedrake.assemble(interpolate(firedrake.as_vector((90 * x / Lx, 0)), V2))
    u_init = firedrake.assemble(0.9 * fields_init["velocity"])

    A = Constant(20)
    C = get_C(fine_mesh, Q2)
    a = get_a(fine_mesh, Q2)
    a2 = firedrake.assemble(interpolate(firedrake.Constant(0.3), Q2))
    print("Difference in compared to constant is {:f} m^3/yr".format(firedrake.assemble((a - a2) * firedrake.ds_t)))

    cm = tripcolor(extract_surface(a))
    plt.colorbar(cm)
    # plt.show()

    solver1 = icepack.solvers.FlowSolver(model, **fast_opts)
    print("Re-initializing velocity")
    u_0 = solver1.diagnostic_solve(
        velocity=u_init,
        thickness=fields_init["thickness"],
        surface=fields_init["surface"],
        fluidity=A,
        friction=C,
    )

    fields_complete = {
        "surface": fields_init["surface"],
        "thickness": fields_init["thickness"],
        "velocity": u_0,
    }
    z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))


    final_time = time_now + 10000
    increment = 500
    dt = 0.625
    for i in range(int((final_time - time_now) / increment)):
        print("Running and saving years {:d} to {:d}".format(int(time_now + i * increment), int(time_now + (i + 1) * increment)))
        exception, fields_complete = run_simulation(solver1, increment, dt, bed=z_b, a=a, C=C, A=A, plot=False, **fields_complete)
        fields_complete["tvm"] = firedrake.project(von_mises_stress_vel(velocity=fields_complete["velocity"], fluidity=A) * 1000, Q2)
        if exception:
            with firedrake.CheckpointFile(output_fn[:-3] + "_err.h5", "w") as chk:
                chk.create_group("metadata")
                chk.set_attr("metadata", "total_time", increment * (i + 1) + time_now)
                chk.save_mesh(fine_mesh)
                for key in fields_complete:
                    chk.save_function(fields_complete[key], name=key)
            raise exception

        with firedrake.CheckpointFile(output_fn, "w") as chk:
            chk.create_group("metadata")
            chk.set_attr("metadata", "total_time", increment * (i + 1) + time_now)
            chk.save_mesh(fine_mesh)
            for key in fields_complete:
                chk.save_function(fields_complete[key], name=key)
