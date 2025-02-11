#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

"""

"""
import tqdm
import numpy as np
import firedrake
import icepack
from firedrake.__future__ import interpolate
from firedrake.pyplot import tripcolor, tricontour
from firedrake import dx, dS_v, dS
from icepack.constants import ice_density as ρ_I, water_density as ρ_W
import matplotlib.pyplot as plt
from icepackaccs import extract_surface
from icepackaccs.mismip import Lx, Ly


ts_name = "partial_stream"


fast_opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtonls",
        "snes_max_it": 1000,
        "snes_stol": 1.0e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "snes_linesearch_type": "bt",
        "snes_linesearch_order": 2,
        "pc_factor_mat_solver_type": "mumps",
        "max_iterations": 2500,
    },
}

reliable_opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "icepack",
    "diagnostic_solver_parameters": {
        "snes_type": "newtontr",
        "snes_max_it": 5000,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "max_iterations": 5000,
    },
}

other_opts = {
    "dirichlet_ids": [4],
    "side_wall_ids": [1, 3],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtonls",
        "snes_linesearch_max_it": 2500,
        "snes_max_it": 5000,
        "snes_stol": 1.0e-8,
        "snes_rtol": 1.0e-6,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "max_iterations": 5000,
        "snes_linesearch_damping": 0.1,
        "snes_monitor": None,
        "snes_linesearch_monitor": None,
    },
}

damping_and_iters = [(0.5, 50), (0.1, 1000), (0.05, 2000)]
# damping_and_iters = [(0.5, 50)]  # , (0.1, 1000)]


def trapezoidal_channel(mesh, depth, x0, y_c, y_width=2000.0, ramp_y=1000.0, ramp_x=5000.0, outer=None, kink_y_outer=None):
    x = firedrake.SpatialCoordinate(mesh)[0]
    y = firedrake.SpatialCoordinate(mesh)[1]
    y_slope = depth / ramp_y
    bottom_slope = firedrake.conditional(y < y_c, (y - (y_c - y_width / 2.0 - ramp_y)) * y_slope, 0.0)
    top_slope = firedrake.conditional(y >= y_c, -(y - (y_c + y_width / 2.0 + ramp_y)) * y_slope, 0.0)
    left_slope = firedrake.min_value(firedrake.conditional(x > (x0 - ramp_x), (x - (x0 - ramp_x)) / ramp_x, 0.0), 1.0)
    if outer is None:
        side_slope = left_slope
    else:
        side_slope = left_slope * firedrake.max_value(0.0, firedrake.min_value(1.0, 1.0 + (outer - x) / ramp_x))
    return firedrake.max_value(firedrake.min_value((bottom_slope + top_slope) * side_slope, depth), 0.0)


def sidehugging_trapezoidal_channels(mesh, depth, x0, outer_start, kink_x=5.2e5, kink_y_outer=1.35e4, y_width=2000.0, ramp_y=1000.0, ramp_x=5000.0, outer=None):
    return sidehugging_trapezoidal_channel(mesh, depth, x0, outer_start, kink_x=kink_x, kink_y_outer=kink_y_outer, y_width=y_width, ramp_y=ramp_y, ramp_x=ramp_x, outer=outer) + sidehugging_trapezoidal_channel(mesh, depth, x0, Ly - outer_start - ramp_y * 2 - y_width, kink_x=kink_x, kink_y_outer=Ly - kink_y_outer - ramp_y * 2 - y_width, y_width=y_width, ramp_y=ramp_y, ramp_x=ramp_x, outer=outer)


def sidehugging_trapezoidal_channel(mesh, depth, x0, outer_start, kink_x=5.2e5, kink_y_outer=1.35e4, y_width=2000.0, ramp_y=1000.0, ramp_x=5000.0, outer=None):
    x = firedrake.SpatialCoordinate(mesh)[0]
    y = firedrake.SpatialCoordinate(mesh)[1]
    y_slope = depth / ramp_y

    # start past the kink
    bottom_slope = firedrake.conditional(y < kink_y_outer + y_width / 2.0 + ramp_y, (y - kink_y_outer) * y_slope, 0.0)
    top_slope = firedrake.conditional(y >= kink_y_outer + y_width / 2.0 + ramp_y, -(y - (kink_y_outer + y_width + 2.0 * ramp_y)) * y_slope, 0.0)
    outer_channel_unclipped = firedrake.max_value(firedrake.min_value((bottom_slope + top_slope), depth), 0.0)

    # now the tough part
    if abs(kink_x - x0) > 1.0e-16:
        outboard_line = outer_start + (x - x0) * (kink_y_outer - outer_start) / (kink_x - x0)
    else:
        outboard_line = outer_start
    inboard_line = outboard_line + y_width + 2.0 * ramp_y
    center_line = outboard_line + y_width / 2.0 + ramp_y
    bottom_slope_inboard = firedrake.conditional(y < center_line, (y - outboard_line) * y_slope, 0.0)
    top_slope_inboard = firedrake.conditional(y >= center_line, -(y - inboard_line) * y_slope, 0.0)
    inner_channel_unclipped = firedrake.max_value(firedrake.min_value((bottom_slope_inboard + top_slope_inboard), depth), 0.0)

    comb_channel = firedrake.conditional(x > kink_x, outer_channel_unclipped, inner_channel_unclipped)

    left_slope = firedrake.min_value(firedrake.conditional(x > (x0 - ramp_x), (x - (x0 - ramp_x)) / ramp_x, 0.0), 1.0)
    if outer is None:
        x_restriction = left_slope
    else:
        x_restriction = left_slope * firedrake.max_value(0.0, firedrake.min_value(1.0, 1.0 + (outer - x) / ramp_x))
    return firedrake.max_value(comb_channel * x_restriction, 0.0)


def trapezoidal_channels(mesh, depth, x0, y_c, y_width=2000.0, ramp_y=2000.0, ramp_x=5000.0, outer=None):
    return trapezoidal_channel(
        mesh, depth, x0, y_c, y_width=y_width, ramp_y=ramp_y, ramp_x=ramp_x, outer=outer
    ) + trapezoidal_channel(
        mesh,
        depth,
        x0,
        Ly - y_c,
        y_width=y_width,
        ramp_y=ramp_y,
        ramp_x=ramp_x,
        outer=outer,
    )


def volume_conserving_trapezoidal_channels(
    mesh,
    depth,
    x0,
    y_c,
    y_width=2000.0,
    ramp_y=2000.0,
    ramp_x=5000.0,
    inboard_x=4.75e5,
    inboard_ramp_x=2.5e4,
    outboard_x=5.8e5,
    outboard_ramp_x=1.0e4,
    outer=None,
):
    x = firedrake.SpatialCoordinate(mesh)[0]
    y = firedrake.SpatialCoordinate(mesh)[1]
    tc = trapezoidal_channels(mesh, depth, x0, y_c, y_width=y_width, ramp_y=ramp_y, ramp_x=ramp_x, outer=outer)
    vol = firedrake.assemble(tc * firedrake.dx)
    outboard_left = firedrake.conditional(y > (Ly - y_c + y_width / 2 + ramp_y), 1.0, 0.0)
    outboard_right = firedrake.conditional(y < (y_c - y_width / 2 - ramp_y), 1.0, 0.0)
    outboard_ramp = firedrake.max_value(firedrake.min_value((x - outboard_x) / outboard_ramp_x, 1.0), 0.0)
    inboard = firedrake.conditional(y > (y_c + y_width / 2 + ramp_y), 1.0, 0.0) * firedrake.conditional(
        y < (Ly - y_c - y_width / 2 - ramp_y), 1.0, 0.0
    )
    inboard_ramp = firedrake.max_value(firedrake.min_value((x - inboard_x) / inboard_ramp_x, 1.0), 0.0)
    not_a_channel = inboard * inboard_ramp + (outboard_right + outboard_left) * outboard_ramp
    non_channel_area = firedrake.assemble(not_a_channel * firedrake.dx)
    thickening = not_a_channel * vol / non_channel_area
    return tc - thickening


def volume_conserving_trapezoidal_channel(
    mesh,
    depth,
    x0,
    y_c,
    y_width=2000.0,
    ramp_y=2000.0,
    ramp_x=5000.0,
    inboard_x=4.75e5,
    inboard_ramp_x=2.5e4,
    outboard_x=5.8e5,
    outboard_ramp_x=1.0e4,
    outin_transition=22.5e3,
    outer=None,
    ramp_y_volcon=3000.0,
):
    if y_c > 40.0e3:
        raise ValueError("We can only do a channel in the bottom half of the domain (including the middle)")
    if outin_transition > 40.0e3:
        raise ValueError("outin_transition must be <= than 40.0e3")

    x = firedrake.SpatialCoordinate(mesh)[0]
    y = firedrake.SpatialCoordinate(mesh)[1]
    tc = trapezoidal_channel(mesh, depth, x0, y_c, y_width=y_width, ramp_y=ramp_y, ramp_x=ramp_x, outer=outer)
    vol = firedrake.assemble(tc * firedrake.dx)

    outboard_left = firedrake.conditional(y > (Ly - outin_transition), 1.0, 0.0)
    outboard_right = firedrake.conditional(y < (min(y_c - y_width / 2 - ramp_y, outin_transition)), 1.0, 0.0)

    inboard_ramp = firedrake.max_value(firedrake.min_value((x - inboard_x) / inboard_ramp_x, 1.0), 0.0)
    outboard_ramp = firedrake.max_value(firedrake.min_value((x - outboard_x) / outboard_ramp_x, 1.0), 0.0)

    inboard_left = firedrake.conditional(y > max(y_c + y_width / 2 + ramp_y, outin_transition), 1.0, 0.0) * firedrake.conditional(
        y < (Ly - outin_transition), 1.0, 0.0
    ) + firedrake.max_value(
        0.0,
        firedrake.min_value(1.0, 1.0 - (y - (Ly - outin_transition)) / ramp_y_volcon),
    ) * (
        1.0 - outboard_ramp
    ) * firedrake.conditional(
        y >= (Ly - outin_transition), 1.0, 0.0
    )

    inboard_right = firedrake.conditional(y < y_c - y_width / 2 - ramp_y, 1.0, 0.0) * firedrake.conditional(
        y > outin_transition, 1.0, 0.0
    ) + firedrake.max_value(
        0.0,
        firedrake.min_value(1.0, (y - (outin_transition - ramp_y_volcon)) / ramp_y_volcon),
    ) * (
        1.0 - outboard_ramp
    ) * firedrake.conditional(
        y <= outin_transition, 1.0, 0.0
    )

    not_a_channel = (inboard_left + inboard_right) * inboard_ramp + (outboard_right + outboard_left) * outboard_ramp
    non_channel_area = firedrake.assemble(not_a_channel * firedrake.dx)
    thickening = not_a_channel * vol / non_channel_area
    return tc - thickening


def equivalent_thinning(
    func,
    surf,
    thick,
    mesh,
    depth,
    x0,
    y_c,
    **kwargs
):
    vol_chan = vol_thinning(func, mesh, depth, x0, y_c, **kwargs)
    floating = gradual_floating_area(mesh, surf, thick)
    vol_float = firedrake.assemble(floating * dx)
    scale = vol_chan / vol_float
    return floating * scale


def vol_thinning(
    func,
    mesh,
    depth,
    x0,
    y_c,
    **kwargs
):
    tc = func(mesh, depth, x0, y_c, **kwargs)
    return firedrake.assemble(tc * dx)


def gradual_floating_area(mesh, s, h0):
    """Find the floating area, easing in gradually

    Parameters
    ----------
    mesh: firedrake.Mesh
        Needed since we construct function space in addition to that on which s and h0 live
    s: firedrake.Function
        The ice surface elevation.
    h0: firedrake.Function
        The ice thickness.

    Returns
    -------
    floating_mask: firedrake.Function
        A gradual mask of the floating area living in the same function space as s.
    """
    height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h0, s.function_space()))
    hob1 = firedrake.assemble(interpolate(height_above_flotation > 0.01, s.function_space()))

    if mesh.geometric_dimension() == 3:
        DG0 = firedrake.FunctionSpace(mesh, "DG", 0, vdegree=0)
        use_dS = dS_v
    else:
        DG0 = firedrake.FunctionSpace(mesh, "DG", 0)
        use_dS = dS

    ϵ = firedrake.Function(DG0)
    J = 0.5 * ((ϵ - hob1) ** 2 * dx + (5e3) * (ϵ("+") - ϵ("-")) ** 2 * use_dS)
    F = firedrake.derivative(J, ϵ)
    firedrake.solve(F == 0, ϵ)
    definitely_ungrounded = firedrake.assemble(interpolate(ϵ < 0.02, DG0))
    ϵ2 = firedrake.Function(DG0)
    J2 = 0.5 * ((ϵ2 - definitely_ungrounded) ** 2 * dx + (5e4) * (ϵ2("+") - ϵ2("-")) ** 2 * use_dS)
    F2 = firedrake.derivative(J2, ϵ2)
    firedrake.solve(F2 == 0, ϵ2)
    return firedrake.assemble(interpolate(firedrake.max_value(0.0, (ϵ2 - 0.5) * 2.0), s.function_space()))


def smooth_floating(res, s, h0, cutoff=0.05):
    """Find the floating area

    Parameters
    ----------
    mesh: firedrake.Mesh
        Needed since we construct function space in addition to that on which s and h0 live
    s: firedrake.Function
        The ice surface elevation.
    h0: firedrake.Function
        The ice thickness.

    Returns
    -------
    floating_mask: firedrake.Function
        A gradual mask of the floating area living in the same function space as s.
    """
    mesh2 = firedrake.RectangleMesh(int(Lx / res), int(Ly / res), Lx, Ly)
    Q2 = firedrake.FunctionSpace(mesh2, "CG", degree=1)
    height_above_flotation = firedrake.assemble(interpolate(s - (1 - ρ_I / ρ_W) * h0, Q2))
    hob1 = firedrake.assemble(interpolate((firedrake.min_value(height_above_flotation, 10) - 5.0) / 5.0, Q2))
    DG0 = firedrake.FunctionSpace(mesh2, "DG", 0)
    ϵ = firedrake.Function(DG0)
    J = 0.5 * ((ϵ - hob1) ** 2 * dx + (1e3) * (ϵ("+") - ϵ("-")) ** 2 * dS)
    F = firedrake.derivative(J, ϵ)
    firedrake.solve(F == 0, ϵ)
    return firedrake.project(ϵ, Q2)


def volume(thickness):
    return firedrake.assemble(thickness * firedrake.dx) / 1.0e9


def run_simulation(solver, time, dt, return_all=False, recomp_u=False, plot=False, **fields):
    if plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        axes[1][1].axis("off")
        cbrs = None
    h, s, u, z_b, a, A, C = map(fields.get, ("thickness", "surface", "velocity", "bed", "a", "A", "C"))
    h_0 = h.copy(deepcopy=True)
    num_steps = int(time / dt)
    H_v_t = np.zeros((num_steps,))
    if return_all:
        thicks = []
        vels = []
        surfs = []
        dH = []

    if recomp_u:
        print("Recomputing velocity because of divergence errors")
        Q = h.function_space()
        mesh = Q.mesh()
        x, _ = firedrake.SpatialCoordinate(mesh)
        # u_init = firedrake.assemble(interpolate(firedrake.as_vector((x / Lx, 0)), u.function_space()))
        u_init = firedrake.assemble(0.5 * u)
        u = solver.diagnostic_solve(
            velocity=u_init,
            thickness=h,
            surface=s,
            fluidity=A,
            friction=C,
        )

    try:
        progress_bar = tqdm.trange(num_steps)
        for step in progress_bar:
            h_prev = h.copy(deepcopy=True)
            h = solver.prognostic_solve(
                dt,
                thickness=h,
                velocity=u,
                accumulation=a,
                thickness_inflow=h_0,
            )
            firedrake.assemble(h.interpolate(firedrake.max_value(h, 10.0)))
            s = icepack.compute_surface(thickness=h, bed=z_b)

            u = solver.diagnostic_solve(
                velocity=u,
                thickness=h,
                surface=s,
                fluidity=A,
                friction=C,
            )

            dh = firedrake.assemble(interpolate(h - h_prev, h.function_space()))
            H_v_t[step] = volume(h)
            if return_all:
                thicks.append(h.copy(deepcopy=True))
                vels.append(u.copy(deepcopy=True))
                surfs.append(s.copy(deepcopy=True))
                dH.append(dh.copy(deepcopy=True))
            description = f"dV,max(abs(dH)): {(H_v_t[step] - H_v_t[step - 1]) / dt:4.2f} [km3/yr] {(np.abs(dh.dat.data_ro).max()) / dt:4.3f} [m/yr]"
            progress_bar.set_description(description)
            if plot:
                axes[0][0].clear()
                axes[0][1].clear()
                axes[1][0].clear()
                colorsv = tripcolor(
                    extract_surface(firedrake.assemble(interpolate(u[0], h.function_space()))),
                    axes=axes[0][0],
                    cmap="Reds",
                    vmin=0,
                    vmax=500,
                )
                colorss = tripcolor(extract_surface(s), axes=axes[0][1], vmin=0, vmax=500)
                colorsdh = tripcolor(
                    extract_surface(firedrake.assemble(interpolate(dh / dt, dh.function_space()))),
                    axes=axes[1][0],
                    vmin=-0.25,
                    vmax=0.25,
                    cmap="bwr",
                )
                if cbrs is None:
                    cbrs = [
                        plt.colorbar(
                            colorsv,
                            ax=axes[1][1],
                            label=r"u$_x$ (m yr$^{-1}$",
                            extend="max",
                            location="left",
                            pad=0.1,
                        ),
                        plt.colorbar(
                            colorss,
                            ax=axes[1][1],
                            label=r"s (m)",
                            extend="max",
                            pad=0.4,
                        ),
                        plt.colorbar(
                            colorsdh,
                            ax=axes[1][1],
                            label=r"$\Delta$H (m yr$^{-1}$",
                            extend="both",
                            pad=0.0,
                        ),
                    ]
                fig.savefig("progress.png", dpi=300)
        if return_all:
            return None, {
                "thickness": thicks,
                "surface": surfs,
                "velocity": vels,
                "dH": dH,
            }
        else:
            return None, {"thickness": h, "surface": s, "velocity": u, "dH": dh}
    except firedrake.exceptions.ConvergenceError as e:
        if return_all:
            return e, {
                "thickness": thicks,
                "surface": surfs,
                "velocity": vels,
                "dH": dH,
            }
        else:
            return e, {"thickness": h, "surface": s, "velocity": u, "dH": dh}


def try_alot(model, u_prev, h_0, s_0, A, C):
    converged = False
    try:
        solver = icepack.solvers.FlowSolver(model, **fast_opts)
        u_0 = solver.diagnostic_solve(velocity=u_prev, thickness=h_0, surface=s_0, fluidity=A, friction=C)
        converged = True
    except firedrake.exceptions.ConvergenceError:
        for damping, iters in damping_and_iters:
            try:
                other_opts["diagnostic_solver_parameters"]["snes_max_it"] = iters
                other_opts["diagnostic_solver_parameters"]["snes_linesearch_damping"] = damping
                print("Retrying with {:1.2f} damping".format(damping))
                solver = icepack.solvers.FlowSolver(model, **other_opts)
                u_0 = solver.diagnostic_solve(
                    velocity=u_prev,
                    thickness=h_0,
                    surface=s_0,
                    fluidity=A,
                    friction=C,
                )
                converged = True
                break
            except firedrake.exceptions.ConvergenceError:
                continue
    if not converged:
        print("Retrying fancily")
        opts = other_opts.copy()
        opts["diagnostic_solver_parameters"]["snes_max_it"] = 2000
        opts["diagnostic_solver_parameters"]["snes_linesearch_damping"] = 0.05
        opts["diagnostic_solver_parameters"]["snes_stol"] = 1.0e-6
        opts["diagnostic_solver_parameters"]["snes_rtol"] = 1.5e-4
        solver = icepack.solvers.FlowSolver(model, **opts)
        u_0 = solver.diagnostic_solve(
            velocity=u_prev,
            thickness=h_0,
            surface=s_0,
            fluidity=A,
            friction=C,
        )
        opts["diagnostic_solver_parameters"]["snes_stol"] = 1.0e-8
        opts["diagnostic_solver_parameters"]["snes_rtol"] = 1.0e-6
        if False:
            opts["diagnostic_solver_parameters"]["snes_stol"] = 1.0e-8
            opts["diagnostic_solver_parameters"]["snes_rtol"] = 1.0e-1
            opts["diagnostic_solver_parameters"]["snes_max_it"] = 10000
            opts["diagnostic_solver_parameters"]["snes_linesearch_damping"] = 0.001
            solver = icepack.solvers.FlowSolver(model, **opts)
            u_0 = solver.diagnostic_solve(
                velocity=u_0,
                thickness=h_0,
                surface=s_0,
                fluidity=A,
                friction=C,
            )
    return {"surface": s_0, "thickness": h_0, "velocity": u_0}


if __name__ == "__main__":
    res = 500
    mesh = firedrake.RectangleMesh(int(Lx / res), int(Ly / res), Lx, Ly)
    Q2 = firedrake.FunctionSpace(mesh, "CG", 2)
    x0 = 4.46e5
    y0 = 1.95e4
    x_kink = 5.2e5
    y_kink = 1.35e4
    fig, ax = plt.subplots()
    center_chan = trapezoidal_channel(mesh, 100, 4.45e5, 4.0e4)
    tricontour(firedrake.assemble(interpolate(center_chan, Q2)), levels=[10], axes=ax, colors='k')
    outboard_chan = sidehugging_trapezoidal_channel(mesh, 100, x0, y0, x_kink, y_kink)
    tricontour(firedrake.assemble(interpolate(outboard_chan, Q2)), levels=[10], axes=ax, colors='C1')
    outer_chan = sidehugging_trapezoidal_channel(mesh, 100, x0, y0, x_kink, y_kink, outer=5.3e5)
    tricontour(firedrake.assemble(interpolate(outer_chan, Q2)), levels=[10], axes=ax, colors="C2")

    outer_chans = sidehugging_trapezoidal_channels(mesh, 50, 520000, 1.35e4)
    tricontour(firedrake.assemble(interpolate(outer_chans, Q2)), levels=[10], axes=ax, colors='C3')

    cm = tripcolor(firedrake.assemble(interpolate(outer_chans, Q2)))
    plt.colorbar(cm)

    field_names = ["velocity", "thickness", "surface", "dH"]

    fields_mismip = {}
    with firedrake.CheckpointFile("../modeling/outputs/{:s}-fine-degree2_comp.h5".format("mismip"), "r") as chk:
        fine_mesh_mismip = chk.load_mesh(name="fine_mesh")
        for key in field_names:
            fields_mismip[key] = chk.load_function(fine_mesh_mismip, key)
        fields_mismip["stress"] = chk.load_function(fine_mesh_mismip, "tvm")
    height_above_flotation_mismip = extract_surface(
        firedrake.assemble(
            interpolate(
                fields_mismip["surface"] - (1 - ρ_I / ρ_W) * fields_mismip["thickness"],
                fields_mismip["thickness"].function_space(),
            )
        )
    )
    is_floating_mismip = smooth_floating(250, extract_surface(fields_mismip["surface"]), extract_surface(fields_mismip["thickness"]))

    fig, ax = plt.subplots()
    tricontour(is_floating_mismip, levels=[0.0], colors="k", axes=ax)
    outboard_chan = sidehugging_trapezoidal_channels(mesh, 100, x0, y0, x_kink, y_kink)
    tricontour(firedrake.assemble(interpolate(outboard_chan, Q2)), levels=[10], axes=ax, colors='C1')

    plt.show()
