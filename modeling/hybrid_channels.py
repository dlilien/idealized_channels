#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 David Lilien <dlilien@iu.edu>
#
# Distributed under terms of the GNU GPL3.0 license.

# Basic packaages
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Modeling packages
import firedrake
from firedrake.__future__ import interpolate
from firedrake import Constant
from firedrake.pyplot import tripcolor, tricontour
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
)
import icepack
import icepack.plot
from icepack.models.viscosity import membrane_stress, sym_grad

# my packages
from icepackaccs.stress import von_mises_stress_vel
from icepackaccs.mismip import mismip_bed_topography, Lx, Ly
from icepackaccs.friction import regularized_coulomb_mismip
from icepackaccs import extract_surface

# Local code
from libchannels import trapezoidal_channel, sidehugging_trapezoidal_channels, sidehugging_trapezoidal_channel, equivalent_thinning, fast_opts, try_alot, vol_thinning, ts_name

ramp_y = 1000
min_thick = 10.0
widths = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
depths = [50, 100, 150, 200, 250]
levels = [5]
chan_fn_template = "outputs/{:s}-hybrid-{:s}-{:s}.h5"

margin_y = 1.95e4
standard_x = 4.46e5
outer_x = 5.2e5
inner_end_x = 5.2e5
central_y = 4.0e4


def twostream_C(mesh, Q, y_w=2.5e3, y0=1.0e-2, y1=1.0e-1):
    y = firedrake.SpatialCoordinate(mesh)[1]
    return firedrake.project(
        firedrake.max_value(firedrake.min_value(y0 + (y - Ly / 2 + y_w) / (2 * y_w) * (y1 - y0), y1), y0),
        Q,
    )


def run_sims(channel_func, model, mesh, left, vert, right, Q2, z_b, widths, depths, h_in, s_in, u_in, A, C, chan_fn, funky=False):
    results_chans = {}
    results_even = {}
    for i, channel_width in enumerate(widths):
        for j, channel_depth in enumerate(depths):
            print(f"Running equivalent thinning for a {channel_width / 1000}-km wide, {channel_depth}-m deep channel...")
            key = (channel_width, channel_depth)
            kink_y_outer = 1.35e4
            if funky:
                vert = Ly - margin_y - ramp_y * 2 - channel_width
                kink_y_outer = Ly - kink_y_outer - ramp_y * 2 - channel_width
            chans = equivalent_thinning(
                channel_func,
                s_in,
                h_in,
                mesh,
                channel_depth,
                left,
                vert,
                outer=right,
                ramp_y=ramp_y,
                kink_y_outer=kink_y_outer,
                y_width=channel_width,
            )
            h_even = firedrake.project(firedrake.max_value(h_in - chans, min_thick), Q2)
            s_even = icepack.compute_surface(thickness=h_even, bed=z_b)
            if j > 0:
                keyt = (channel_width, depths[j - 1])
                u_prev = results_even[keyt]["velocity"]
            elif i > 0:
                keyt = (widths[i - 1], channel_depth)
                u_prev = results_even[keyt]["velocity"]
            else:
                u_prev = u_in
            results_even[key] = try_alot(model, u_prev, h_even, s_even, A, C)
            # print("Calculating principal stresses")
            # princ_stresses = principal_stress(velocity=results_even[key]["velocity"], fluidity=A)
            # print("Calculating von Mises stresses")
            # results_even[key]["tvm"] = firedrake.project(von_mises_stress(princ_stresses) * 1000, Q2)
            results_even[key]["tvm"] = firedrake.project(von_mises_stress_vel(velocity=results_even[key]["velocity"], fluidity=A) * 1000, Q2)
            print(f"Running a {channel_width / 1000}-km wide, {channel_depth}-m deep channel...")
            chans = channel_func(
                mesh,
                channel_depth,
                left,
                vert,
                outer=right,
                ramp_y=ramp_y,
                y_width=channel_width,
                kink_y_outer=kink_y_outer,
            )
            h_chan = firedrake.project(firedrake.max_value(h_in - chans, min_thick), Q2)
            s_chan = icepack.compute_surface(thickness=h_chan, bed=z_b)

            if j > 0:
                keyt = (channel_width, depths[j - 1])
                u_prev = results_chans[keyt]["velocity"]
            elif i > 0:
                keyt = (widths[i - 1], channel_depth)
                u_prev = results_chans[keyt]["velocity"]
            else:
                u_prev = u_in
            results_chans[key] = try_alot(model, u_prev, h_chan, s_chan, A, C)
            # print("Calculating principal stresses")
            # princ_stresses = principal_stress(velocity=results_chans[key]["velocity"], fluidity=A)
            # print("Calculating von Mises stresses")
            # results_chans[key]["tvm"] = firedrake.project(von_mises_stress(princ_stresses) * 1000, Q2)
            results_chans[key]["tvm"] = firedrake.project(von_mises_stress_vel(velocity=results_chans[key]["velocity"], fluidity=A) * 1000, Q2)

    with firedrake.CheckpointFile(chan_fn, "w") as chk:
        chk.save_mesh(mesh)
        for channel_width in widths:
            for channel_depth in depths:
                key = (channel_width, channel_depth)
                chk.create_group("{:d} {:d}".format(channel_width, channel_depth))
                chk.set_attr(
                    "{:d} {:d}".format(channel_width, channel_depth),
                    "volume",
                    vol_thinning(channel_func, mesh, channel_depth, left, vert, outer=right, ramp_y=ramp_y, y_width=channel_width),
                )
                for f in ["thickness", "velocity", "surface", "tvm"]:
                    if results_even[key] is not None:
                        chk.save_function(
                            results_even[key][f],
                            name="{:d} {:d} {:s} even".format(channel_width, channel_depth, f),
                        )
                    if results_chans[key] is not None:
                        chk.save_function(
                            results_chans[key][f],
                            name="{:d} {:d} {:s} chan".format(channel_width, channel_depth, f),
                        )
    return results_even, results_chans


def subplots(**kwargs):
    fig, axes = icepack.plot.subplots(**kwargs)
    axes.set_aspect(2)
    axes.set_xlim((0, Lx))
    axes.set_ylim((0, Ly))
    return fig, axes


def colorbar(fig, colors, **kwargs):
    return fig.colorbar(colors, fraction=0.012, pad=0.025, **kwargs)


def channels_and_plots(setup="mismip", pos_name="channels", chan_name="full", restart=False):
    print("Running:", setup, chan_name, pos_name)
    vert = margin_y
    left = standard_x
    right = None
    channel_func = sidehugging_trapezoidal_channels
    funky = False
    if chan_name == "outer":
        left = outer_x
        vert = 1.35e4
    if chan_name == "inner":
        right = inner_end_x
    if pos_name == "center":
        channel_func = trapezoidal_channel
        vert = central_y
    if pos_name == "margin":
        channel_func = sidehugging_trapezoidal_channel
    if pos_name == "funky":
        channel_func = sidehugging_trapezoidal_channel
        funky = True

    chan_fn = chan_fn_template.format(setup, pos_name, chan_name)
    plot_pref = "_".join([setup, pos_name, chan_name])
    A = Constant(20)

    model = icepack.models.HybridModel(friction=regularized_coulomb_mismip)
    field_names = ["velocity", "thickness", "surface"]

    fields_3 = {}
    with firedrake.CheckpointFile("outputs/{:s}-fine-degree2_comp.h5".format(setup), "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for key in field_names:
            fields_3[key] = chk.load_function(fine_mesh, key)

    Q2 = fields_3["thickness"].function_space()
    V2 = fields_3["velocity"].function_space()

    if setup == ts_name:
        C = twostream_C(fine_mesh, Q2)
    else:
        C = Constant(1e-2)

    h_0 = firedrake.project(fields_3["thickness"], Q2)
    u_0 = firedrake.project(fields_3["velocity"], V2)

    z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))
    s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)

    solver = icepack.solvers.FlowSolver(model, **fast_opts)
    u_0c = solver.diagnostic_solve(velocity=u_0, thickness=h_0, surface=s_0, fluidity=A, friction=C)
    u_0xc = firedrake.project(u_0c[0], Q2)
    u_0 = u_0c.copy(deepcopy=True)

    height_above_flotation_control = extract_surface(firedrake.assemble(interpolate(s_0 - (1 - ρ_I / ρ_W) * h_0, Q2)))
    if restart:
        thinning = equivalent_thinning(channel_func, s_0, h_0, fine_mesh, 100.0, left, vert, ramp_y=ramp_y, y_width=2000)
        h_equiv = firedrake.project(fields_3["thickness"] - thinning, Q2)
        s_equiv = icepack.compute_surface(thickness=h_equiv, bed=z_b)
        solver = icepack.solvers.FlowSolver(model, **fast_opts)
        u_equiv = solver.diagnostic_solve(velocity=u_0c, thickness=h_equiv, surface=s_equiv, fluidity=A, friction=C)

        # Need this to bootstrap a solution (same width channel but shallower)
        chanst = channel_func(fine_mesh, 50.0, left, vert, ramp_y=ramp_y, y_width=2000)
        h_chant = firedrake.project(fields_3["thickness"] - chanst, Q2)
        s_chant = icepack.compute_surface(thickness=h_chant, bed=z_b)

        chans = channel_func(fine_mesh, 100.0, left, vert, ramp_y=ramp_y, y_width=2000)
        h_chan = firedrake.project(fields_3["thickness"] - chans, Q2)
        s_chan = icepack.compute_surface(thickness=h_chan, bed=z_b)

        solver = icepack.solvers.FlowSolver(model, **fast_opts)
        u_chan = solver.diagnostic_solve(velocity=u_0c, thickness=h_chant, surface=s_chant, fluidity=A, friction=C)
        u_chan = solver.diagnostic_solve(velocity=u_chan, thickness=h_chan, surface=s_chan, fluidity=A, friction=C)

        gs = gridspec.GridSpec(3, 7, width_ratios=(1, 1, 1, 1, 1, 1, 0.05), height_ratios=(1, 1, 1.5))
        fig = plt.figure(figsize=(12, 10))
        ax0a = fig.add_subplot(gs[0, 0:2])
        ax0b = fig.add_subplot(gs[0, 2:4])
        ax0c = fig.add_subplot(gs[0, 4:6])

        ax1a = fig.add_subplot(gs[1, 0:2])
        ax1b = fig.add_subplot(gs[1, 2:4])
        ax1c = fig.add_subplot(gs[1, 4:6])

        ax2a = fig.add_subplot(gs[2, 0:3])
        ax2b = fig.add_subplot(gs[2, 3:6])

        cax1 = fig.add_subplot(gs[0, 6])
        cax1b = fig.add_subplot(gs[1, 6])
        cax2 = fig.add_subplot(gs[2, 6])

        height_above_flotation_chan = extract_surface(firedrake.assemble(interpolate(s_chan - (1 - ρ_I / ρ_W) * h_chan, Q2)))

        umax = 1000
        colors = tripcolor(
            extract_surface(firedrake.project(u_0[0], Q2)),
            axes=ax0a,
            cmap="Reds",
            vmin=0,
            vmax=umax,
        )
        plt.colorbar(colors, label="$u_x$ [m yr$^{-1}$]", extend="max", cax=cax1)
        tricontour(height_above_flotation_control, levels=levels, axes=ax0a, colors=["k"])
        colors = tripcolor(
            extract_surface(firedrake.project(u_equiv[0], Q2)),
            axes=ax0b,
            cmap="Reds",
            vmin=0,
            vmax=umax,
        )
        tricontour(height_above_flotation_chan, levels=levels, axes=ax0b, colors=["0.6"])
        colors = tripcolor(
            extract_surface(firedrake.project(u_chan[0], Q2)),
            axes=ax0c,
            cmap="Reds",
            vmin=0,
            vmax=umax,
        )
        tricontour(height_above_flotation_chan, levels=levels, axes=ax0c, colors=["0.6"])

        hmax = 750
        colors = tripcolor(
            extract_surface(firedrake.project(h_0, Q2)),
            axes=ax1a,
            cmap="viridis",
            vmin=0,
            vmax=hmax,
        )
        plt.colorbar(colors, label="H [m]", extend="max", cax=cax1b)
        tricontour(height_above_flotation_control, levels=levels, axes=ax1a, colors=["k"])
        colors = tripcolor(
            extract_surface(firedrake.project(h_equiv, Q2)),
            axes=ax1b,
            cmap="viridis",
            vmin=0,
            vmax=hmax,
        )
        tricontour(height_above_flotation_chan, levels=levels, axes=ax1b, colors=["0.6"])
        colors = tripcolor(
            extract_surface(firedrake.project(h_chan, Q2)),
            axes=ax1c,
            cmap="viridis",
            vmin=0,
            vmax=hmax,
        )
        tricontour(height_above_flotation_chan, levels=levels, axes=ax1c, colors=["0.6"])

        du = firedrake.project(firedrake.project(u_chan[0], Q2) - firedrake.project(u_0[0], Q2), Q2)
        dumax = 150
        colors = tripcolor(extract_surface(du), axes=ax2a, cmap="PuOr", vmin=-dumax, vmax=dumax)
        tricontour(height_above_flotation_control, levels=levels, axes=ax2a, colors=["k"])
        tricontour(height_above_flotation_chan, levels=levels, axes=ax2a, colors=["0.6"])

        du = firedrake.project(firedrake.project(u_chan[0], Q2) - firedrake.project(u_equiv[0], Q2), Q2)
        colors = tripcolor(extract_surface(du), axes=ax2b, cmap="PuOr", vmin=-dumax, vmax=dumax)
        plt.colorbar(
            colors,
            label=r"$\Delta u_x$ (w/ - w/o channels) [m yr$^{-1}$]",
            extend="both",
            cax=cax2,
        )
        tricontour(height_above_flotation_control, levels=levels, axes=ax2b, colors=["k"])
        tricontour(height_above_flotation_chan, levels=levels, axes=ax2b, colors=["0.6"])

        for ax in (ax0a, ax0b, ax0c, ax1a, ax1b, ax1c, ax2a, ax2b):
            ax.set_aspect(2)
            ax.set_xlim(40e4, 64e4)
            ax.set_ylim((0, Ly))

        for ax in (ax0b, ax0c, ax1b, ax1c, ax2b):
            ax.yaxis.set_tick_params(labelleft=False)

        ax0a.set_title("Control")
        ax0b.set_title("Even melt")
        ax0c.set_title("Channels")

        ax2a.set_title("$u_x$ (Chan - Ctrl)")
        ax2b.set_title("$u_x$ (Chan - Even)")
        fig.savefig("plots/hybrid/{:s}_init.png".format(plot_pref), dpi=300)
        plt.close(fig)

    if restart or not os.path.exists(chan_fn):
        results_even, results_chans = run_sims(channel_func, model, fine_mesh, left, vert, right, Q2, z_b, widths, depths, h_0, s_0, u_0c, A, C, chan_fn, funky=funky)
    else:
        print("Reloading channel output")
        results_even = {}
        results_chans = {}

        with firedrake.CheckpointFile(chan_fn, "r") as chk:
            fine_mesh = chk.load_mesh(name="fine_mesh")
            for channel_width in widths:
                for channel_depth in depths:
                    key = (channel_width, channel_depth)
                    results_even[key] = {}
                    results_chans[key] = {}
                    for f in ["thickness", "velocity", "surface"]:
                        results_even[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} even".format(channel_width, channel_depth, f),
                        )
                        results_chans[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} chan".format(channel_width, channel_depth, f),
                        )
        Q2 = results_even[(widths[0], depths[0])]["thickness"].function_space()
        V2 = results_even[(widths[0], depths[0])]["velocity"].function_space()

    center_pt = (630000, 40000)
    center_vels_chans = np.empty((len(widths), len(depths)))
    center_vels_even = np.empty((len(widths), len(depths)))
    outer_pt = (630000, 10000)
    outer_vels_chans = np.empty((len(widths), len(depths)))
    outer_vels_even = np.empty((len(widths), len(depths)))
    gl_pt = (445000, 40000)
    gl_vels_chans = np.empty((len(widths), len(depths)))
    gl_vels_even = np.empty((len(widths), len(depths)))

    control_center = extract_surface(u_0xc).at(*center_pt)
    control_outer = extract_surface(u_0xc).at(*outer_pt)
    control_gl = extract_surface(u_0xc).at(*gl_pt)

    fig, (ax1, ax2, ax3, cax) = plt.subplots(1, 4, gridspec_kw={"width_ratios": (1, 1, 1, 0.08)}, figsize=(7.8, 3.5))
    for i, channel_width in enumerate(widths):
        for j, channel_depth in enumerate(depths):
            center_vels_chans[i, j] = extract_surface(firedrake.project(results_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
            center_vels_even[i, j] = extract_surface(firedrake.project(results_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
            outer_vels_chans[i, j] = extract_surface(firedrake.project(results_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
            outer_vels_even[i, j] = extract_surface(firedrake.project(results_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
            gl_vels_chans[i, j] = extract_surface(firedrake.project(results_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
            gl_vels_even[i, j] = extract_surface(firedrake.project(results_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)

    fig, ax = plt.subplots()
    for i, width in enumerate(widths):
        lns = ax.plot([0] + depths, np.hstack(([control_center], center_vels_chans[i, :])))
        ax.plot(
            [0] + depths,
            np.hstack(([control_center], center_vels_even[i, :])),
            color=lns[0].get_color(),
            linestyle="dashed",
        )
    ax.set_xlabel("Channel depth (m)")
    ax.set_ylabel(r"Central flow speed (m yr$^{-1}$)")
    fig.savefig("plots/hybrid/{:s}_vel_bywidth.pdf".format(plot_pref))
    plt.close(fig)

    fig, ax = plt.subplots()
    for i, width in enumerate(widths):
        lns = ax.plot(
            [0] + depths,
            np.hstack(([0], center_vels_chans[i, :] - center_vels_even[i, :])),
            label="{:2.1f}-km wide".format(width / 1000),
        )
        ax.plot(
            [0] + depths,
            np.hstack(([0], outer_vels_chans[i, :] - outer_vels_even[i, :])),
            linestyle="dashed",
            color=lns[0].get_color(),
        )
        ax.plot(
            [0] + depths,
            np.hstack(([0], gl_vels_chans[i, :] - gl_vels_even[i, :])),
            linestyle="dotted",
            color=lns[0].get_color(),
        )
    ax.plot([], [], color="k", linestyle="solid", label="Center")
    ax.plot([], [], color="k", linestyle="dashed", label="Outer")
    ax.plot([], [], color="k", linestyle="dotted", label="GL")
    ax.set_xlabel("Channel depth (m)")
    ax.set_ylabel(r"Relative flow speed (m yr$^{-1}$)")
    ax.set_xlim(0, depths[-1])
    ax.legend(loc="upper left", frameon=False)
    fig.savefig("plots/hybrid/{:s}_vel_bywidth_paired.pdf".format(plot_pref))
    plt.close(fig)

    fig, (ax1, ax2, ax3, cax) = plt.subplots(1, 4, gridspec_kw={"width_ratios": (1, 1, 1, 0.08)}, figsize=(7.8, 3.5))
    ax1.imshow(
        np.flipud((center_vels_chans - center_vels_even).T),
        vmin=-750,
        vmax=750,
        cmap="PuOr",
        extent=(-0.25, 5.25, -25, 275),
    )
    ax1.set_title("a Shelf center")
    ax2.imshow(
        np.flipud((outer_vels_chans - outer_vels_even).T),
        vmin=-750,
        vmax=750,
        cmap="PuOr",
        extent=(-0.25, 5.25, -25, 275),
    )
    ax2.set_title("b Shelf edge")
    cm = ax3.imshow(
        np.flipud((gl_vels_chans - gl_vels_even).T),
        vmin=-750,
        vmax=750,
        cmap="PuOr",
        extent=(-0.25, 5.25, -25, 275),
    )
    plt.colorbar(cm, cax=cax, label=r"$\Delta u$ (m yr$^{-1}$)", extend="both")
    ax3.set_title("c Grounding line")
    ax1.set_ylabel("Channel depth [m]")
    ax2.set_xlabel("Channel width [km]")
    for ax in (ax1, ax2, ax3):
        ax.axis("auto")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 250)

    for ax in (ax2, ax3):
        ax.set_yticklabels([])

    fig.tight_layout(pad=0.1)
    fig.savefig("plots/hybrid/{:s}_outer_inner_gl_paired.png".format(plot_pref), dpi=300)
    plt.close(fig)

    fig, (ax1, ax2, ax3, cax) = plt.subplots(1, 4, gridspec_kw={"width_ratios": (1, 1, 1, 0.08)}, figsize=(7.8, 3.5))
    ax1.imshow(
        np.flipud(center_vels_chans - control_center),
        vmin=-750,
        vmax=750,
        cmap="PuOr",
        extent=(depths[0], depths[-1], widths[0], widths[-1]),
    )
    ax1.set_title("Shelf center")
    ax2.imshow(
        np.flipud(outer_vels_chans - control_outer),
        vmin=-750,
        vmax=750,
        cmap="PuOr",
        extent=(depths[0], depths[-1], widths[0], widths[-1]),
    )
    ax2.set_title("Shelf edge")
    cm = ax3.imshow(
        np.flipud(gl_vels_chans - control_gl),
        vmin=-750,
        vmax=750,
        cmap="PuOr",
        extent=(depths[0], depths[-1], widths[0], widths[-1]),
    )
    plt.colorbar(cm, cax=cax, label=r"$\Delta u$ (m yr$^{-1}$)", extend="both")
    ax3.set_title("Grounding line")
    ax1.set_ylabel("Channel width (m)")
    ax1.set_xlabel("Channel depth (m)")
    ax2.set_xlabel("Channel depth (m)")
    ax3.set_xlabel("Channel depth (m)")
    for ax in (ax1, ax2, ax3):
        ax.axis("auto")

    for ax in (ax2, ax3):
        ax.set_yticklabels([])

    fig.tight_layout(pad=0.1)
    fig.savefig("plots/hybrid/{:s}_outer_inner_gl_unpaired.png".format(plot_pref), dpi=300)
    plt.close(fig)

    gs = gridspec.GridSpec(len(widths), len(depths) + 1, width_ratios=[1 for d in depths] + [0.08])
    fig = plt.figure(figsize=(12, 12))
    axes = [[None for j in depths] for i in widths]
    for i, channel_width in enumerate(widths):
        for j, channel_depth in enumerate(depths):
            axes[i][j] = fig.add_subplot(gs[i, j])
            ax = axes[i][j]
            ax.set_aspect(2)
            ax.set_xlim(40e4, 64e4)
            ax.set_ylim((0, Ly))
            key = (channel_width, channel_depth)
            if (results_even[key] is not None) and (results_chans[key] is not None):
                du = firedrake.project(
                    firedrake.project(results_even[key]["velocity"][0], Q2) - firedrake.assemble(interpolate(u_0xc, Q2)),
                    Q2,
                )
                colors = tripcolor(extract_surface(du), axes=ax, cmap="PuOr", vmin=-100, vmax=100)
                tricontour(height_above_flotation_control, levels=levels, axes=ax, colors=["k"])

                chansi = channel_func(
                    fine_mesh,
                    channel_depth,
                    left,
                    vert,
                    outer=right,
                    ramp_y=ramp_y,
                    y_width=channel_width,
                )
                chans = firedrake.project(chansi, Q2)
                tricontour(extract_surface(chans), levels=levels, axes=ax, colors=["0.6"])
        axes[i][0].set_ylabel(f"{channel_width / 1000} km wide", fontsize=12)

    cax = fig.add_subplot(gs[1:-1, -1])
    plt.colorbar(colors, cax=cax, extend="both", label=r"$\Delta$V (w/ - w/o melt)")

    for j in range(1, len(depths)):
        for i in range(len(widths)):
            axes[i][j].yaxis.set_tick_params(labelleft=False)

    for j, channel_depth in enumerate(depths):
        axes[0][j].set_title(str(channel_depth) + " m deep", fontsize=12)
    fig.savefig("plots/hybrid/{:s}_width_depth_even.png".format(plot_pref), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 12))
    axes = [[None for j in depths] for i in widths]
    for i, channel_width in enumerate(widths):
        for j, channel_depth in enumerate(depths):
            axes[i][j] = fig.add_subplot(gs[i, j])
            ax = axes[i][j]
            ax.set_aspect(2)
            ax.set_xlim(40e4, 64e4)
            ax.set_ylim((0, Ly))
            key = (channel_width, channel_depth)
            if (results_even[key] is not None) and (results_chans[key] is not None):
                du = firedrake.project(
                    firedrake.project(results_chans[key]["velocity"][0], Q2) - firedrake.assemble(interpolate(u_0xc, Q2)),
                    Q2,
                )
                colors = tripcolor(extract_surface(du), axes=ax, cmap="PuOr", vmin=-100, vmax=100)
                tricontour(height_above_flotation_control, levels=levels, axes=ax, colors=["k"])

                chansi = channel_func(
                    fine_mesh,
                    channel_depth,
                    left,
                    vert,
                    outer=right,
                    y_width=channel_width,
                )
                chans = firedrake.project(chansi, Q2)
                tricontour(extract_surface(chans), levels=levels, axes=ax, colors=["0.6"])
        axes[i][0].set_ylabel(f"{channel_width / 1000} km wide", fontsize=12)

    cax = fig.add_subplot(gs[1:-1, -1])
    plt.colorbar(colors, cax=cax, extend="both", label=r"$\Delta$V (w/ - w/o chan)")

    for j in range(1, len(depths)):
        for i in range(len(widths)):
            axes[i][j].yaxis.set_tick_params(labelleft=False)

    for j, channel_depth in enumerate(depths):
        axes[0][j].set_title(str(channel_depth) + " m deep", fontsize=12)
    fig.savefig("plots/hybrid/{:s}_width_depth_unpaired.png".format(plot_pref), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 5))
    axes = [[None for j in depths[::2]] for i in widths[::2]]
    gs_abbrv = gridspec.GridSpec(len(widths[::2]), len(depths[::2]) + 1, width_ratios=[1 for d in depths[::2]] + [0.08], wspace=0.01, hspace=0.05)
    for i, channel_width in enumerate(widths[::2]):
        for j, channel_depth in enumerate(depths[::2]):
            axes[i][j] = fig.add_subplot(gs_abbrv[i, j])
            ax = axes[i][j]
            ax.axis("equal")
            # ax.set_aspect(1)
            ax.set_xlim(4e5, 6.4e5)
            ax.set_ylim(0, 8e4)
            ax.set_yticks([0, 4e4, 8e4])
            ax.set_xticks([4e5, 5e5, 6e5])
            ax.tick_params(axis="both", which="major", labelsize=8)

            key = (channel_width, channel_depth)
            if (results_even[key] is not None) and (results_chans[key] is not None):
                du = firedrake.project(
                    firedrake.project(results_chans[key]["velocity"][0], Q2) - firedrake.project(results_even[key]["velocity"][0], Q2),
                    Q2,
                )
                colors = tripcolor(extract_surface(du), axes=ax, cmap="PuOr", vmin=-100, vmax=100)
                tricontour(height_above_flotation_control, levels=levels, axes=ax, colors=["k"])

                chansi = channel_func(
                    fine_mesh,
                    channel_depth,
                    left,
                    vert,
                    outer=right,
                    y_width=channel_width,
                )
                chans = firedrake.project(chansi, Q2)
                tricontour(extract_surface(chans), levels=levels, axes=ax, colors=["0.6"])
        axes[i][0].set_ylabel(f"{channel_width / 1000} km", fontsize=12)

    cax = fig.add_subplot(gs[1:-1, -1])
    plt.colorbar(colors, cax=cax, extend="both", label=r"$\Delta u_x$ (w/ - w/o chan)")

    for j in range(1, len(depths[::2])):
        for i in range(len(widths[::2])):
            axes[i][j].yaxis.set_tick_params(labelleft=False)

    for j in range(len(depths[::2])):
        for i in range(len(widths[::2]) - 1):
            axes[i][j].xaxis.set_tick_params(labelbottom=False)

    for j, channel_depth in enumerate(depths[::2]):
        axes[0][j].set_title(str(channel_depth) + " m deep", fontsize=12)
    fig.savefig("plots/hybrid/{:s}_width_depth_paired.png".format(plot_pref), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 12))
    axes = [[None for j in depths] for i in widths]
    for i, channel_width in enumerate(widths[2:]):
        for j, channel_depth in enumerate(depths[3:]):
            axes[i][j] = fig.add_subplot(gs[i, j])
            ax = axes[i][j]
            ax.set_aspect(2)
            ax.set_xlim(40e4, 64e4)
            ax.set_ylim((15000, 65000))
            key = (channel_width, channel_depth)
            if (results_even[key] is not None) and (results_chans[key] is not None):
                du = results_chans[key]["thickness"]
                colors = tripcolor(extract_surface(du), axes=ax, cmap="PiYG", vmin=-20, vmax=20)
                tricontour(height_above_flotation_control, levels=levels, axes=ax, colors=["k"])

                chansi = channel_func(
                    fine_mesh,
                    channel_depth,
                    left,
                    vert,
                    outer=right,
                    ramp_y=ramp_y,
                    y_width=channel_width,
                )
                chans = firedrake.project(chansi, Q2)
                tricontour(extract_surface(chans), levels=levels, axes=ax, colors=["0.6"])
        axes[i][0].set_ylabel(f"{channel_width / 1000} km wide", fontsize=12)

    cax = fig.add_subplot(gs[1:-1, -1])
    plt.colorbar(colors, cax=cax, extend="both", label=r"$\Delta$V (w/ - w/o chan)")

    for j in range(1, len(depths) - 3):
        for i in range(len(widths[2:])):
            axes[i][j].yaxis.set_tick_params(labelleft=False)

    for j, channel_depth in enumerate(depths[3:]):
        axes[0][j].set_title(str(channel_depth) + " m deep", fontsize=12)
    fig.savefig("plots/hybrid/{:s}_checkthickness.png".format(plot_pref), dpi=300)
    plt.close(fig)

    return
    epsilon_dot = sym_grad(u_0)
    tau = membrane_stress(strain_rate=epsilon_dot, fluidity=A)
    tau_xx = firedrake.project(tau[0, 0] * 1000, Q2)
    tau_xy = firedrake.project(tau[0, 1] * 1000, Q2)
    tau_yx = firedrake.project(tau[1, 0] * 1000, Q2)
    tau_yy = firedrake.project(tau[1, 1] * 1000, Q2)

    fig, ((ax1, ax2), (ax3, ax4), (cax1, cax2)) = plt.subplots(3, 2, figsize=(12, 12), gridspec_kw={"height_ratios": (1, 1, 0.05)})
    axes = [ax1, ax2, ax3, ax4]
    colors = tripcolor(extract_surface(tau_xx), axes=ax1, cmap="PiYG", vmin=-150, vmax=150)
    colors = tripcolor(extract_surface(tau_xy), axes=ax2, cmap="PiYG", vmin=-150, vmax=150)
    colors = tripcolor(extract_surface(tau_yx), axes=ax3, cmap="PiYG", vmin=-150, vmax=150)
    colors = tripcolor(extract_surface(tau_yy), axes=ax4, cmap="PiYG", vmin=-150, vmax=150)

    plt.colorbar(colors, cax=cax1, orientation="horizontal", extend="both", label="Stress [kPa]")
    for ax in axes:
        ax.set_aspect(2)
        ax.set_xlim(40e4, 64e4)
        ax.set_ylim((0, Ly))
        tricontour(height_above_flotation_control, levels=levels, axes=ax, colors=["k"])
    fig.savefig("plots/hybrid/{:s}_tau_xy.png".format(plot_pref), dpi=300)
    plt.close(fig)

    epsilon_dot_chan = sym_grad(results_chans[(widths[0], depths[-1])]["velocity"][-1])
    tau_chan = membrane_stress(strain_rate=epsilon_dot_chan, fluidity=A)
    tau_xx = firedrake.project(tau_chan[0, 0] * 1000, Q2)
    tau_xy = firedrake.project(tau_chan[0, 1] * 1000, Q2)
    tau_yx = firedrake.project(tau_chan[1, 0] * 1000, Q2)
    tau_yy = firedrake.project(tau_chan[1, 1] * 1000, Q2)

    fig, ((ax1, ax2), (ax3, ax4), (cax1, cax2)) = plt.subplots(3, 2, figsize=(12, 12), gridspec_kw={"height_ratios": (1, 1, 0.05)})
    axes = [ax1, ax2, ax3, ax4]
    colors = tripcolor(tau_xx, axes=ax1, cmap="PiYG", vmin=-150, vmax=150)
    colors = tripcolor(tau_xy, axes=ax2, cmap="PiYG", vmin=-150, vmax=150)
    colors = tripcolor(tau_yx, axes=ax3, cmap="PiYG", vmin=-150, vmax=150)
    colors = tripcolor(tau_yy, axes=ax4, cmap="PiYG", vmin=-150, vmax=150)

    plt.colorbar(colors, cax=cax1, orientation="horizontal", extend="both")
    for ax in axes:
        ax.set_aspect(2)
        ax.set_xlim(40e4, 64e4)
        ax.set_ylim((0, Ly))
        tricontour(height_above_flotation_control, levels=levels, axes=ax, colors=["k"])

    dtau_xx = firedrake.project((tau[0, 0] - tau_chan[0, 0]) * 1000, Q2)
    dtau_xy = firedrake.project((tau[0, 1] - tau_chan[0, 1]) * 1000, Q2)
    dtau_yx = firedrake.project((tau[1, 0] - tau_chan[1, 0]) * 1000, Q2)
    dtau_yy = firedrake.project((tau[1, 1] - tau_chan[1, 1]) * 1000, Q2)

    fig, ((ax1, ax2), (ax3, ax4), (cax1, cax2)) = plt.subplots(3, 2, figsize=(12, 12), gridspec_kw={"height_ratios": (1, 1, 0.05)})
    axes = [ax1, ax2, ax3, ax4]
    colors = tripcolor(dtau_xx, axes=ax1, cmap="PiYG", vmin=-100, vmax=100)
    colors = tripcolor(dtau_xy, axes=ax2, cmap="PiYG", vmin=-100, vmax=100)
    colors = tripcolor(dtau_yx, axes=ax3, cmap="PiYG", vmin=-100, vmax=100)
    colors = tripcolor(dtau_yy, axes=ax4, cmap="PiYG", vmin=-100, vmax=100)

    plt.colorbar(colors, cax=cax1, orientation="horizontal", extend="both")
    for ax in axes:
        ax.set_aspect(2)
        ax.set_xlim(40e4, 64e4)
        ax.set_ylim((0, Ly))
        tricontour(height_above_flotation_control, levels=[5], axes=ax, colors=["k"])

    eps_zz_chan = firedrake.project(-epsilon_dot_chan[0, 0] - epsilon_dot_chan[1, 1], Q2)
    eps_zz = firedrake.project(-epsilon_dot[0, 0] - epsilon_dot[1, 1], Q2)

    fig, ax = subplots()
    colors = tripcolor(eps_zz_chan, axes=ax, cmap="BrBG", vmin=-0.05, vmax=0.05)
    colorbar(fig, colors)
    ax.set_aspect(2)
    ax.set_xlim(40e4, 64e4)
    ax.set_ylim((0, Ly))

    fig, ax = subplots()
    colors = tripcolor(eps_zz, axes=ax, cmap="BrBG", vmin=-0.05, vmax=0.05)
    colorbar(fig, colors)
    ax.set_aspect(2)
    ax.set_xlim(40e4, 64e4)
    ax.set_ylim((0, Ly))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true")
    parser.add_argument("-t", action="store_true")
    parser.add_argument("-i", action="store_true")
    parser.add_argument("-o", action="store_true")
    parser.add_argument("-c", action="store_true")
    parser.add_argument("-m", action="store_true")
    parser.add_argument("-a", action="store_true")
    parser.add_argument("-f", action="store_true")
    args = parser.parse_args()

    if args.i and args.o:
        raise ValueError("Cannot be both inner and inner and outer")

    if args.a:
        for setup in [
            ts_name,
            "mismip",
        ]:
            for chan_name in ["full", "inner", "outer"]:
                if chan_name == "full":
                    poss = ["channels", "margin", "center"]
                else:
                    poss = ["channels"]
                for pos_name in poss:
                    channels_and_plots(setup, pos_name, chan_name, restart=args.r)
        for setup in [ts_name]:
            for chan_name in ["full"]:
                for pos_name in ["funky"]:
                    channels_and_plots(setup, pos_name, chan_name, restart=args.r)

    else:
        if args.t:
            setup = ts_name
        else:
            setup = "mismip"

        chan_name = "full"
        if args.o:
            chan_name = "outer"
        if args.i:
            chan_name = "inner"

        pos_name = "channels"
        if args.c:
            pos_name = "center"
        if args.m:
            pos_name = "margin"
        if args.f:
            pos_name = "funky"
        channels_and_plots(setup, pos_name, chan_name, restart=args.r)
