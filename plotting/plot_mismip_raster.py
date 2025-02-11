#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 dlilien <dlilien@noatak.local>
#
# Distributed under terms of the MIT license.

"""

"""
# Basic packaages
import string
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Modeling packages
import firedrake
from firedrake.__future__ import interpolate
from firedrake import Constant
from firedrake.pyplot import tripcolor, tricontour

# import icepack
# import icepack.plot
from icepack.models.viscosity import sym_grad

# my packages
# from icepackaccs.mismip import mismip_bed_topography
from icepackaccs import extract_surface
from libchannels import smooth_floating

from plot_mismip_bywidth import key_full, key_margin, key_center, key_inner, key_outer, center_pt, outer_pt, gl_pt

chan_fn_template = "../modeling/outputs/{:s}-hybrid-{:s}-{:s}.h5"
setup = "mismip"
pos_name = "channels"
full_fn = chan_fn_template.format(setup, pos_name, "full")
inner_fn = chan_fn_template.format(setup, pos_name, "inner")
outer_fn = chan_fn_template.format(setup, pos_name, "outer")
margin_fn = chan_fn_template.format(setup, "margin", "full")
center_fn = chan_fn_template.format(setup, "center", "full")

A = Constant(20)
C = Constant(1e-2)
a = Constant(0.3)

field_names = ["velocity", "thickness", "surface"]


def load():
    fields_3 = {}
    with firedrake.CheckpointFile("../modeling/outputs/mismip-fine-degree2_comp.h5", "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for keyt in field_names:
            fields_3[keyt] = chk.load_function(fine_mesh, keyt)

    # Q2 = fields_3["thickness"].function_space()
    # V2 = fields_3["velocity"].function_space()

    # h_0 = firedrake.project(fields_3["thickness"], Q2)
    # u_0c = firedrake.project(fields_3["velocity"], V2)

    # z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))
    # s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)
    # x = firedrake.SpatialCoordinate(fine_mesh)[0]
    # u_0xc = firedrake.project(u_0c[0], Q2)

    full_dict_even = {}
    full_dict_chans = {}
    with firedrake.CheckpointFile(full_fn, "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for f in ["thickness", "velocity", "surface", "tvm"]:
            full_dict_even[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} even".format(key_full[0], key_full[1], f),
            )
            full_dict_chans[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} chan".format(key_full[0], key_full[1], f),
            )
        v_full = chk.get_attr("{:d} {:d}".format(*key_full), "volume")

    inner_dict_even = {}
    inner_dict_chans = {}
    with firedrake.CheckpointFile(inner_fn, "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for f in ["thickness", "velocity", "surface", "tvm"]:
            inner_dict_even[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} even".format(key_inner[0], key_inner[1], f),
            )
            inner_dict_chans[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} chan".format(key_inner[0], key_inner[1], f),
            )
        v_inner = chk.get_attr("{:d} {:d}".format(*key_inner), "volume")

    outer_dict_even = {}
    outer_dict_chans = {}
    with firedrake.CheckpointFile(outer_fn, "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for f in ["thickness", "velocity", "surface", "tvm"]:
            outer_dict_even[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} even".format(key_outer[0], key_outer[1], f),
            )
            outer_dict_chans[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} chan".format(key_outer[0], key_outer[1], f),
            )

        v_outer = chk.get_attr("{:d} {:d}".format(*key_outer), "volume")

    margin_dict_even = {}
    margin_dict_chans = {}
    with firedrake.CheckpointFile(margin_fn, "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for f in ["thickness", "velocity", "surface", "tvm"]:
            margin_dict_even[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} even".format(key_margin[0], key_margin[1], f),
            )
            margin_dict_chans[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} chan".format(key_margin[0], key_margin[1], f),
            )
        v_margin = chk.get_attr("{:d} {:d}".format(*key_margin), "volume")

    center_dict_even = {}
    center_dict_chans = {}
    with firedrake.CheckpointFile(center_fn, "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for f in ["thickness", "velocity", "surface", "tvm"]:
            center_dict_even[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} even".format(key_center[0], key_center[1], f),
            )
            center_dict_chans[f] = chk.load_function(
                fine_mesh,
                "{:d} {:d} {:s} chan".format(key_center[0], key_center[1], f),
            )
        v_center = chk.get_attr("{:d} {:d}".format(*key_center), "volume")

    vols = np.array([v_full, v_inner, v_outer, v_margin, v_center])
    print("full", "inner", "outer", "margin", "center")
    print("Volumes [m^3]:", vols)
    print("Percentage more volume than 2 full: [%]", (vols - vols[0]) / vols[0] * 100)

    even_dicts = [full_dict_even, inner_dict_even, outer_dict_even, margin_dict_even, center_dict_even]
    chans_dicts = [full_dict_chans, inner_dict_chans, outer_dict_chans, margin_dict_chans, center_dict_chans]
    return even_dicts, chans_dicts


def mismip_raster_inner_outer_full(even_dicts, chans_dicts):
    gs = gridspec.GridSpec(4, 4, width_ratios=(1, 1, 1, 0.04), top=0.98, left=0.075, wspace=0.10, right=0.905, bottom=0.12)
    fig = plt.figure(figsize=(7.0, 3.2))
    h_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    u_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    du_axes = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]

    ss_axes = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])]

    cax_h = fig.add_subplot(gs[0, 3])
    cax_u = fig.add_subplot(gs[1, 3])
    cax_du = fig.add_subplot(gs[2, 3])
    cax_ss = fig.add_subplot(gs[3, 3])

    for h_ax, u_ax, du_ax, ss_ax, even_dict, chan_dict in zip(h_axes, u_axes, du_axes, ss_axes, even_dicts, chans_dicts):
        Q2 = even_dict["thickness"].function_space()
        du = firedrake.project(
            firedrake.assemble(interpolate(chan_dict["velocity"][0], Q2)) - firedrake.assemble(interpolate(even_dict["velocity"][0], Q2)),
            Q2,
        )
        cm_h = tripcolor(extract_surface(chan_dict["thickness"]), vmin=0, vmax=1000, cmap="viridis", axes=h_ax)
        cm_u = tripcolor(
            extract_surface(firedrake.assemble(interpolate(chan_dict["velocity"][0], Q2))),
            vmin=0,
            vmax=1000,
            cmap="Reds",
            axes=u_ax,
        )
        cm_du = tripcolor(extract_surface(du), vmin=-250, vmax=250, cmap="PuOr", axes=du_ax)

        epsilon_dot = sym_grad(chan_dict["velocity"])
        cm_ss = tripcolor(
            extract_surface(firedrake.project(epsilon_dot[0, 1], Q2)), vmin=-0.05, vmax=0.05, cmap="PiYG", axes=ss_ax
        )

        is_floating = smooth_floating(250, extract_surface(chan_dict["surface"]), extract_surface(chan_dict["thickness"]))
        tricontour(is_floating, levels=[0], colors="k", axes=h_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=u_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=du_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=ss_ax)

        vm_cutoff = 265  # 265  # from Grinsted et al.
        tau_vm = extract_surface(firedrake.project(chan_dict["tvm"], Q2))
        tricontour(tau_vm, levels=[vm_cutoff], colors="0.6", axes=ss_ax)

    cbr_h = plt.colorbar(cm_h, cax=cax_h, extend="max")
    cbr_h.set_label("H [m]", fontsize=8)
    cbr_u = plt.colorbar(cm_u, cax=cax_u, extend="max")
    cbr_u.set_label(r"$u_x$ [m yr$^{-1}$]", fontsize=8)
    cbr_du = plt.colorbar(cm_du, cax=cax_du, extend="both")
    cbr_du.set_label(r"$\Delta u_x$ [m yr$^{-1}$]", fontsize=8)
    cbr_ss = plt.colorbar(cm_ss, cax=cax_ss, extend="both")
    cbr_ss.set_label(r"$\dot{\epsilon}_{xy}$ [yr$^{-1}$]", fontsize=8)

    for cax in [cax_h, cax_u, cax_du, cax_ss]:
        cax.tick_params(axis="both", which="major", labelsize=8)

    for i, ax in enumerate(h_axes + u_axes + du_axes + ss_axes):
        ax.axis("equal")
        ax.text(0.01, 0.97, string.ascii_lowercase[i], fontsize=12, ha="left", va="top", transform=ax.transAxes)
        ax.set_xlim(4e5, 6.4e5)
        ax.set_ylim(0, 8e4)
        ax.set_yticks([0, 4e4, 8e4])
        ax.set_xticks([4e5, 5e5, 6e5])
        ax.tick_params(axis="both", which="major", labelsize=8)
        if ax in [h_axes[0], u_axes[0], du_axes[0], ss_axes[0]]:
            ax.set_yticklabels(["0", "40", "80"])
        else:
            ax.set_yticklabels(["", "", ""])

        if ax in du_axes:
            ax.plot(*center_pt, marker="o", linestyle="none", color="0.3")
            ax.plot(*gl_pt, marker="s", linestyle="none", color="0.7")
            ax.plot(*outer_pt, marker="d", linestyle="none", color="0.5")
        if ax in ss_axes:
            ax.set_xticklabels(["400", "500", "600"])
        else:
            ax.set_xticklabels(["", "", ""])

    # u_axes[0].set_ylabel("Distance (km)", fontsize=8)
    u_axes[0].text(-0.25, -0.1, "Distance (km)", rotation=90, ha="center", va="center", fontsize=8, transform=u_axes[0].transAxes)
    ss_axes[1].set_xlabel("Distance (km)", fontsize=8)
    fig.savefig("../plots/hybrid/mismip_channel_raster_inner_outer_full.png", dpi=300)


def mismip_raster_both_edge_center(even_dicts, chans_dicts):
    gs = gridspec.GridSpec(4, 4, width_ratios=(1, 1, 1, 0.04), top=0.98, left=0.075, wspace=0.10, right=0.905, bottom=0.12)
    fig = plt.figure(figsize=(7.0, 3.2))
    h_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]

    u_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]

    du_axes = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]

    ss_axes = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2])]

    cax_h = fig.add_subplot(gs[0, 3])
    cax_u = fig.add_subplot(gs[1, 3])
    cax_du = fig.add_subplot(gs[2, 3])
    cax_ss = fig.add_subplot(gs[3, 3])

    for h_ax, u_ax, du_ax, ss_ax, even_dict, chan_dict in zip(h_axes, u_axes, du_axes, ss_axes, [even_dicts[0]] + even_dicts[3:], [chans_dicts[0]] + chans_dicts[3:]):
        Q2 = even_dict["thickness"].function_space()
        du = firedrake.project(
            firedrake.assemble(interpolate(chan_dict["velocity"][0], Q2)) - firedrake.assemble(interpolate(even_dict["velocity"][0], Q2)),
            Q2,
        )
        cm_h = tripcolor(extract_surface(chan_dict["thickness"]), vmin=0, vmax=1000, cmap="viridis", axes=h_ax)
        cm_u = tripcolor(
            extract_surface(firedrake.assemble(interpolate(chan_dict["velocity"][0], Q2))),
            vmin=0,
            vmax=1000,
            cmap="Reds",
            axes=u_ax,
        )
        cm_du = tripcolor(extract_surface(du), vmin=-250, vmax=250, cmap="PuOr", axes=du_ax)

        epsilon_dot = sym_grad(chan_dict["velocity"])
        cm_ss = tripcolor(
            extract_surface(firedrake.project(epsilon_dot[0, 1], Q2)), vmin=-0.05, vmax=0.05, cmap="PiYG", axes=ss_ax
        )

        is_floating = smooth_floating(250, extract_surface(chan_dict["surface"]), extract_surface(chan_dict["thickness"]))
        tricontour(is_floating, levels=[0], colors="k", axes=h_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=u_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=du_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=ss_ax)

        vm_cutoff = 265  # 265  # from Grinsted et al.
        tau_vm = extract_surface(firedrake.project(chan_dict["tvm"], Q2))
        tricontour(tau_vm, levels=[vm_cutoff], colors="0.6", axes=ss_ax)

    cbr_h = plt.colorbar(cm_h, cax=cax_h, extend="max")
    cbr_h.set_label("H [m]", fontsize=8)
    cbr_u = plt.colorbar(cm_u, cax=cax_u, extend="max")
    cbr_u.set_label(r"$u_x$ [m yr$^{-1}$]", fontsize=8)
    cbr_du = plt.colorbar(cm_du, cax=cax_du, extend="both")
    cbr_du.set_label(r"$\Delta u_x$ [m yr$^{-1}$]", fontsize=8)
    cbr_ss = plt.colorbar(cm_ss, cax=cax_ss, extend="both")
    cbr_ss.set_label(r"$\dot{\epsilon}_{xy}$ [yr$^{-1}$]", fontsize=8)

    for cax in [cax_h, cax_u, cax_du, cax_ss]:
        cax.tick_params(axis="both", which="major", labelsize=8)

    for i, ax in enumerate(h_axes + u_axes + du_axes + ss_axes):
        ax.axis("equal")
        ax.text(0.01, 0.97, string.ascii_lowercase[i], fontsize=12, ha="left", va="top", transform=ax.transAxes)
        ax.set_xlim(4e5, 6.4e5)
        ax.set_ylim(0, 8e4)
        ax.set_yticks([0, 4e4, 8e4])
        ax.set_xticks([4e5, 5e5, 6e5])
        ax.tick_params(axis="both", which="major", labelsize=8)
        if ax in [h_axes[0], u_axes[0], du_axes[0], ss_axes[0]]:
            ax.set_yticklabels(["0", "40", "80"])
        else:
            ax.set_yticklabels(["", "", ""])

        if ax in du_axes:
            ax.plot(*center_pt, marker="o", linestyle="none", color="0.3")
            ax.plot(*gl_pt, marker="s", linestyle="none", color="0.7")
            ax.plot(*outer_pt, marker="d", linestyle="none", color="0.5")
        if ax in ss_axes:
            ax.set_xticklabels(["400", "500", "600"])
        else:
            ax.set_xticklabels(["", "", ""])

    # u_axes[0].set_ylabel("Distance (km)", fontsize=8)
    u_axes[0].text(-0.25, -0.1, "Distance (km)", rotation=90, ha="center", va="center", fontsize=8, transform=u_axes[0].transAxes)
    ss_axes[1].set_xlabel("Distance (km)", fontsize=8)
    fig.savefig("../plots/hybrid/mismip_channel_raster_full_edge_center.png", dpi=300)


def mismip_raster_inner_outer_full_side(even_dicts, chans_dicts):
    gs = gridspec.GridSpec(4, 5, width_ratios=(1, 1, 1, 1, 0.05), top=0.98, left=0.060, wspace=0.27, right=0.905, bottom=0.10)
    fig = plt.figure(figsize=(7.0, 4.0))
    h_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]

    u_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[1, 3])]

    du_axes = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2]), fig.add_subplot(gs[2, 3])]

    ss_axes = [fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1]), fig.add_subplot(gs[3, 2]), fig.add_subplot(gs[3, 3])]

    cax_h = fig.add_subplot(gs[0, 4])
    cax_u = fig.add_subplot(gs[1, 4])
    cax_du = fig.add_subplot(gs[2, 4])
    cax_ss = fig.add_subplot(gs[3, 4])

    for h_ax, u_ax, du_ax, ss_ax, even_dict, chan_dict in zip(h_axes, u_axes, du_axes, ss_axes, even_dicts, chans_dicts):
        Q2 = even_dict["thickness"].function_space()
        du = firedrake.project(
            firedrake.assemble(interpolate(chan_dict["velocity"][0], Q2)) - firedrake.assemble(interpolate(even_dict["velocity"][0], Q2)),
            Q2,
        )
        cm_h = tripcolor(extract_surface(chan_dict["thickness"]), vmin=0, vmax=1000, cmap="viridis", axes=h_ax)
        cm_u = tripcolor(
            extract_surface(firedrake.assemble(interpolate(chan_dict["velocity"][0], Q2))),
            vmin=0,
            vmax=1000,
            cmap="Reds",
            axes=u_ax,
        )
        cm_du = tripcolor(extract_surface(du), vmin=-250, vmax=250, cmap="PuOr", axes=du_ax)

        epsilon_dot = sym_grad(chan_dict["velocity"])
        cm_ss = tripcolor(
            extract_surface(firedrake.project(epsilon_dot[0, 1], Q2)), vmin=-0.05, vmax=0.05, cmap="PiYG", axes=ss_ax
        )

        is_floating = smooth_floating(250, extract_surface(chan_dict["surface"]), extract_surface(chan_dict["thickness"]))
        tricontour(is_floating, levels=[0], colors="k", axes=h_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=u_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=du_ax)
        tricontour(is_floating, levels=[0], colors="k", axes=ss_ax)

        vm_cutoff = 265  # 265  # from Grinsted et al.
        tau_vm = extract_surface(firedrake.project(chan_dict["tvm"], Q2))
        tricontour(tau_vm, levels=[vm_cutoff], colors="0.6", axes=ss_ax)

    cbr_h = plt.colorbar(cm_h, cax=cax_h, extend="max")
    cbr_h.set_label("Thickness [m]", fontsize=8)
    cbr_u = plt.colorbar(cm_u, cax=cax_u, extend="max")
    cbr_u.set_label(r"$u_x$ [m yr$^{-1}$]", fontsize=8)
    cbr_du = plt.colorbar(cm_du, cax=cax_du, extend="both")
    cbr_du.set_label(r"$\Delta u_x$ [m yr$^{-1}$]", fontsize=8)
    cbr_ss = plt.colorbar(cm_ss, cax=cax_ss, extend="both")
    cbr_ss.set_label(r"$\dot{\epsilon}_{xy}$ [yr$^{-1}$]", fontsize=8)

    for cax in [cax_h, cax_u, cax_du, cax_ss]:
        cax.tick_params(axis="both", which="major", labelsize=8)

    for i, ax in enumerate(h_axes + u_axes + du_axes + ss_axes):
        ax.text(0.01, 0.97, string.ascii_lowercase[i], fontsize=12, ha="left", va="top", transform=ax.transAxes)
        ax.set_xlim(2.4e5, 6.4e5)
        ax.set_ylim(0, 8e4)
        ax.set_yticks([0, 4e4, 8e4])
        ax.set_xticks([24e4, 44e4, 64e4])
        ax.tick_params(axis="both", which="major", labelsize=8)
        if ax in [h_axes[0], u_axes[0], du_axes[0], ss_axes[0]]:
            ax.set_yticklabels(["0", "40", "80"])
        else:
            ax.set_yticklabels(["", "", ""])

        if ax in du_axes:
            ax.plot(*center_pt, marker="o", linestyle="none", color="0.3")
            ax.plot(*gl_pt, marker="s", linestyle="none", color="0.7")
        if ax in ss_axes:
            ax.set_xticklabels(["240", "440", "640"])
        else:
            ax.set_xticklabels(["", "", ""])

    # u_axes[0].set_ylabel("Distance (km)", fontsize=8)
    u_axes[0].text(-0.25, -0.1, "Distance (km)", rotation=90, ha="center", va="center", fontsize=8, transform=u_axes[0].transAxes)
    ss_axes[1].text(1.1, -0.29, "Distance (km)", ha="center", va="top", fontsize=8, transform=ss_axes[1].transAxes)
    fig.savefig("../plots/hybrid/mismip_channel_raster_all.png", dpi=300)


if __name__ == "__main__":
    even_dicts, chans_dicts = load()
    mismip_raster_inner_outer_full(even_dicts, chans_dicts)
    mismip_raster_both_edge_center(even_dicts, chans_dicts)
    # mismip_raster_inner_outer_full_side(even_dicts, chans_dicts)
