#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 dlilien <dlilien@noatak.local>
#
# Distributed under terms of the MIT license.

"""

"""
# Basic packaages
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Modeling packages
import firedrake
from firedrake.__future__ import interpolate
import icepack
import icepack.plot

# my packages
from icepackaccs.mismip import mismip_bed_topography
from icepackaccs import extract_surface

from hybrid_channels import widths, depths

center_pt = (630000, 40000)
outer_pt = (630000, 10000)
gl_pt = (437262, 40000)
ugl_pt = (427262, 40000)

key_full = (1500, 200)
key_margin = (4000, 200)
key_center = (4000, 200)
key_inner = (5000, 200)
key_outer = (3000, 200)

names = [
    "full_center_vels_chans",
    "full_center_vels_even",
    "full_outer_vels_chans",
    "full_outer_vels_even",
    "full_gl_vels_chans",
    "full_gl_vels_even",
    "full_ugl_vels_chans",
    "full_ugl_vels_even",
    "inner_center_vels_chans",
    "inner_center_vels_even",
    "inner_outer_vels_chans",
    "inner_outer_vels_even",
    "inner_gl_vels_chans",
    "inner_gl_vels_even",
    "inner_ugl_vels_chans",
    "inner_ugl_vels_even",
    "outer_center_vels_chans",
    "outer_center_vels_even",
    "outer_outer_vels_chans",
    "outer_outer_vels_even",
    "outer_gl_vels_chans",
    "outer_gl_vels_even",
    "outer_ugl_vels_chans",
    "outer_ugl_vels_even",
    "margin_center_vels_chans",
    "margin_center_vels_even",
    "margin_outer_vels_chans",
    "margin_outer_vels_even",
    "margin_gl_vels_chans",
    "margin_gl_vels_even",
    "margin_ugl_vels_chans",
    "margin_ugl_vels_even",
    "center_center_vels_chans",
    "center_center_vels_even",
    "center_outer_vels_chans",
    "center_outer_vels_even",
    "center_gl_vels_chans",
    "center_gl_vels_even",
    "center_ugl_vels_chans",
    "center_ugl_vels_even",
]


def relative_single(relative_changes, relative_changes_gl):
    gs = GridSpec(
        4,
        2,
        left=0.16,
        right=0.87,
        top=0.98,
        bottom=0.105,
        hspace=0.15,
        wspace=0.05,
        width_ratios=(1, 0.05),
        height_ratios=(1, 2, 2, 1),
    )
    fig = plt.figure(figsize=(3.4, 4))
    ax1 = fig.add_subplot(gs[:2, 0])
    ax2 = fig.add_subplot(gs[2:, 0])
    cax = fig.add_subplot(gs[1:3, 1])
    cm = ax1.imshow(
        np.flipud(relative_changes.T),
        vmin=35,
        vmax=65,
        cmap="PiYG",
        extent=(-0.25, 5.25, -25, 275),
    )
    cm = ax2.imshow(
        np.flipud(relative_changes_gl.T),
        vmin=35,
        vmax=65,
        cmap="PiYG",
        extent=(-0.25, 5.25, -25, 275),
    )
    plt.colorbar(cm, extend="both", label=r"Relative $\Delta u$ with 1 vs 2 channels [\%]", cax=cax)
    for i, ax in enumerate([ax1, ax2]):
        ax.axis("auto")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 250)
        ax.text(0.1, 245, "ab"[i], ha="left", va="top", fontsize=12)
    ax1.set_ylabel("Channel depth [m]")
    ax1.set_xticklabels([])
    ax2.set_ylabel("Channel depth [m]")
    ax2.set_xlabel("Channel width [km]")
    fig.tight_layout(pad=0.1)
    fig.savefig("../plots/hybrid/relative_effect_single_channel.png", dpi=300)


def marker_plot_simplified():
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 3.25),
        gridspec_kw={"hspace": 0.1, "right": 0.910, "top": 0.98, "left": 0.095, "bottom": 0.13},
    )
    pwidth = 2
    colors = {"full": "C0", "inner": "C3", "outer": "C6", "margin": "C1", "center": "C2"}
    names = {"full": "Two full marginal", "inner": "Two inner marginal", "outer": "Two outer marginal", "margin": "One marginal", "center": "One central"}
    for pref in ["full", "inner", "outer", "center", "margin"]:
        center = eval("{:s}_center_vels_chans - {:s}_center_vels_even".format(pref, pref))
        gl = eval("{:s}_gl_vels_chans - {:s}_gl_vels_even".format(pref, pref))
        for j, depth in enumerate(depths):
            label = None
            if j == 0:
                label = names[pref]
            ax1.fill([depth - pwidth, depth + pwidth, depth + pwidth, depth - pwidth, depth - pwidth], [center[0, j], center[0, j], center[-1, j], center[-1, j], center[0, j]], color=colors[pref], label=label, alpha=0.5)
            ax2.fill([depth - pwidth, depth + pwidth, depth + pwidth, depth - pwidth, depth - pwidth], [gl[0, j], gl[0, j], gl[-1, j], gl[-1, j], gl[0, j]], color=colors[pref], alpha=0.5)
    ax1.set_ylim(0, 1500)
    ax2.set_ylim(0, 400)
    ax1.set_xlim(40, 260)
    ax2.set_xlim(40, 260)

    ax1_perc = ax1.twinx()
    ax1_perc.set_ylim(0, 100 * ax1.get_ylim()[1] / control_center)
    ax2_perc = ax2.twinx()
    ax2_perc.set_ylim(0, 100 * ax2.get_ylim()[1] / control_gl)

    ax1.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{a }\fontsize{10pt}{3em}\selectfont{}{Mid shelf}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax1.transAxes,
    )
    ax2.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{b }\fontsize{10pt}{3em}\selectfont{}{At Grounding line}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax2.transAxes,
    )
    ax2.set_xlabel("Channel depth [m]")
    ax1.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax2.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax1_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax2_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax1.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    # ax2.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout(pad=0.1)
    fig.savefig("../plots/hybrid/comp_inner_outer_single_mismip_simp.pdf")


def marker_plot_up_simplified():
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 3.25),
        gridspec_kw={"hspace": 0.1, "right": 0.910, "top": 0.98, "left": 0.095, "bottom": 0.13},
    )
    pwidth = 2
    colors = {"full": "C0", "inner": "C3", "outer": "C6", "margin": "C1", "center": "C2"}
    names = {"full": "Two full marginal", "inner": "Two inner marginal", "outer": "Two outer marginal", "margin": "One marginal", "center": "One central"}
    for pref in ["full", "inner", "outer", "center", "margin"]:
        center = eval("{:s}_center_vels_chans - control_center".format(pref))
        gl = eval("{:s}_gl_vels_chans - control_gl".format(pref))
        for j, depth in enumerate(depths):
            label = None
            if j == 0:
                label = names[pref]
            ax1.fill([depth - pwidth, depth + pwidth, depth + pwidth, depth - pwidth, depth - pwidth], [center[0, j], center[0, j], center[-1, j], center[-1, j], center[0, j]], color=colors[pref], label=label, alpha=0.5)
            ax2.fill([depth - pwidth, depth + pwidth, depth + pwidth, depth - pwidth, depth - pwidth], [gl[0, j], gl[0, j], gl[-1, j], gl[-1, j], gl[0, j]], color=colors[pref], alpha=0.5)
    ax1.set_ylim(0, 1500)
    ax2.set_ylim(0, 400)
    ax1.set_xlim(40, 260)
    ax2.set_xlim(40, 260)

    ax1_perc = ax1.twinx()
    ax1_perc.set_ylim(0, 100 * ax1.get_ylim()[1] / control_center)
    ax2_perc = ax2.twinx()
    ax2_perc.set_ylim(0, 100 * ax2.get_ylim()[1] / control_gl)

    ax1.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{a }\fontsize{10pt}{3em}\selectfont{}{Mid shelf}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax1.transAxes,
    )
    ax2.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{b }\fontsize{10pt}{3em}\selectfont{}{At Grounding line}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax2.transAxes,
    )
    ax2.set_xlabel("Channel depth [m]")
    ax1.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax2.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax1_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax2_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax1.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    # ax2.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout(pad=0.1)
    fig.savefig("../plots/hybrid/comp_inner_outer_single_mismip_up_simp.pdf")


def marker_plot():
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 3.25),
        gridspec_kw={"hspace": 0.1, "right": 0.910, "top": 0.98, "left": 0.095, "bottom": 0.13},
    )
    lines = -1
    for k, width in enumerate(widths):
        if width in [500, 2500, 5000]:
            i = np.where(widths == width)[0][0]
            lines += 1
            ax1.plot(
                depths,
                full_center_vels_chans[i, :] - full_center_vels_even[i, :],
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
            )
            ax2.plot(
                depths,
                full_gl_vels_chans[i, :] - full_gl_vels_even[i, :],
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
            )
            print(full_gl_vels_chans[i, :] - full_gl_vels_even[i, :])

            ax1.plot(
                depths + 6,
                inner_center_vels_chans[i, :] - inner_center_vels_even[i, :],
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="w",
            )
            ax2.plot(
                depths + 6,
                inner_gl_vels_chans[i, :] - inner_gl_vels_even[i, :],
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="w",
            )

            ax1.plot(
                depths - 6,
                outer_center_vels_chans[i, :] - outer_center_vels_even[i, :],
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="0.6",
            )
            ax2.plot(
                depths - 6,
                outer_gl_vels_chans[i, :] - outer_gl_vels_even[i, :],
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="0.6",
            )

            ax1.plot(
                depths - 3,
                center_center_vels_chans[i, :] - center_center_vels_even[i, :],
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
                markerfacecoloralt="w",
                fillstyle="top",
            )
            ax2.plot(
                depths - 3,
                center_gl_vels_chans[i, :] - center_gl_vels_even[i, :],
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
                markerfacecoloralt="w",
                fillstyle="top",
            )

            ax1.plot(
                depths + 3,
                margin_center_vels_chans[i, :] - margin_center_vels_even[i, :],
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
                markerfacecoloralt="w",
                fillstyle="right",
            )
            ax2.plot(
                depths + 3,
                margin_gl_vels_chans[i, :] - margin_gl_vels_even[i, :],
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
                markerfacecoloralt="w",
                fillstyle="right",
            )

            ax1.plot([], [], linestyle="none", marker="o", color="C{:d}".format(lines + 4), label="{:d}-m wide".format(width))

    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.6", markerfacecolor="k", label="Two full marginal")
    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.6", markerfacecolor="w", label="Two inner marginal")
    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.4", markerfacecolor="0.6", label="Two outer marginal")
    ax2.plot(
        [],
        [],
        linestyle="none",
        marker="s",
        markeredgecolor="0.4",
        markerfacecolor="k",
        markerfacecoloralt="w",
        fillstyle="top",
        label="One central",
    )
    ax2.plot(
        [],
        [],
        linestyle="none",
        marker="s",
        markeredgecolor="0.4",
        markerfacecolor="k",
        markerfacecoloralt="w",
        fillstyle="right",
        label="One marginal",
    )
    # ax2.plot([], [], linestyle='none', marker='o', color='k', label='Shelf center')
    # ax.plot([], [], linestyle='none', marker='s', color='k', label='Shelf edge')
    # ax2.plot([], [], linestyle='none', marker='s', color='k', label='At GL')
    ax1.set_ylim(0, 1500)
    ax2.set_ylim(0, 400)
    ax1.set_xlim(40, 260)
    ax2.set_xlim(40, 260)

    ax1_perc = ax1.twinx()
    ax1_perc.set_ylim(0, 100 * ax1.get_ylim()[1] / control_center)
    ax2_perc = ax2.twinx()
    ax2_perc.set_ylim(0, 100 * ax2.get_ylim()[1] / control_gl)

    ax1.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{a }\fontsize{10pt}{3em}\selectfont{}{Mid shelf}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax1.transAxes,
    )
    ax2.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{b }\fontsize{10pt}{3em}\selectfont{}{At Grounding line}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax2.transAxes,
    )
    ax2.set_xlabel("Channel depth [m]")
    ax1.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax2.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax1_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax2_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax1.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    ax2.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout(pad=0.1)
    fig.savefig("../plots/hybrid/comp_inner_outer_single_mismip.pdf")


def marker_plot_unpaired():
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 3.25),
        gridspec_kw={"hspace": 0.1, "right": 0.910, "top": 0.98, "left": 0.095, "bottom": 0.13},
    )
    lines = -1
    for k, width in enumerate(widths):
        if width in [500, 2500, 5000]:
            i = np.where(widths == width)[0][0]
            lines += 1
            ax1.plot(
                depths,
                full_center_vels_chans[i, :] - control_center,
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
            )
            ax2.plot(
                depths,
                full_gl_vels_chans[i, :] - control_gl,
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
            )

            ax1.plot(
                depths + 6,
                inner_center_vels_chans[i, :] - control_center,
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="w",
            )
            ax2.plot(
                depths + 6,
                inner_gl_vels_chans[i, :] - control_gl,
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="w",
            )

            ax1.plot(
                depths - 6,
                outer_center_vels_chans[i, :] - control_center,
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="0.6",
            )
            ax2.plot(
                depths - 6,
                outer_gl_vels_chans[i, :] - control_gl,
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="0.6",
            )

            ax1.plot(
                depths - 3,
                center_center_vels_chans[i, :] - control_center,
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
                markerfacecoloralt="w",
                fillstyle="top",
            )
            ax2.plot(
                depths - 3,
                center_gl_vels_chans[i, :] - control_gl,
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
                markerfacecoloralt="w",
                fillstyle="top",
            )

            ax1.plot(
                depths + 3,
                margin_center_vels_chans[i, :] - control_center,
                linestyle="none",
                marker="o",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
                markerfacecoloralt="w",
                fillstyle="right",
            )
            ax2.plot(
                depths + 3,
                margin_gl_vels_chans[i, :] - control_gl,
                linestyle="none",
                marker="s",
                markeredgecolor="C{:d}".format(lines + 4),
                markerfacecolor="k",
                markerfacecoloralt="w",
                fillstyle="right",
            )

            ax1.plot([], [], linestyle="none", marker="o", color="C{:d}".format(lines + 4), label="{:d}-m wide".format(width))

    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.6", markerfacecolor="k", label="Two full marginal")
    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.6", markerfacecolor="w", label="Two inner marginal")
    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.4", markerfacecolor="0.6", label="Two outer marginal")
    ax2.plot(
        [],
        [],
        linestyle="none",
        marker="s",
        markeredgecolor="0.4",
        markerfacecolor="k",
        markerfacecoloralt="w",
        fillstyle="top",
        label="One central",
    )
    ax2.plot(
        [],
        [],
        linestyle="none",
        marker="s",
        markeredgecolor="0.4",
        markerfacecolor="k",
        markerfacecoloralt="w",
        fillstyle="right",
        label="One marginal",
    )
    # ax2.plot([], [], linestyle='none', marker='o', color='k', label='Shelf center')
    # ax.plot([], [], linestyle='none', marker='s', color='k', label='Shelf edge')
    # ax2.plot([], [], linestyle='none', marker='s', color='k', label='At GL')
    ax1.set_ylim(0, 1500)
    ax2.set_ylim(0, 400)

    ax1_perc = ax1.twinx()
    ax1_perc.set_ylim(0, 100 * ax1.get_ylim()[1] / control_center)
    ax2_perc = ax2.twinx()
    ax2_perc.set_ylim(0, 100 * ax2.get_ylim()[1] / control_gl)

    ax1.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{a }\fontsize{10pt}{3em}\selectfont{}{Mid shelf}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax1.transAxes,
    )
    ax2.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{b }\fontsize{10pt}{3em}\selectfont{}{At Grounding line}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax2.transAxes,
    )
    ax2.set_xlabel("Channel depth [m]")
    ax1.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax2.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax1_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax2_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax1.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    ax2.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout(pad=0.1)
    fig.savefig("../plots/hybrid/comp_inner_outer_single_mismip_up.pdf")


if __name__ == "__main__":
    params = {
        "text.usetex": "true",
        "font.family": "sans-serif",
        "font.sans-serif": "cmss",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Bitstream Vera Sans",
        "mathtext.it": "Bitstream Vera Sans:italic",
        "mathtext.bf": "Bitstream Vera Sans:bold",
        "text.latex.preamble": r"\usepackage{cmbright}",
    }
    plt.rcParams.update(params)

    depths = np.array(depths)
    widths = np.array(widths)

    chan_fn_template = "../modeling/outputs/{:s}-hybrid-{:s}-{:s}.h5"
    setup = "mismip"
    pos_name = "channels"
    full_fn = chan_fn_template.format(setup, pos_name, "full")
    inner_fn = chan_fn_template.format(setup, pos_name, "inner")
    outer_fn = chan_fn_template.format(setup, pos_name, "outer")
    center_fn = chan_fn_template.format(setup, "center", "full")
    margin_fn = chan_fn_template.format(setup, "margin", "full")

    field_names = ["velocity", "thickness", "surface"]

    fields_3 = {}
    with firedrake.CheckpointFile("../modeling/outputs/mismip-fine-degree2_comp.h5", "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for key in field_names:
            fields_3[key] = chk.load_function(fine_mesh, key)

    Q2 = fields_3["thickness"].function_space()
    V2 = fields_3["velocity"].function_space()

    h_0 = firedrake.project(fields_3["thickness"], Q2)
    u_0c = firedrake.project(fields_3["velocity"], V2)

    z_b = firedrake.assemble(interpolate(mismip_bed_topography(fine_mesh), Q2))
    s_0 = icepack.compute_surface(thickness=h_0, bed=z_b)

    x = firedrake.SpatialCoordinate(fine_mesh)[0]

    u_0xc = firedrake.project(u_0c[0], Q2)

    control_center = extract_surface(u_0xc).at(*center_pt)
    control_outer = extract_surface(u_0xc).at(*outer_pt)
    control_gl = extract_surface(u_0xc).at(*gl_pt)
    control_ugl = extract_surface(u_0xc).at(*ugl_pt)

    cache_name = "pointwise_mismip.npz"
    if not os.path.exists(cache_name):
        full_dict_even = {}
        full_dict_chans = {}
        with firedrake.CheckpointFile(full_fn, "r") as chk:
            fine_mesh = chk.load_mesh(name="fine_mesh")
            for channel_width in widths:
                for channel_depth in depths:
                    key = (channel_width, channel_depth)
                    full_dict_even[key] = {}
                    full_dict_chans[key] = {}
                    for f in ["thickness", "velocity", "surface"]:
                        full_dict_even[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} even".format(channel_width, channel_depth, f),
                        )
                        full_dict_chans[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} chan".format(channel_width, channel_depth, f),
                        )
            Q2 = full_dict_even[(widths[0], depths[0])]["thickness"].function_space()
            V2 = full_dict_even[(widths[0], depths[0])]["velocity"].function_space()

        full_center_vels_chans = np.empty((len(widths), len(depths)))
        full_center_vels_even = np.empty((len(widths), len(depths)))
        full_outer_vels_chans = np.empty((len(widths), len(depths)))
        full_outer_vels_even = np.empty((len(widths), len(depths)))
        full_gl_vels_chans = np.empty((len(widths), len(depths)))
        full_gl_vels_even = np.empty((len(widths), len(depths)))
        full_ugl_vels_chans = np.empty((len(widths), len(depths)))
        full_ugl_vels_even = np.empty((len(widths), len(depths)))

        for i, channel_width in enumerate(widths):
            for j, channel_depth in enumerate(depths):
                full_center_vels_chans[i, j] = extract_surface(firedrake.project(full_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                full_center_vels_even[i, j] = extract_surface(firedrake.project(full_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                full_outer_vels_chans[i, j] = extract_surface(firedrake.project(full_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                full_outer_vels_even[i, j] = extract_surface(firedrake.project(full_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                full_gl_vels_chans[i, j] = extract_surface(firedrake.project(full_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                full_gl_vels_even[i, j] = extract_surface(firedrake.project(full_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                full_ugl_vels_chans[i, j] = extract_surface(firedrake.project(full_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)
                full_ugl_vels_even[i, j] = extract_surface(firedrake.project(full_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)

        inner_dict_even = {}
        inner_dict_chans = {}
        with firedrake.CheckpointFile(inner_fn, "r") as chk:
            fine_mesh = chk.load_mesh(name="fine_mesh")
            for channel_width in widths:
                for channel_depth in depths:
                    key = (channel_width, channel_depth)
                    inner_dict_even[key] = {}
                    inner_dict_chans[key] = {}
                    for f in ["thickness", "velocity", "surface"]:
                        inner_dict_even[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} even".format(channel_width, channel_depth, f),
                        )
                        inner_dict_chans[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} chan".format(channel_width, channel_depth, f),
                        )
            Q2 = inner_dict_even[(widths[0], depths[0])]["thickness"].function_space()
            V2 = inner_dict_even[(widths[0], depths[0])]["velocity"].function_space()

        inner_center_vels_chans = np.empty((len(widths), len(depths)))
        inner_center_vels_even = np.empty((len(widths), len(depths)))
        inner_outer_vels_chans = np.empty((len(widths), len(depths)))
        inner_outer_vels_even = np.empty((len(widths), len(depths)))
        inner_gl_vels_chans = np.empty((len(widths), len(depths)))
        inner_gl_vels_even = np.empty((len(widths), len(depths)))
        inner_ugl_vels_chans = np.empty((len(widths), len(depths)))
        inner_ugl_vels_even = np.empty((len(widths), len(depths)))

        for i, channel_width in enumerate(widths):
            for j, channel_depth in enumerate(depths):
                inner_center_vels_chans[i, j] = extract_surface(firedrake.project(inner_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                inner_center_vels_even[i, j] = extract_surface(firedrake.project(inner_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                inner_outer_vels_chans[i, j] = extract_surface(firedrake.project(inner_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                inner_outer_vels_even[i, j] = extract_surface(firedrake.project(inner_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                inner_gl_vels_chans[i, j] = extract_surface(firedrake.project(inner_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                inner_gl_vels_even[i, j] = extract_surface(firedrake.project(inner_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                inner_ugl_vels_chans[i, j] = extract_surface(firedrake.project(inner_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)
                inner_ugl_vels_even[i, j] = extract_surface(firedrake.project(inner_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)

        outer_dict_even = {}
        outer_dict_chans = {}
        with firedrake.CheckpointFile(outer_fn, "r") as chk:
            fine_mesh = chk.load_mesh(name="fine_mesh")
            for channel_width in widths:
                for channel_depth in depths:
                    key = (channel_width, channel_depth)
                    outer_dict_even[key] = {}
                    outer_dict_chans[key] = {}
                    for f in ["thickness", "velocity", "surface"]:
                        outer_dict_even[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} even".format(channel_width, channel_depth, f),
                        )
                        outer_dict_chans[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} chan".format(channel_width, channel_depth, f),
                        )
            Q2 = outer_dict_even[(widths[0], depths[0])]["thickness"].function_space()
            V2 = outer_dict_even[(widths[0], depths[0])]["velocity"].function_space()

        outer_center_vels_chans = np.empty((len(widths), len(depths)))
        outer_center_vels_even = np.empty((len(widths), len(depths)))
        outer_outer_vels_chans = np.empty((len(widths), len(depths)))
        outer_outer_vels_even = np.empty((len(widths), len(depths)))
        outer_gl_vels_chans = np.empty((len(widths), len(depths)))
        outer_gl_vels_even = np.empty((len(widths), len(depths)))
        outer_ugl_vels_chans = np.empty((len(widths), len(depths)))
        outer_ugl_vels_even = np.empty((len(widths), len(depths)))

        for i, channel_width in enumerate(widths):
            for j, channel_depth in enumerate(depths):
                outer_center_vels_chans[i, j] = extract_surface(firedrake.project(outer_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                outer_center_vels_even[i, j] = extract_surface(firedrake.project(outer_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                outer_outer_vels_chans[i, j] = extract_surface(firedrake.project(outer_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                outer_outer_vels_even[i, j] = extract_surface(firedrake.project(outer_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                outer_gl_vels_chans[i, j] = extract_surface(firedrake.project(outer_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                outer_gl_vels_even[i, j] = extract_surface(firedrake.project(outer_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                outer_ugl_vels_chans[i, j] = extract_surface(firedrake.project(outer_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)
                outer_ugl_vels_even[i, j] = extract_surface(firedrake.project(outer_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)

        margin_dict_even = {}
        margin_dict_chans = {}
        with firedrake.CheckpointFile(margin_fn, "r") as chk:
            fine_mesh = chk.load_mesh(name="fine_mesh")
            for channel_width in widths:
                for channel_depth in depths:
                    key = (channel_width, channel_depth)
                    margin_dict_even[key] = {}
                    margin_dict_chans[key] = {}
                    for f in ["thickness", "velocity", "surface"]:
                        margin_dict_even[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} even".format(channel_width, channel_depth, f),
                        )
                        margin_dict_chans[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} chan".format(channel_width, channel_depth, f),
                        )
            Q2 = margin_dict_even[(widths[0], depths[0])]["thickness"].function_space()
            V2 = margin_dict_even[(widths[0], depths[0])]["velocity"].function_space()

        margin_center_vels_chans = np.empty((len(widths), len(depths)))
        margin_center_vels_even = np.empty((len(widths), len(depths)))
        margin_outer_vels_chans = np.empty((len(widths), len(depths)))
        margin_outer_vels_even = np.empty((len(widths), len(depths)))
        margin_gl_vels_chans = np.empty((len(widths), len(depths)))
        margin_gl_vels_even = np.empty((len(widths), len(depths)))
        margin_ugl_vels_chans = np.empty((len(widths), len(depths)))
        margin_ugl_vels_even = np.empty((len(widths), len(depths)))

        for i, channel_width in enumerate(widths):
            for j, channel_depth in enumerate(depths):
                margin_center_vels_chans[i, j] = extract_surface(firedrake.project(margin_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                margin_center_vels_even[i, j] = extract_surface(firedrake.project(margin_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                margin_outer_vels_chans[i, j] = extract_surface(firedrake.project(margin_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                margin_outer_vels_even[i, j] = extract_surface(firedrake.project(margin_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                margin_gl_vels_chans[i, j] = extract_surface(firedrake.project(margin_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                margin_gl_vels_even[i, j] = extract_surface(firedrake.project(margin_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                margin_ugl_vels_chans[i, j] = extract_surface(firedrake.project(margin_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)
                margin_ugl_vels_even[i, j] = extract_surface(firedrake.project(margin_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)

        center_dict_even = {}
        center_dict_chans = {}
        with firedrake.CheckpointFile(center_fn, "r") as chk:
            fine_mesh = chk.load_mesh(name="fine_mesh")
            for channel_width in widths:
                for channel_depth in depths:
                    key = (channel_width, channel_depth)
                    center_dict_even[key] = {}
                    center_dict_chans[key] = {}
                    for f in ["thickness", "velocity", "surface"]:
                        center_dict_even[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} even".format(channel_width, channel_depth, f),
                        )
                        center_dict_chans[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} chan".format(channel_width, channel_depth, f),
                        )
            Q2 = center_dict_even[(widths[0], depths[0])]["thickness"].function_space()
            V2 = center_dict_even[(widths[0], depths[0])]["velocity"].function_space()

        center_center_vels_chans = np.empty((len(widths), len(depths)))
        center_center_vels_even = np.empty((len(widths), len(depths)))
        center_outer_vels_chans = np.empty((len(widths), len(depths)))
        center_outer_vels_even = np.empty((len(widths), len(depths)))
        center_gl_vels_chans = np.empty((len(widths), len(depths)))
        center_gl_vels_even = np.empty((len(widths), len(depths)))
        center_ugl_vels_chans = np.empty((len(widths), len(depths)))
        center_ugl_vels_even = np.empty((len(widths), len(depths)))

        for i, channel_width in enumerate(widths):
            for j, channel_depth in enumerate(depths):
                center_center_vels_chans[i, j] = extract_surface(firedrake.project(center_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                center_center_vels_even[i, j] = extract_surface(firedrake.project(center_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*center_pt)
                center_outer_vels_chans[i, j] = extract_surface(firedrake.project(center_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                center_outer_vels_even[i, j] = extract_surface(firedrake.project(center_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*outer_pt)
                center_gl_vels_chans[i, j] = extract_surface(firedrake.project(center_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                center_gl_vels_even[i, j] = extract_surface(firedrake.project(center_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*gl_pt)
                center_ugl_vels_chans[i, j] = extract_surface(firedrake.project(center_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)
                center_ugl_vels_even[i, j] = extract_surface(firedrake.project(center_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*ugl_pt)

        arrs = [
            full_center_vels_chans,
            full_center_vels_even,
            full_outer_vels_chans,
            full_outer_vels_even,
            full_gl_vels_chans,
            full_gl_vels_even,
            full_ugl_vels_chans,
            full_ugl_vels_even,
            inner_center_vels_chans,
            inner_center_vels_even,
            inner_outer_vels_chans,
            inner_outer_vels_even,
            inner_gl_vels_chans,
            inner_gl_vels_even,
            inner_ugl_vels_chans,
            inner_ugl_vels_even,
            outer_center_vels_chans,
            outer_center_vels_even,
            outer_outer_vels_chans,
            outer_outer_vels_even,
            outer_gl_vels_chans,
            outer_gl_vels_even,
            outer_ugl_vels_chans,
            outer_ugl_vels_even,
            margin_center_vels_chans,
            margin_center_vels_even,
            margin_outer_vels_chans,
            margin_outer_vels_even,
            margin_gl_vels_chans,
            margin_gl_vels_even,
            margin_ugl_vels_chans,
            margin_ugl_vels_even,
            center_center_vels_chans,
            center_center_vels_even,
            center_outer_vels_chans,
            center_outer_vels_even,
            center_gl_vels_chans,
            center_gl_vels_even,
            center_ugl_vels_chans,
            center_ugl_vels_even,
        ]
        outs = {name: arr for name, arr in zip(names, arrs)}

        np.savez(cache_name, **outs)
    else:
        with np.load(cache_name) as data:
            for name in names:
                exec("{:s} = data['{:s}']".format(name, name))

    rel_dd_gl = []
    rel_dd_center = []
    rel_dw_gl = []
    rel_dw_center = []
    for name, key, evengl, changl, eveno, chano in zip(
        ["full", "margin", "center", "inner", "outer"],
        [key_full, key_margin, key_center, key_inner, key_outer],
        [full_gl_vels_even, margin_gl_vels_even, center_gl_vels_even, inner_gl_vels_even, outer_gl_vels_even],
        [full_gl_vels_chans, margin_gl_vels_chans, center_gl_vels_chans, inner_gl_vels_chans, outer_gl_vels_chans],
        [
            full_center_vels_even,
            margin_center_vels_even,
            center_center_vels_even,
            inner_center_vels_even,
            outer_center_vels_even,
        ],
        [
            full_center_vels_chans,
            margin_center_vels_chans,
            center_center_vels_chans,
            inner_center_vels_chans,
            outer_center_vels_chans,
        ],
    ):
        i = np.where(widths == key[0])[0][0]
        j = np.where(depths == key[1])[0][0]
        print("For {:s}, Max du at gl is {:4.1f} m/yr, at center du is {:4.1f} m/yr".format(name, np.max(changl[:, :] - evengl[:, :]), np.max(chano[:, :] - eveno[:, :])))
        print(
            "or {:4.1f}%, {:4.1f}% of control".format(
                100 * (np.max(changl[:, :] - evengl[:, :])) / control_gl,
                100 * (np.max(chano[:, :] - eveno[:, :])) / control_center,
            )
        )
        print("For {:s}, at gl du is {:4.1f} m/yr, at center du is {:4.1f} m/yr".format(name, changl[i, j] - evengl[i, j], chano[i, j] - eveno[i, j]))
        if name == "full":
            full_vals = [changl[i, j] - evengl[i, j], chano[i, j] - eveno[i, j]]
            full_maxes = [np.max(changl[:, :] - evengl[:, :]), np.max(chano[:, :] - eveno[:, :])]
        else:
            print(
                "Compared to full, that is {:4.1f}% at gl, {:4.1f}% at center".format(
                    100.0 * (changl[i, j] - evengl[i, j] - full_vals[0]) / full_vals[0],
                    100.0 * (chano[i, j] - eveno[i, j] - full_vals[1]) / full_vals[1],
                )
            )
            print(
                "or {:4.1f}% of the change at gl, {:4.1f}% at center".format(
                    100.0 * (changl[i, j] - evengl[i, j]) / full_vals[0],
                    100.0 * (chano[i, j] - eveno[i, j]) / full_vals[1],
                )
            )
            print(
                "Maximum changes are {:4.1f}% at gl, {:4.1f}% at center".format(
                    100.0 * np.max(changl[:, :] - evengl[:, :] - full_maxes[0]) / full_maxes[0],
                    100.0 * np.max(chano[:, :] - eveno[:, :] - full_maxes[1]) / full_maxes[1],
                )
            )
            print(
                "or {:4.1f}% of the change at gl, {:4.1f}% at center".format(
                    100.0 * np.max(changl[:, :] - evengl[:, :]) / full_maxes[0],
                    100.0 * np.max(chano[:, :] - eveno[:, :]) / full_maxes[1],
                )
            )

        depth_pairs = [(0, 1), (1, 3)]
        width_pairs = [(1 + i, 4 + 2 * i) for i in range(4)]
        dw_gl = np.hstack([(changl[i2, :] - evengl[i2, :]) / (changl[i1, :] - evengl[i1, :]) for (i1, i2) in width_pairs])
        dw_c = np.hstack([(chano[i2, :] - eveno[i2, :]) / (chano[i1, :] - eveno[i1, :]) for (i1, i2) in width_pairs])
        dd_gl = np.hstack([(changl[:, i2] - evengl[:, i2]) / (changl[:, i1] - evengl[:, i1]) for (i1, i2) in depth_pairs])
        dd_c = np.hstack([(chano[:, i2] - eveno[:, i2]) / (chano[:, i1] - eveno[:, i1]) for (i1, i2) in depth_pairs])
        if name in ["full", "margin"]:
            rel_dd_gl.append(dd_gl)
            rel_dw_gl.append(dw_gl)
            rel_dd_center.append(dd_c)
            rel_dw_center.append(dw_c)

        print(
            "Double width, change u at GL by (min, max, median, mean - 1): {:4.1f}%, {:4.1f}%, {:4.1f}%, {:4.1f}±{:4.1f}%".format(
                np.min(dw_gl - 1) * 100.0,
                np.max(dw_gl - 1) * 100.0,
                np.median(dw_gl - 1) * 100.0,
                np.mean(dw_gl - 1) * 100.0,
                np.std(dw_gl - 1) * 100.0,
            )
        )
        print(
            "Double width, change u at C by(min, max, median, mean - 1): {:4.1f}%, {:4.1f}%, {:4.1f}%, {:4.1f}±{:4.1f}%".format(
                np.min(dw_c - 1) * 100.0,
                np.max(dw_c - 1) * 100.0,
                np.median(dw_c - 1) * 100.0,
                np.mean(dw_c - 1) * 100.0,
                np.std(dw_c - 1) * 100.0,
            )
        )
        print(
            "Double depth, change u at GL by (min, max, median, mean - 1): {:4.1f}%, {:4.1f}%, {:4.1f}%, {:4.1f}±{:4.1f}%".format(
                np.min(dd_gl - 1) * 100.0,
                np.max(dd_gl - 1) * 100.0,
                np.median(dd_gl - 1) * 100.0,
                np.mean(dd_gl - 1) * 100.0,
                np.std(dd_gl - 1) * 100.0,
            )
        )
        print(
            "Double depth, change u at C by (min, max, median, mean - 1): {:4.1f}%, {:4.1f}%, {:4.1f}%, {:4.1f}±{:4.1f}%".format(
                np.min(dd_c - 1) * 100.0,
                np.max(dd_c - 1) * 100.0,
                np.median(dd_c - 1) * 100.0,
                np.mean(dd_c - 1) * 100.0,
                np.std(dw_c - 1) * 100.0,
            )
        )
        print("")

    relative_changes = ((margin_center_vels_chans - margin_center_vels_even) / (full_center_vels_chans - full_center_vels_even)) * 100
    depths_mat = np.repeat(np.atleast_2d(depths), len(widths), axis=0).flatten()
    widths_mat = np.repeat(np.atleast_2d(widths), len(depths), axis=0).T.flatten()
    print("Mean diff 1 vs 2 channels is {:4.1f}±{:4.1f}% at center".format(np.mean(relative_changes), np.std(relative_changes)))
    relative_changes_gl = ((margin_gl_vels_chans - margin_gl_vels_even) / (full_gl_vels_chans - full_gl_vels_even)) * 100
    print("Mean diff 1 vs 2 channels is {:4.1f}±{:4.1f}% at gl".format(np.mean(relative_changes_gl), np.std(relative_changes_gl)))

    relative_single(relative_changes, relative_changes_gl)
    marker_plot_simplified()
    marker_plot()
    marker_plot_unpaired()
    marker_plot_up_simplified()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 5), sharex=True, sharey=True)
    step = 50
    bin_edges = np.arange(-100, 500 + step, step)

    ax1.hist((np.hstack(rel_dd_center) - 1) * 100.0, bin_edges, alpha=0.5, color="C0", label="Double depth")
    ax1.hist((np.hstack(rel_dw_center) - 1) * 100.0, bin_edges, alpha=0.5, color="C1", label="Double width")
    ax1.legend(loc="best")

    ax2.hist((np.hstack(rel_dd_gl) - 1) * 100.0, bin_edges, alpha=0.5, color="C0", label="Double depth")
    ax2.hist((np.hstack(rel_dw_gl) - 1) * 100.0, bin_edges, alpha=0.5, color="C1", label="Double width")
    ax1.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{a }\fontsize{10pt}{3em}\selectfont{}{Mid shelf}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax1.transAxes,
    )
    ax2.text(
        0.01,
        0.97,
        r"\fontsize{14pt}{3em}\selectfont{}{b }\fontsize{10pt}{3em}\selectfont{}{At Grounding line}",
        ha="left",
        va="top",
        fontsize=12,
        transform=ax2.transAxes,
    )
    ax1.set_xlim(-100, 500)
    fig.supxlabel(r"Relative speed change (\%)", fontsize=10)
    fig.supylabel("Frequency", fontsize=10)
    fig.tight_layout(pad=0.6)
    fig.savefig("../plots/hybrid/dd_vs_dw_hist.pdf")
