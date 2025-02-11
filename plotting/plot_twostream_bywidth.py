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

# Modeling packages
import firedrake

# my packages
from icepackaccs import extract_surface

from hybrid_channels import widths, depths
from plot_mismip_raster import key_full, key_margin, key_center, key_inner, key_outer
from libchannels import ts_name

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

key_funky = key_margin

depths = np.array(depths)
widths = np.array(widths)

# center, outerr, outerl, gl, lo, ro, ri, ro

pt_names = ["Mid-shelf", "Right margin", "Left margin", "GL", "Outer left", "Outer right", "Left GL", "Right", "UGL", "Left UGL", "Right UGL"]
all_pts = [(630000, 40000), (630000, 10000), (630000, 70000), (432683, 40000), (630000, 50000), (630000, 30000), (425070, 50000), (438794, 30000), (422683, 40000), (415070, 50000), (428794, 30000)]
pt_symbols = "o+xs^v<>312"


field_names = ["velocity", "thickness", "surface"]


def twopanel(control_pts, full_chans, full_even, margin_chans, margin_even, funky_chans, funky_even, center_chans, center_even, pt_inds=[0, 3]):
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
            for pt_ind, ax in zip(pt_inds, [ax1, ax2]):
                ax.plot(
                    depths,
                    full_chans[pt_ind, i, :] - full_even[pt_ind, i, :],
                    linestyle="none",
                    marker=pt_symbols[pt_ind],
                    markeredgecolor="C{:d}".format(lines + 4),
                    markerfacecolor="k",
                )
                ax.plot(depths, center_chans[pt_ind, i, :] - center_even[pt_ind, i, :], linestyle="none", marker=pt_symbols[pt_ind], markeredgecolor="C{:d}".format(lines + 4), markerfacecolor="k", markerfacecoloralt="w", fillstyle="top")
                ax.plot(
                    depths + 3, margin_chans[pt_ind, i, :] - margin_even[pt_ind, i, :], linestyle="none", marker=pt_symbols[pt_ind], markeredgecolor="C{:d}".format(lines + 4), markerfacecolor="k", markerfacecoloralt="w", fillstyle="right"
                )
                ax.plot(depths - 3, funky_chans[pt_ind, i, :] - funky_even[pt_ind, i, :], linestyle="none", marker=pt_symbols[pt_ind], markeredgecolor="C{:d}".format(lines + 4), markerfacecolor="k", markerfacecoloralt="w", fillstyle="left")

            ax1.plot([], [], linestyle="none", marker="o", color="C{:d}".format(lines + 4), label="{:d} m wide".format(width))

    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.6", markerfacecolor="k", label="Two full marginal")
    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.4", markerfacecolor="k", markerfacecoloralt="w", fillstyle="top", label="One central")
    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.4", markerfacecolor="k", markerfacecoloralt="w", fillstyle="right", label="Right marginal")
    ax2.plot([], [], linestyle="none", marker="s", markeredgecolor="0.4", markerfacecolor="k", markerfacecoloralt="w", fillstyle="left", label="Left marginal")
    # ax2.plot([], [], linestyle='none', marker='o', color='k', label='Shelf center')
    # ax.plot([], [], linestyle='none', marker='s', color='k', label='Shelf edge')
    # ax2.plot([], [], linestyle='none', marker='s', color='k', label='At GL')
    ax1.set_ylim(0, 1500)
    ax2.set_ylim(0, 300)

    ax1_perc = ax1.twinx()
    ax1_perc.set_ylim(0, 100 * ax1.get_ylim()[1] / control_pts[pt_inds[0]])
    ax2_perc = ax2.twinx()
    ax2_perc.set_ylim(0, 100 * ax2.get_ylim()[1] / control_pts[pt_inds[1]])

    ax1.text(0.01, 0.97, r"\fontsize{14pt}{3em}\selectfont{}{a }\fontsize{10pt}{3em}\selectfont{}{Mid shelf}", ha="left", va="top", fontsize=12, transform=ax1.transAxes)
    ax2.text(0.01, 0.97, r"\fontsize{14pt}{3em}\selectfont{}{b }\fontsize{10pt}{3em}\selectfont{}{At Grounding line}", ha="left", va="top", fontsize=12, transform=ax2.transAxes)
    ax2.set_xlabel("Channel depth [m]")
    ax1.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax2.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax1_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax2_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax1.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    ax2.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout(pad=0.1)
    fig.savefig("../plots/hybrid/comp_single_ts.pdf")


def twopanel_simplified(control_pts, full_chans, full_even, margin_chans, margin_even, funky_chans, funky_even, center_chans, center_even, pt_inds=[0, 3]):
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(7, 3.25),
        gridspec_kw={"hspace": 0.1, "right": 0.910, "top": 0.98, "left": 0.095, "bottom": 0.13},
    )
    pwidth = 2

    colors = {"full": "C0", "inner": "C3", "outer": "C6", "margin": "C1", "center": "C2", "funky": "C5"}
    names = {"full": "Two full marginal", "inner": "Two inner marginal", "outer": "Two outer marginal", "margin": "Right marginal", "center": "One central", "funky": "Left marginal"}
    for pref in ["full", "margin", "funky", "center"]:
        d = eval("{:s}_chans - {:s}_even".format(pref, pref))
        for pt_ind, ax in zip(pt_inds, [ax1, ax2]):
            for j, depth in enumerate(depths):
                label = None
                if j == 0:
                    label = names[pref]
                ax.fill([depth - pwidth, depth + pwidth, depth + pwidth, depth - pwidth, depth - pwidth], [d[pt_ind, 0, j], d[pt_ind, 0, j], d[pt_ind, -1, j], d[pt_ind, -1, j], d[pt_ind, 0, j]], color=colors[pref], label=label, alpha=0.5)
    ax1.set_ylim(0, 1500)
    ax2.set_ylim(0, 300)

    ax1_perc = ax1.twinx()
    ax1_perc.set_ylim(0, 100 * ax1.get_ylim()[1] / control_pts[pt_inds[0]])
    ax2_perc = ax2.twinx()
    ax2_perc.set_ylim(0, 100 * ax2.get_ylim()[1] / control_pts[pt_inds[1]])

    ax1.text(0.01, 0.97, r"\fontsize{14pt}{3em}\selectfont{}{a }\fontsize{10pt}{3em}\selectfont{}{Mid shelf}", ha="left", va="top", fontsize=12, transform=ax1.transAxes)
    ax2.text(0.01, 0.97, r"\fontsize{14pt}{3em}\selectfont{}{b }\fontsize{10pt}{3em}\selectfont{}{At Grounding line}", ha="left", va="top", fontsize=12, transform=ax2.transAxes)
    ax2.set_xlabel("Channel depth [m]")
    ax1.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax2.set_ylabel(r"$\Delta u_x$ [m yr$^{-1}$]")
    ax1_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax2_perc.set_ylabel(r"Relative $\Delta u_x$ [\%]")
    ax1.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout(pad=0.1)
    fig.savefig("../plots/hybrid/comp_single_ts_simp.pdf")


if __name__ == "__main__":
    fields_3 = {}
    with firedrake.CheckpointFile("../modeling/outputs/partial_stream-fine-degree2_comp.h5", "r") as chk:
        fine_mesh = chk.load_mesh(name="fine_mesh")
        for key in field_names:
            fields_3[key] = chk.load_function(fine_mesh, key)

    Q2 = fields_3["thickness"].function_space()
    V2 = fields_3["velocity"].function_space()

    u_0c = firedrake.project(fields_3["velocity"], V2)
    u_0xc = firedrake.project(u_0c[0], Q2)

    control_pts = np.array([extract_surface(u_0xc).at(*pt) for pt in all_pts])
    chan_fn_template = "../modeling/outputs/{:s}-hybrid-{:s}-{:s}.h5"
    setup = ts_name
    pos_name = "channels"
    full_fn = chan_fn_template.format(setup, pos_name, "full")
    inner_fn = chan_fn_template.format(setup, pos_name, "inner")
    outer_fn = chan_fn_template.format(setup, pos_name, "outer")
    center_fn = chan_fn_template.format(setup, "center", "full")
    margin_fn = chan_fn_template.format(setup, "margin", "full")
    funky_fn = chan_fn_template.format(setup, "funky", "full")
    names = ["full_chans", "full_even", "inner_chans", "inner_even", "outer_chans", "outer_even", "margin_chans", "margin_even", "funky_chans", "funky_even", "center_chans", "center_even"]

    cache_name = "pointwise_partial_stream.npz"
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

        full_chans = np.empty((len(all_pts), len(widths), len(depths)))
        full_even = np.empty((len(all_pts), len(widths), len(depths)))

        for k, pt in enumerate(all_pts):
            for i, channel_width in enumerate(widths):
                for j, channel_depth in enumerate(depths):
                    full_chans[k, i, j] = extract_surface(firedrake.project(full_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)
                    full_even[k, i, j] = extract_surface(firedrake.project(full_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)

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

        outer_chans = np.empty((len(all_pts), len(widths), len(depths)))
        outer_even = np.empty((len(all_pts), len(widths), len(depths)))

        for k, pt in enumerate(all_pts):
            for i, channel_width in enumerate(widths):
                for j, channel_depth in enumerate(depths):
                    outer_chans[k, i, j] = extract_surface(firedrake.project(outer_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)
                    outer_even[k, i, j] = extract_surface(firedrake.project(outer_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)

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

        inner_chans = np.empty((len(all_pts), len(widths), len(depths)))
        inner_even = np.empty((len(all_pts), len(widths), len(depths)))

        for k, pt in enumerate(all_pts):
            for i, channel_width in enumerate(widths):
                for j, channel_depth in enumerate(depths):
                    inner_chans[k, i, j] = extract_surface(firedrake.project(inner_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)
                    inner_even[k, i, j] = extract_surface(firedrake.project(inner_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)

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

        margin_chans = np.empty((len(all_pts), len(widths), len(depths)))
        margin_even = np.empty((len(all_pts), len(widths), len(depths)))

        for k, pt in enumerate(all_pts):
            for i, channel_width in enumerate(widths):
                for j, channel_depth in enumerate(depths):
                    margin_chans[k, i, j] = extract_surface(firedrake.project(margin_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)
                    margin_even[k, i, j] = extract_surface(firedrake.project(margin_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)

        funky_dict_even = {}
        funky_dict_chans = {}
        with firedrake.CheckpointFile(funky_fn, "r") as chk:
            fine_mesh = chk.load_mesh(name="fine_mesh")
            for channel_width in widths:
                for channel_depth in depths:
                    key = (channel_width, channel_depth)
                    funky_dict_even[key] = {}
                    funky_dict_chans[key] = {}
                    for f in ["thickness", "velocity", "surface"]:
                        funky_dict_even[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} even".format(channel_width, channel_depth, f),
                        )
                        funky_dict_chans[key][f] = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} {:s} chan".format(channel_width, channel_depth, f),
                        )
            Q2 = funky_dict_even[(widths[0], depths[0])]["thickness"].function_space()
            V2 = funky_dict_even[(widths[0], depths[0])]["velocity"].function_space()

        funky_chans = np.empty((len(all_pts), len(widths), len(depths)))
        funky_even = np.empty((len(all_pts), len(widths), len(depths)))

        for k, pt in enumerate(all_pts):
            for i, channel_width in enumerate(widths):
                for j, channel_depth in enumerate(depths):
                    funky_chans[k, i, j] = extract_surface(firedrake.project(funky_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)
                    funky_even[k, i, j] = extract_surface(firedrake.project(funky_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)

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

        center_chans = np.empty((len(all_pts), len(widths), len(depths)))
        center_even = np.empty((len(all_pts), len(widths), len(depths)))

        for k, pt in enumerate(all_pts):
            for i, channel_width in enumerate(widths):
                for j, channel_depth in enumerate(depths):
                    center_chans[k, i, j] = extract_surface(firedrake.project(center_dict_chans[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)
                    center_even[k, i, j] = extract_surface(firedrake.project(center_dict_even[(channel_width, channel_depth)]["velocity"][0], Q2)).at(*pt)

        arrs = [full_chans, full_even, inner_chans, inner_even, outer_chans, outer_even, margin_chans, margin_even, funky_chans, funky_even, center_chans, center_even]
        outs = {name: arr for name, arr in zip(names, arrs)}

        np.savez(cache_name, **outs)
    else:
        with np.load(cache_name) as data:
            print(data)
            for name in names:
                exec("""{:s} = data['{:s}']""".format(name, name))

    for name, key, even, chan in zip(
        ["full", "margin", "funky", "center", "inner", "outer"],
        [key_full, key_margin, key_funky, key_center, key_inner, key_outer],
        [full_even, margin_even, funky_even, center_even, inner_even, outer_even],
        [full_chans, margin_chans, funky_chans, center_chans, inner_chans, outer_chans],
    ):
        i = np.where(widths == key[0])[0][0]
        j = np.where(depths == key[1])[0][0]
        print("For {:s}: Max du ".format(name), end="")
        print(", ".join(["at {:s} is {:4.1f} m/yr".format(locname, np.max(chan[k, :, :] - even[k, :, :])) for k, locname in enumerate(pt_names)]))
        print("or ", end="")
        print(", ".join(["{:4.1f}% at {:s}".format(np.max(chan[k, :, :] - even[k, :, :]) / control_pts[k] * 100, locname) for k, locname in enumerate(pt_names)]))
        print("With equal volume, du  ", end="")
        print(", ".join(["at {:s} is {:4.1f} m/yr".format(locname, np.max(chan[k, i, j] - even[k, i, j])) for k, locname in enumerate(pt_names)]))
        if name == "full":
            full_vals = [chan[k, i, j] - even[k, i, j] for k in range(len(pt_names))]
            full_max = [np.max(chan[k, :, :] - even[k, :, :]) for k in range(len(pt_names))]
        else:
            print("Compared to full, that is ", end="")
            print(", ".join(["{:4.1f}% of the change at {:s}".format(100.0 * (chan[k, i, j] - even[k, i, j]) / full_vals[k], locname) for k, locname in enumerate(pt_names)]))
            print("Maximum changes are ", end="")
            print(", ".join(["{:4.1f}% at {:s}".format(100.0 * np.max(chan[k, :, :] - even[k, :, :]) / full_max[k], locname) for k, locname in enumerate(pt_names)]))

        depth_pairs = [(0, 1), (1, 3)]
        width_pairs = [(1 + i, 4 + 2 * i) for i in range(4)]
        for k, locname in enumerate(pt_names):
            dw = np.hstack([(chan[k, i2, :] - even[k, i2, :]) / (chan[k, i1, :] - even[k, i1, :]) for (i1, i2) in width_pairs])
            dd = np.hstack([(chan[k, :, i2] - even[k, :, i2]) / (chan[k, :, i1] - even[k, :, i1]) for (i1, i2) in depth_pairs])
            print(
                "Double width, change u at {:s} by (min, max, median, mean - 1): {:4.1f}%, {:4.1f}%, {:4.1f}%, {:4.1f}±{:4.1f}%".format(
                    locname, np.min(dw - 1) * 100.0, np.max(dw - 1) * 100.0, np.median(dw - 1) * 100.0, np.mean(dw - 1) * 100.0, np.std(dw - 1) * 100.0
                )
            )
            print(
                "Double depth, change u at {:s} by (min, max, median, mean - 1): {:4.1f}%, {:4.1f}%, {:4.1f}%, {:4.1f}±{:4.1f}%".format(
                    locname, np.min(dd - 1) * 100.0, np.max(dd - 1) * 100.0, np.median(dd - 1) * 100.0, np.mean(dd - 1) * 100.0, np.std(dd - 1) * 100.0
                )
            )
        print("")

    twopanel(control_pts, full_chans, full_even, margin_chans, margin_even, funky_chans, funky_even, center_chans, center_even)
    twopanel_simplified(control_pts, full_chans, full_even, margin_chans, margin_even, funky_chans, funky_even, center_chans, center_even)
