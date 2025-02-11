#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 dlilien <dlilien@noatak.local>
#
# Distributed under terms of the MIT license.

"""

"""
# Basic packaages
import numpy as np
import matplotlib.pyplot as plt

# Modeling packages
import firedrake

# my packages
from icepackaccs import extract_surface

from hybrid_channels import widths, depths

params = {"text.usetex": "true", "font.family": "sans-serif", "font.sans-serif": "cmss", 'mathtext.fontset': 'custom', 'mathtext.rm': 'Bitstream Vera Sans', 'mathtext.it': 'Bitstream Vera Sans:italic', 'mathtext.bf': 'Bitstream Vera Sans:bold', 'text.latex.preamble': r'\usepackage{cmbright}'}
plt.rcParams.update(params)

cutoff = 265.0
depths = np.array([0] + depths)
widths = np.array(widths)

center_pt = (630000, 40000)
outer_pt = (630000, 10000)
gl_pt = (425000, 40000)

levels = [5]

chan_fn_template = "../modeling/outputs/{:s}-hybrid-{:s}-{:s}.h5"

with firedrake.CheckpointFile("../modeling/outputs/mismip-fine-degree2_comp.h5", "r") as chk:
    fine_mesh = chk.load_mesh(name="fine_mesh")
    tvm_control = chk.load_function(fine_mesh, "tvm")
# Q2 = fields_3["tvm"].function_space()
control_area = firedrake.assemble((extract_surface(tvm_control) > cutoff) * firedrake.dx) / 1.0e6

mismip_names = ["Two full marginal", "Two inner marginal", "Two outer marginal"]
mismip_pos = ["full", "inner", "outer"]
mismip_loc = ["channels"]
ts_loc = ["channels", "margin", "funky", "center"]
ts_names = ["Two full marginal", "Right margin only", "Left margin only", "Central only"]
ts_pos = ["full"]
tvm_dict = {
    "mismip": {loc: {pos: np.empty((len(widths), len(depths))) for pos in mismip_pos} for loc in mismip_loc},
    "partial_stream": {loc: {pos: np.empty((len(widths), len(depths))) for pos in ts_pos} for loc in ts_loc},
}

for setup, sdict in tvm_dict.items():
    for loc, ldict in sdict.items():
        for pos, pmat in ldict.items():
            pmat[:, 0] = control_area
            fn = chan_fn_template.format(setup, loc, pos)
            with firedrake.CheckpointFile(fn, "r") as chk:
                fine_mesh = chk.load_mesh(name="fine_mesh")
                for i, channel_width in enumerate(widths):
                    for j, channel_depth in enumerate(depths[1:]):
                        key = (channel_width, channel_depth)
                        tvm = chk.load_function(
                            fine_mesh,
                            "{:d} {:d} tvm chan".format(channel_width, channel_depth),
                        )
                        Q2 = tvm.function_space()
                        pmat[i, j + 1] = firedrake.assemble((extract_surface(tvm) > cutoff) * firedrake.dx) / 1.0e6

depth_pairs = [(1, 2), (2, 4)]
width_pairs = [(1 + i, 4 + 2 * i) for i in range(4)]
for name, locdict in tvm_dict.items():
    for locname, posdict in locdict.items():
        for posname, mat in posdict.items():
            print("For", name, locname, posname)
            dw = np.hstack([(mat[i2, :]) / (mat[i1, :]) for (i1, i2) in width_pairs])
            dd = np.hstack([(mat[:, i2]) / (mat[:, i1]) for (i1, i2) in depth_pairs])
            for cname, d in zip(["width", "depth"], [dw, dd]):
                if np.any(np.isfinite(d)):
                    print("Double {:s}, change area by (min, max, median, mean - 1): {:4.1f}%, {:4.1f}%, {:4.1f}%, {:4.1f}±{:4.1f}%".format(cname, np.nanmin(d[np.isfinite(d)] - 1) * 100.0, np.nanmax(d[np.isfinite(d)] - 1) * 100.0, np.nanmedian(d[np.isfinite(d)] - 1) * 100.0, np.nanmean(d[np.isfinite(d)] - 1) * 100.0, np.nanstd(d[np.isfinite(d)] - 1) * 100.0))

circles = True
fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    sharex=True,
    sharey=True,
    figsize=(6.8, 3.25),
    gridspec_kw={"hspace": 0.1, "right": 0.995, "top": 0.98, "left": 0.105, "bottom": 0.13},
)
if circles:
    markers = 'o' * 10
    fcs = [{"markerfacecolor": "k"},
           {"markerfacecolor": "w"},
           {"markerfacecolor": "0.6"},
           {"markerfacecolor": "k"},
           {"markerfacecolor": "k", "markerfacecoloralt": "w", "fillstyle": "right"},
           {"markerfacecolor": "k", "markerfacecoloralt": "w", "fillstyle": "left"},
           {"markerfacecolor": "k", "markerfacecoloralt": "w", "fillstyle": "bottom"}]
else:
    markers = "osdo+xv"
    fcs = [{"markerfacecolor": "C5"}, {"markerfacecolor": "C6"}, {"markerfacecolor": "C7"}]
lines = -1
for k, width in enumerate(widths):
    if width in [500, 2500, 5000]:
        lines += 1
        i = np.where(widths == width)[0][0]
        for ax in [ax1, ax2]:
            ax.plot([], [], marker="o", linestyle="none", color="C{:d}".format(lines + 4), label="{:d} m wide".format(width))

        for j, pos in enumerate(mismip_pos):
            offset = 0
            if j == 1:
                offset = -2
            elif j == 2:
                offset = 2

            if circles:
                fc = fcs[j]
            else:
                fc = fcs[lines - 1]
            ax1.plot(
                depths + offset,
                tvm_dict["mismip"][mismip_loc[0]][pos][i, :],
                linestyle="none",
                marker=markers[j],
                markeredgecolor="C{:d}".format(lines + 4),
                **fc
            )
        for j, loc in enumerate(ts_loc):
            # offset left to the left, right to the right
            if j in [0, 3]:
                offset = 0
            elif j == 1:
                offset = 2
            elif j == 2:
                offset = -2

            if circles:
                fc = fcs[j + 3]
            else:
                fc = fcs[lines - 1]
            ax2.plot(
                depths + offset,
                tvm_dict["partial_stream"][loc][ts_pos[0]][i, :],
                linestyle="none",
                marker=markers[j + 3],
                markeredgecolor="C{:d}".format(lines + 4),
                **fc
            )

for j, pos in enumerate(mismip_pos):
    if circles:
        ax1.plot([], [], label=mismip_names[j], markeredgecolor="0.7", linestyle="none", marker=markers[j], **fcs[j])
    else:
        ax1.plot([], [], label=mismip_names[j], color="k", linestyle="none", marker=markers[j])
for j, loc in enumerate(ts_loc):
    if circles:
        ax2.plot([], [], label=ts_names[j], markeredgecolor="0.7", linestyle="none", marker=markers[j + 3], **fcs[j + 3])
    else:
        ax2.plot([], [], label=ts_names[j], color="k", linestyle="none", marker=markers[j + 3])

ax1.set_ylim(0, 2000)
ax1.set_xlim(40, 260)

ax1.text(0.01, 0.97, "a MISMIP+", ha="left", va="top", fontsize=14, transform=ax1.transAxes)
ax2.text(0.01, 0.97, "b Partial stream", ha="left", va="top", fontsize=14, transform=ax2.transAxes)
ax2.set_xlabel("Channel depth [m]")
ax1.text(
    -0.1, -0.1, r"Area where $\tau_\textnormal{vM}>$265 kPa [km$^2$]", rotation=90, transform=ax1.transAxes, ha="center", va="center"
)
ax1.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
ax2.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
fig.savefig("../plots/hybrid/tvm_bydepth.pdf")


fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    sharex=True,
    figsize=(6.8, 3.25),
    gridspec_kw={"hspace": 0.1, "right": 0.995, "top": 0.98, "left": 0.105, "bottom": 0.13},
)
pwidth = 2

colors = {"full": "C0", "inner": "C3", "outer": "C6", "margin": "C1", "center": "C2", "funky": "C5"}
names = {"full": "Two full marginal", "inner": "Two inner marginal", "outer": "Two outer marginal", "margin": "Right marginal", "center": "One central", "funky": "Left marginal"}

for j, pos in enumerate(mismip_pos):
    for i, depth in enumerate(depths):
        label = None
        if i == 0:
            label = names[pos]
        b, t = np.min(tvm_dict["mismip"][mismip_loc[0]][pos][:, i]), np.max(tvm_dict["mismip"][mismip_loc[0]][pos][:, i])
        ax1.fill([depth - pwidth, depth + pwidth, depth + pwidth, depth - pwidth, depth - pwidth],
                 [b, b, t, t, b],
                 color=colors[pos],
                 label=label,
                 alpha=0.5)

for j, loc in enumerate(ts_loc[1:]):
    for i, depth in enumerate(depths):
        label = None
        if i == 0:
            label = names[loc]
        b, t = np.min(tvm_dict["partial_stream"][loc][ts_pos[0]][:, i]), np.max(tvm_dict["partial_stream"][loc][ts_pos[0]][:, i])
        ax2.fill([depth - pwidth, depth + pwidth, depth + pwidth, depth - pwidth, depth - pwidth],
                 [b, b, t, t, b],
                 color=colors[loc],
                 label=label,
                 alpha=0.5)


ax1.set_ylim(0, 2000)
ax2.set_ylim(0, 1000)
ax1.set_xlim(40, 260)

ax1.text(0.01, 0.97, "a MISMIP+", ha="left", va="top", fontsize=14, transform=ax1.transAxes)
ax2.text(0.01, 0.97, "b Partial stream", ha="left", va="top", fontsize=14, transform=ax2.transAxes)
ax2.set_xlabel("Channel depth [m]")
ax1.text(
    -0.1, -0.1, r"Area where $\tau_\textnormal{vM}>$265 kPa [km$^2$]", rotation=90, transform=ax1.transAxes, ha="center", va="center"
)
ax1.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
ax2.legend(ncol=2, loc="upper center", frameon=False, fontsize=8)
fig.savefig("../plots/hybrid/tvm_bydepth_simp.pdf")
