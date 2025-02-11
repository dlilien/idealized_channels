#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2024 dlilien <dlilien@noatak.local>
#
# Distributed under terms of the MIT license.

"""

"""
import string
import matplotlib.pyplot as plt
from matplotlib import gridspec
from firedrake.pyplot import tripcolor, tricontour
import firedrake
from firedrake.__future__ import interpolate
from icepack.models.viscosity import sym_grad
from icepackaccs import extract_surface
from icepack.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
)

from hybrid_channels import standard_x, inner_end_x, outer_x, central_y, twostream_C
from libchannels import smooth_floating, ts_name

params = {"text.usetex": "true", "font.family": "sans-serif", "font.sans-serif": "cmss", 'mathtext.fontset': 'custom', 'mathtext.rm': 'Bitstream Vera Sans', 'mathtext.it': 'Bitstream Vera Sans:italic', 'mathtext.bf': 'Bitstream Vera Sans:bold', 'text.latex.preamble': r'\usepackage{cmbright}'}
plt.rcParams.update(params)

width = 3.0e3


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

fields_ts = {}
with firedrake.CheckpointFile("../modeling/outputs/{:s}-fine-degree2_comp.h5".format(ts_name), "r") as chk:
    fine_mesh_ts = chk.load_mesh(name="fine_mesh")
    for key in field_names:
        fields_ts[key] = chk.load_function(fine_mesh_ts, key)
    fields_ts["stress"] = chk.load_function(fine_mesh_ts, "tvm")
height_above_flotation_ts = extract_surface(
    firedrake.assemble(
        interpolate(fields_ts["surface"] - (1 - ρ_I / ρ_W) * fields_ts["thickness"], fields_ts["thickness"].function_space())
    )
)

Q2_ts = firedrake.FunctionSpace(fine_mesh_ts, "CG", 2, vfamily="GL", vdegree=2)
Q2_mismip = firedrake.FunctionSpace(fine_mesh_mismip, "CG", 2, vfamily="GL", vdegree=2)

is_floating_mismip = smooth_floating(250, extract_surface(fields_mismip["surface"]), extract_surface(fields_mismip["thickness"]))
is_floating_ts = smooth_floating(250, extract_surface(fields_ts["surface"]), extract_surface(fields_ts["thickness"]))

fig, ax = plt.subplots()
tricontour(is_floating_mismip, levels=[0.0], colors="k", axes=ax)
tricontour(is_floating_ts, levels=[0.0], colors="0.7", axes=ax)

vm_cutoff = 265  # 265  # from Grinsted et al.
tau_vm_mismip = extract_surface(fields_mismip["stress"])
tau_vm_ts = extract_surface(fields_ts["stress"])

if False:
    colors_dH = tripcolor(
        extract_surface(
            firedrake.assemble(
                interpolate(
                    fields_mismip["thickness"] - firedrake.assemble(interpolate(fields_ts["thickness"], Q2_mismip)), Q2_mismip
                )
            )
        ),
        cmap="PuOr",
        vmin=-500,
        vmax=500,
    )
    plt.colorbar(colors_dH, extend="both")
    plt.title("MISMIP+ - TwoStream")

    colors_dH = tripcolor(
        extract_surface(fields_ts["dH"]),
        cmap="PuOr",
        vmin=-0.01,
        vmax=0.01,
    )
    plt.colorbar(colors_dH, extend="both")
    plt.title("TwoStream dH/dt")

    colors_du = tripcolor(
        extract_surface(firedrake.assemble(interpolate(fields_ts["velocity"][1], Q2_ts))),
        cmap="bwr",
        vmin=-25.0,
        vmax=25.0,
    )
    plt.colorbar(colors_du, extend="both")
    plt.title(ts_name + r" $u_y$")

    C = twostream_C(fine_mesh_ts, fields_ts["thickness"].function_space())

gs = gridspec.GridSpec(
    5,
    3,
    hspace=0.15,
    wspace=0.16,
    width_ratios=(1, 1, 0.03),
    height_ratios=(1, 1, 1, 0.32, 2.10),
    right=0.90,
    top=0.940,
    bottom=0.09,
    left=0.07,
)
fig = plt.figure(figsize=(7, 4.98))
ax_mismip_H = fig.add_subplot(gs[0, 0])
ax_mismip_u = fig.add_subplot(gs[1, 0])
ax_mismip_eps = fig.add_subplot(gs[2, 0])
ax_chan = fig.add_subplot(gs[4, :-1])

ax_ts_H = fig.add_subplot(gs[0, 1])
ax_ts_u = fig.add_subplot(gs[1, 1])
ax_ts_eps = fig.add_subplot(gs[2, 1])

cax_H = fig.add_subplot(gs[0, 2])
cax_u = fig.add_subplot(gs[1, 2])
cax_eps = fig.add_subplot(gs[2, 2])

colors_H = tripcolor(
    extract_surface(fields_mismip["thickness"]),
    axes=ax_mismip_H,
    cmap="viridis",
    vmin=0,
    vmax=1500,
)
colors_u = tripcolor(
    extract_surface(firedrake.project(fields_mismip["velocity"][0], Q2_mismip)),
    axes=ax_mismip_u,
    cmap="Reds",
    vmin=0,
    vmax=1000,
)
colors_H = tripcolor(
    extract_surface(fields_ts["thickness"]),
    axes=ax_ts_H,
    cmap="viridis",
    vmin=0,
    vmax=1500,
)
colors_u = tripcolor(
    extract_surface(firedrake.project(fields_ts["velocity"][0], Q2_ts)),
    axes=ax_ts_u,
    cmap="Reds",
    vmin=0,
    vmax=1000,
)

epsilon_dot_mismip = sym_grad(fields_mismip["velocity"])
colors_eps = tripcolor(
    extract_surface(firedrake.project(epsilon_dot_mismip[0, 1], Q2_mismip)),
    vmin=-0.05,
    vmax=0.05,
    cmap="PiYG",
    axes=ax_mismip_eps,
)
epsilon_dot_ts = sym_grad(fields_ts["velocity"])
colors_eps = tripcolor(
    extract_surface(firedrake.project(epsilon_dot_ts[0, 1], Q2_ts)), vmin=-0.05, vmax=0.05, cmap="PiYG", axes=ax_ts_eps
)

for i, ax in enumerate([ax_mismip_H, ax_mismip_u, ax_mismip_eps, ax_ts_H, ax_ts_u, ax_ts_eps, ax_chan]):
    ax.axis('equal')
    if ax in (ax_ts_H, ax_ts_u, ax_ts_eps):
        hob = is_floating_ts
        glc = "0.7"
    else:
        hob = is_floating_mismip
        glc = "k"
    if ax not in [ax_chan]:
        tricontour(hob, levels=[0.0], colors=glc, axes=ax)
    ax.set_xlim(3.4e5, 6.4e5)
    ax.set_ylim(0, 8e4)
    ax.set_yticks([0, 4e4, 8e4])
    ax.set_xticks([34e4, 44e4, 54e4, 64e4])
    ax.tick_params(axis="both", which="major", labelsize=10)
    if ax in (ax_mismip_u, ax_chan):
        ax.set_yticklabels(["0", "40", "80"])
        ax.set_ylabel("Distance (km)", fontsize=10)
    else:
        ax.set_yticklabels(["", "", ""])

    if ax in (ax_mismip_eps, ax_ts_eps, ax_chan):
        ax.set_xticklabels(["340", "440", "540", "640"])
        ax.set_xlabel("Distance (km)", fontsize=10)
    else:
        ax.set_xticklabels(["", "", "", ""])
    ax.text(0.01, 0.97, string.ascii_lowercase[i], fontsize=12, ha="left", va="top", transform=ax.transAxes)

tricontour(tau_vm_mismip, levels=[vm_cutoff], colors="0.3", axes=ax_mismip_eps)
tricontour(tau_vm_ts, levels=[vm_cutoff], colors="0.3", axes=ax_ts_eps)

plt.colorbar(colors_H, label="Thickness [m]", extend="max", cax=cax_H, orientation="vertical")
cax_H.tick_params(axis="both", which="major", labelsize=10)

plt.colorbar(colors_u, label="$u_x$ [m yr$^{-1}$]", extend="max", cax=cax_u, orientation="vertical")
cax_u.tick_params(axis="both", which="major", labelsize=10)

cbar_eps = plt.colorbar(
    colors_eps, label=r"$\dot{\epsilon}_{xy}$ [yr$^{-1}$]", extend="both", cax=cax_eps, orientation="vertical"
)
cax_eps.tick_params(axis="both", which="major", labelsize=10)
ax_mismip_H.set_title("MISMIP+")
ax_ts_H.set_title(ts_name.replace("_", " ").capitalize())


tricontour(is_floating_mismip, levels=[0.0], colors="k", axes=ax_chan)
tricontour(is_floating_ts, levels=[0.0], colors="0.7", axes=ax_chan)


def plot_chan(ax, inc, outc, w, ll, r=None, **kwargs):
    if r is None:
        r = 8e5
    kk = 5.2e5
    x = [r, kk, ll, ll, kk, r, r]
    y = [outc + w, outc + w, inc + w, inc, outc, outc, outc + w]
    ax.plot(x, y, **kwargs)


def plot_central_chan(ax, c, w, ll, r=None, **kwargs):
    if r is None:
        r = 8e5
    x = [r, ll, ll, r, r]
    y = [c + w / 2, c + w / 2, c - w / 2, c - w / 2, c + w / 2]
    ax.plot(x, y, **kwargs)


ax_mismip_H.annotate('Grounding\nline', xy=(51.8e4, 6.8e4), xytext=(55e4, 3e4), arrowprops=dict(arrowstyle="->"))
ax_ts_H.annotate('Grounding\nline', xy=(51.5e4, 6.8e4), xytext=(55e4, 3e4), arrowprops=dict(arrowstyle="->", color='0.7'), color='0.7')

ax_ts_eps.annotate(r"$\tau_{\textnormal{vM}}$=265 kPa", xy=(49e4, 6.3e4), xytext=(51e4, 3e4), arrowprops=dict(arrowstyle="->", color="0.3"), color="0.3")

standard_x -= 5e3
inc = 2.15e4 + 0.032e4  # This is where the taper ends
outc = 1.55e4
# ax_chan.plot([], [], color="k", label="MISMIP+ GL")
# ax_chan.plot([], [], color="0.7", label=ts_name.replace("_", " ").capitalize() + " GL")
# ax_chan.plot([], [], color="0.3", label=r"$\tau_{\textrm{vM}}$=265 kPa")
# ax_chan.plot([], [], color="0.3", label=ts_name + r" $\tau_{\textrm{vM}}$=265 kPa")
plot_chan(ax_chan, inc, outc, width, standard_x, color="C0", lw=3, label="Two full channels")
plot_chan(ax_chan, 8e4 - inc, 8e4 - outc, -width, standard_x, color="C0", lw=3)
plot_chan(ax_chan, inc, outc, width, standard_x, color="C1", lw=3, linestyle="dashed", label="Margin channel")
plot_central_chan(ax_chan, central_y, width, standard_x, color="C2", lw=2, linestyle="solid", label="Central channel")
plot_chan(ax_chan, inc, outc, width, standard_x, r=inner_end_x, lw=2, color="C3", linestyle="dashdot", label="Inner channels")
plot_chan(ax_chan, 8e4 - inc, 8e4 - outc, -width, standard_x, r=inner_end_x, lw=2, color="C3", linestyle="dashdot")
plot_chan(ax_chan, outc, outc, width, outer_x, color="C6", linestyle="dotted", lw=2, label="Outer channels")
plot_chan(ax_chan, 8e4 - outc, 8e4 - outc, -width, outer_x, color="C6", lw=3, linestyle="dotted")

ax_chan.legend(loc="lower left", frameon=False, fontsize=10)
fig.savefig("../plots/setup_overall.png", dpi=300)

plt.show()
